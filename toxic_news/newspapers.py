import urllib.parse
from datetime import datetime
from typing import Callable, Optional, Protocol, runtime_checkable

import lxml.html
from lxml.html import HtmlElement
from pydantic import BaseModel, HttpUrl, validator

HeadlinesFnOut = list[tuple[str, str]]


@runtime_checkable
class HeadlinesFn(Protocol):
    def __call__(
        self,
        content: str,
        base_url: str,
        request_date: datetime,
    ) -> HeadlinesFnOut:
        ...


class Newspaper(BaseModel):
    name: str
    language: str
    url: HttpUrl
    expected_headlines: int
    get_headlines_fn: HeadlinesFn

    @validator("url")
    def normalize_url(cls, v):
        return v.strip("/")

    class Config:
        arbitrary_types_allowed = True

    def get_headlines(self, content: str, request_date: datetime) -> HeadlinesFnOut:
        return self.get_headlines_fn(
            content, base_url=self.url, request_date=request_date
        )


def default_extract_headline(element: HtmlElement) -> str:
    return "".join(element.itertext()).strip(" \t\n\r|")


def extract_text_and_url(
    element: HtmlElement,
    url_xpath: str,
    base_url: str,
    extract_headline_fn: Callable[[HtmlElement], str],
) -> tuple[str, str]:
    title = extract_headline_fn(element)
    res = element.xpath(url_xpath)
    url = res[0].get("href")
    absolute_url = urllib.parse.urljoin(base_url, url)
    return title, absolute_url


def get_xpath_fn(
    headline_xpath: str,
    *,
    href_xpath: str,
    extract_headline_fn: Callable[[HtmlElement], str] = default_extract_headline,
) -> HeadlinesFn:
    """
    :param headline_xpath: Xpath expression to get the headlines in the page.
    :param href_xpath: Xpath expression that will be applied to each headline
    element, and should return its url.
    :param extract_headline_fn: An optional function that takes an element with
    the headline and extracts the text of the headline. By default, all
    text under the element is extracted.
    :return: a function that receives HTML as a string and outputs a list of
    headlines and respective urls.
    """

    # noinspection PyUnusedLocal
    def f(content: str, base_url: str, *args, **kwargs) -> HeadlinesFnOut:
        tree: HtmlElement = lxml.html.fromstring(content)
        headlines = [
            extract_text_and_url(
                x, href_xpath, base_url, extract_headline_fn=extract_headline_fn
            )
            for x in tree.xpath(headline_xpath)
        ]
        # often the same headline appears multiple times in the page
        deduped_headlines = set(headlines)
        # sort by order in which they appear in the page
        return sorted(deduped_headlines, key=lambda x: headlines.index(x))

    return f


class DatedXpath(BaseModel):
    from_date: Optional[datetime]
    headline_xpath: str
    href_xpath: str
    extract_headline_fn: Callable[[HtmlElement], str] = default_extract_headline


def get_dated_xpath_fn(*xpaths: DatedXpath) -> HeadlinesFn:
    sorted_xpaths = sorted(
        xpaths,
        key=lambda x: x.from_date
        if x.from_date is not None
        else datetime.fromtimestamp(0),
        reverse=True,
    )

    def f(content: str, base_url: str, request_date: datetime) -> HeadlinesFnOut:
        for xp in sorted_xpaths:
            if xp.from_date is None or request_date > xp.from_date:
                return get_xpath_fn(
                    xp.headline_xpath,
                    href_xpath=xp.href_xpath,
                    extract_headline_fn=xp.extract_headline_fn,
                )(
                    content,
                    base_url,
                    request_date,
                )
        raise RuntimeError("Something went wrong, should never reach here!")

    return f


def nytimes_fn(content: str, base_url: str, request_date: datetime) -> HeadlinesFnOut:
    ignore_list = {
        "Spelling Bee",
        "The Crossword",
        "Letter Boxed",
        "Tiles",
        "Vertex",
        "Wordle",
    }
    f = get_dated_xpath_fn(
        DatedXpath(
            from_date=None,
            headline_xpath="//a/div[not(@class)]/h3 | "
            "//section[contains(@class, 'story-wrapper')]/a/div/div/div/h3[2] | "
            "//section[contains(@class, 'story-wrapper')]/div/h3",
            href_xpath="ancestor::a",
        ),
        DatedXpath(
            from_date=datetime(2022, 3, 16),
            headline_xpath="//section[contains(@class, 'story-wrapper')]//"
            "h3[contains(@class, 'indicate-hover')]",
            href_xpath="ancestor::a",
        ),
    )
    headlines = f(content, base_url=base_url, request_date=request_date)
    return [x for x in headlines if x[0] not in ignore_list]


newspapers = [
    Newspaper(
        name="BBC",
        language="en",
        url="https://bbc.com",
        get_headlines_fn=get_xpath_fn(
            "//h3[@class='media__title' and a]", href_xpath="a"
        ),
        expected_headlines=47,
    ),
    Newspaper(
        name="Fox News",
        language="en",
        url="https://foxnews.com",
        get_headlines_fn=get_xpath_fn("//article//h3[a]", href_xpath="a"),
        expected_headlines=125,
    ),
    Newspaper(
        name="Newsmax",
        language="en",
        url="https://newsmax.com",
        get_headlines_fn=get_xpath_fn(
            "//div[@id='nmCanvas1Headline']//h1 "
            "| //div[@id='nmLeadStoryContent']/div[@id='nmleadStoryHead']/h2[a] "
            "| //div[@class='nmNewsfrontT ' or @class='nmleadStoryHead']//h2[a]",
            href_xpath="a",
        ),
        expected_headlines=50,
    ),
    Newspaper(
        name="The Washington Times",
        language="en",
        url="https://washingtontimes.com",
        get_headlines_fn=get_xpath_fn(
            "//div[@class='homeholding']//article/h2/a", href_xpath="."
        ),
        expected_headlines=46,
    ),
    Newspaper(
        name="The New York Times",
        language="en",
        url="https://www.nytimes.com/",
        get_headlines_fn=nytimes_fn,
        expected_headlines=67,
    ),
    Newspaper(
        name="New York Post",
        language="en",
        url="https://nypost.com",
        get_headlines_fn=get_xpath_fn(
            "//h2[contains(@class, 'story__headline')][a]", href_xpath="a"
        ),
        expected_headlines=38,
    ),
    Newspaper(
        name="Associated Press",
        language="en",
        url="https://apnews.com",
        get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=datetime(2023, 3, 21),
                headline_xpath="//div[@class='Body']//a[contains(@class, 'headline')]"
                "  /h2"
                "| //article[@class='cards']//h3",
                href_xpath="ancestor::a",
            ),
            DatedXpath(
                from_date=datetime(2022, 3, 2),
                headline_xpath="//div[@class='Body']//a[contains(@class, 'headline')]"
                "  /h2"
                "| //article[@class='cards']//h4",
                href_xpath="ancestor::a",
            ),
            DatedXpath(
                from_date=None,
                headline_xpath="//div[@class='Body']//a[contains(@class, 'headline')]"
                "  /h3"
                "| //article[@class='cards']//h4",
                href_xpath="ancestor::a",
            ),
        ),
        expected_headlines=45,
    ),
    Newspaper(
        name="NBC News",
        language="en",
        url="https://www.nbcnews.com",
        get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=datetime(2022, 4, 20),
                headline_xpath="//div[@class='tease-card__info']"
                "  //*[self::h2 or self::h3]/a"
                "| //*[contains(@class, 'styles_headline')"
                "  and (self::h2 or self::h3)]/a",
                href_xpath=".",
            ),
            DatedXpath(
                from_date=datetime(2022, 4, 16),
                headline_xpath="//div[@class='tease-card__info']"
                "  //span[contains(@class, '__headline')]/parent::a"
                "| //*[contains(@class, 'styles_headline')"
                "  and (self::h2 or self::h3)]/a"
                "| //div[@class='alacarte__content']"
                "  //h2[@class='alacarte__headline']/parent::a",
                href_xpath=".",
            ),
            DatedXpath(
                from_date=None,
                headline_xpath="//div[@class='tease-card__info']"
                "  //span[contains(@class, '__headline')]/parent::a"
                "| //div[@class='alacarte__content']"
                "  //h2[@class='alacarte__headline']/parent::a"
                "| //div[@class='pancake__info']/h3/a",
                href_xpath=".",
            ),
        ),
        expected_headlines=67,
    ),
    Newspaper(
        name="Newsweek",
        language="en",
        url="https://www.newsweek.com",
        get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=datetime(2023, 4, 3),
                headline_xpath="//div[@class='news-title']/a",
                href_xpath=".",
            ),
            DatedXpath(
                from_date=None,
                headline_xpath="//article//*[self::h1 or self::h3 "
                "  or self::h4 or self::li]/a "
                "| //div[@class='title' or @class='info']/a",
                href_xpath=".",
            ),
        ),
        expected_headlines=102,
    ),
    Newspaper(
        name="One America News Network (OAN)",
        language="en",
        url="https://www.oann.com",
        get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=datetime(2022, 9, 20),
                headline_xpath="//div[@class='site']//h2/a[@title]",
                href_xpath=".",
            ),
            DatedXpath(from_date=None, headline_xpath="//h3/a", href_xpath="."),
        ),
        expected_headlines=28,
    ),
    Newspaper(
        name="Reuters",
        language="en",
        url="https://www.reuters.com",
        get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=None,
                headline_xpath="//a[contains(@class, 'heading__')]",
                href_xpath=".",
                extract_headline_fn=lambda x: x.text,
            ),
            DatedXpath(
                from_date=datetime(2022, 4, 22),
                headline_xpath="//a[@data-testid='Heading']",
                href_xpath=".",
                extract_headline_fn=lambda x: x.text,
            ),
            DatedXpath(
                from_date=datetime(2022, 5, 6),
                headline_xpath="//a[@data-testid='Heading' and not(span)]"
                "| //a[@data-testid='Heading']/span[not(@style) and not(@class)]",
                href_xpath="self::a | parent::a",
            ),
        ),
        expected_headlines=49,
    ),
    Newspaper(
        name="The Epoch Times",
        language="en",
        url="https://www.theepochtimes.com",
        get_headlines_fn=get_xpath_fn(
            "//a/*[contains(@class,'title')"
            "  and not(ancestor::*[contains(@class, 'live_video')"
            "  or contains(@class, 'games')])]",
            href_xpath="parent::a",
        ),
        expected_headlines=97,
    ),
    Newspaper(
        name="The Guardian (US)",
        language="en",
        url="https://www.theguardian.com/us",
        get_headlines_fn=get_xpath_fn(
            "//a[contains(@class, 'js-headline-text')]", href_xpath="."
        ),
        expected_headlines=92,
    ),
    Newspaper(
        name="The Hill",
        language="en",
        url="https://thehill.com",
        get_headlines_fn=get_dated_xpath_fn(
            DatedXpath(
                from_date=None, headline_xpath="//h1/a | //h4/a", href_xpath="."
            ),
            DatedXpath(
                from_date=datetime(2022, 3, 31),
                headline_xpath="//div[contains(@class, 'content')]"
                "  /h1[contains(@class, 'headline')"
                "  and not(@data-module-type='thehill-video')]/a[@href]"
                "| //ul[@data-module-type='more-news']/li/a[@href]"
                "| //ol[@data-module-type='most-popular']/li/a[@href]",
                href_xpath=".",
            ),
        ),
        expected_headlines=39,
    ),
    Newspaper(
        name="The Wall Street Journal",
        language="en",
        url="https://www.wsj.com",
        get_headlines_fn=get_xpath_fn(
            "//div[contains(@class, '-headline-')]//*[contains(@class, 'headlineText')"
            "  and not(./span)]"
            "| //div[contains(@class, '-headline-')]//*["
            "  contains(@class, 'headlineText')]/span[not("
            "  contains(@class, 'time-to-read'))]",
            href_xpath="ancestor::a",
        ),
        expected_headlines=66,
    ),
    Newspaper(
        name="Washington Examiner",
        language="en",
        url="https://www.washingtonexaminer.com",
        get_headlines_fn=get_xpath_fn(
            "//*["
            "  (self::div or self::h1 or self::h2 or self::h4 or self::h5 or self::h6)"
            "  and contains(@class, 'title')]/a",
            href_xpath=".",
        ),
        expected_headlines=62,
    ),
    Newspaper(
        name="The Washington Post",
        language="en",
        url="https://www.washingtonpost.com",
        get_headlines_fn=get_xpath_fn(
            "//h2[contains(@class, 'headline')]/a/span", href_xpath="parent::a"
        ),
        expected_headlines=98,
    ),
]
newspapers_dict: dict[HttpUrl, Newspaper] = {n.url: n for n in newspapers}
