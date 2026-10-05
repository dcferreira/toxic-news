# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Screenshot a saved front page with every XPath match outlined.

The self-fix PR job shows its fix working with a screenshot: canary loads the
saved page with JavaScript off, outlines what the XPath matches and takes a
full-page screenshot. Both lxml and the browser implement XPath 1.0, so the
outlined count should equal the count lxml gets on the same page; `check`
fails when it does not.

`prepare` writes the page, the canary script and lxml's count; canary runs the
script; `check` compares the two counts.
"""

import html
import json
import re
from pathlib import Path
from typing import Annotated

import lxml.html
import typer

from toxic_news.fetchers import clean_url
from toxic_news.newspapers import Newspaper, newspapers

app = typer.Typer()

# blocks every script, inline or not, while keeping <script> elements in the
# DOM, so the browser's tree matches the one lxml parsed
CSP_META = '<meta http-equiv="Content-Security-Policy" content="script-src \'none\'">'
OUTLINE_STYLE = "3px solid #e4007c"
_HEAD = re.compile(r"<head\b[^>]*>", re.IGNORECASE)


def prepare_page(content: str, base_url: str) -> str:
    """Make a saved page load offline: resolve its CSS, never run its scripts.

    The `<base>` and the CSP go first in `<head>`, ahead of any script.
    """
    tags = f'<base href="{html.escape(base_url.rstrip("/") + "/")}">{CSP_META}'
    match = _HEAD.search(content)
    if match is None:
        return tags + content
    return content[: match.end()] + tags + content[match.end() :]


def count_matches(content: str, xpath: str) -> int:
    """Count the nodes `xpath` matches, parsed the way the scraper parses."""
    found = lxml.html.fromstring(content).xpath(xpath)
    if not isinstance(found, list):
        msg = f"{xpath} evaluates to {found!r}, not a node-set"
        raise TypeError(msg)
    return len(found)


def canary_script(page_url: str, xpath: str, screenshot: str, result: str) -> str:
    """Build the canary step that outlines every match and screenshots the page.

    It writes `{"matches": ..., "outlined": ..., "hidden": ...}` to `result` in
    canary's tmp dir; a hidden match has an empty box, so nothing to outline.
    """
    return f"""\
const page = await browser.getPage("evidence");
try {{
  await page.goto({json.dumps(page_url)}, {{ waitUntil: "load", timeout: 60000 }});
}} catch (e) {{
  // the DOM is already parsed; only some images or CSS are still loading
  console.warn("WARN load did not finish: " + e.message);
}}
const counts = await page.evaluate(([xpath, style]) => {{
  const found = document.evaluate(
    xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
  let outlined = 0;
  let hidden = 0;
  for (let i = 0; i < found.snapshotLength; i++) {{
    let node = found.snapshotItem(i);
    if (node.nodeType !== Node.ELEMENT_NODE) node = node.parentElement;
    const rect = node ? node.getBoundingClientRect() : null;
    if (!rect || rect.width === 0 || rect.height === 0) {{
      hidden++;
      continue;
    }}
    // a box over the match, not a style on it, so no ancestor clips it
    const box = document.createElement("div");
    box.style.cssText = "position:absolute;pointer-events:none;"
      + "z-index:2147483647;box-sizing:border-box;border:" + style + ";"
      + "left:" + (rect.left + window.scrollX - 4) + "px;"
      + "top:" + (rect.top + window.scrollY - 4) + "px;"
      + "width:" + (rect.width + 8) + "px;height:" + (rect.height + 8) + "px";
    document.documentElement.appendChild(box);
    outlined++;
  }}
  return {{ matches: found.snapshotLength, outlined, hidden }};
}}, [{json.dumps(xpath)}, {json.dumps(OUTLINE_STYLE)}]);
const shot = await saveScreenshot(
  await page.screenshot({{ fullPage: true }}), {json.dumps(screenshot)});
await writeFile({json.dumps(result)}, JSON.stringify(counts));
console.log(JSON.stringify({{ ...counts, screenshot: shot }}));
"""


def live_script(url: str) -> str:
    """Build the canary step that opens the outlet's live front page."""
    return f"""\
const page = await browser.getPage("live");
await page.goto({json.dumps(url)}, {{ waitUntil: "domcontentloaded", timeout: 60000 }});
console.log(JSON.stringify({{ url: page.url(), title: await page.title() }}));
"""


def find_page(raw_html_dir: Path, outlet: str) -> tuple[Newspaper, Path]:
    """Find the outlet saved as `<outlet>.html`, named like `save_pages` names it."""
    for newspaper in newspapers:
        if clean_url(str(newspaper.url)) == outlet:
            return newspaper, raw_html_dir / f"{outlet}.html"
    msg = f"no outlet is saved as {outlet}.html"
    raise ValueError(msg)


@app.command()
def prepare(
    outlet: Annotated[str, typer.Argument(help="The saved page's file stem.")],
    xpath: Annotated[str, typer.Argument(help="The headline XPath to outline.")],
    raw_html_dir: Annotated[Path, typer.Option(help="The Update run's raw-html.")],
    out_dir: Annotated[Path, typer.Option(help="Where page.html etc. are written.")],
) -> None:
    """Write page.html, the canary steps and lxml.json with lxml's count.

    outline.js screenshots page.html with every match outlined; live.js opens
    the outlet's live front page, for a reference recording.
    """
    newspaper, path = find_page(raw_html_dir, outlet)
    # decoded the way the scraper decodes it
    content = path.read_bytes().decode()
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / "page.html"
    page.write_text(prepare_page(content, str(newspaper.url)), encoding="utf-8")
    script = canary_script(
        page.resolve().as_uri(), xpath, f"{outlet}.png", "outline.json"
    )
    (out_dir / "outline.js").write_text(script)
    (out_dir / "live.js").write_text(live_script(str(newspaper.url)))
    matches = count_matches(content, xpath)
    lxml_result = {"outlet": outlet, "xpath": xpath, "matches": matches}
    (out_dir / "lxml.json").write_text(json.dumps(lxml_result))
    typer.echo(json.dumps(lxml_result))


@app.command()
def check(
    lxml_result: Annotated[Path, typer.Argument(help="lxml.json from `prepare`.")],
    outlined_result: Annotated[Path, typer.Argument(help="The canary step's JSON.")],
) -> None:
    """Fail unless the browser matched what lxml matched, and outlined each one.

    Matches the page hides (an empty box) cannot be outlined; they are counted
    and reported instead.
    """
    lxml_matches = json.loads(lxml_result.read_text())["matches"]
    browser = json.loads(outlined_result.read_text())
    typer.echo(
        f"lxml {lxml_matches}; browser {browser['matches']}, of which "
        f"{browser['outlined']} outlined and {browser['hidden']} hidden"
    )
    if lxml_matches != browser["matches"]:
        raise typer.Exit(code=1)
    if browser["outlined"] + browser["hidden"] != browser["matches"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
