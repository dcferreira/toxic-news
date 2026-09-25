# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the per-outlet health report in ``toxic_news.health``."""

import datetime
import json
from pathlib import Path

import pytest

from toxic_news.health import (
    OutletHealth,
    Verdict,
    health_path,
    outlet_health,
    read_health,
    reference_sizes,
    write_health,
)
from toxic_news.newspapers import Newspaper, get_xpath_fn

BBC = Newspaper(
    name="BBC",
    language="en",
    url="https://bbc.com",
    expected_headlines=47,
    get_headlines_fn=get_xpath_fn("//h3", href_xpath="a"),
)
DAY = datetime.date(2026, 9, 24)
ONE_DAY = datetime.timedelta(days=1)
PAGE = b"<html><body>" + b"x" * 100_000 + b"</body></html>"


def _health(  # noqa: PLR0913 (mirrors outlet_health's keyword-only inputs)
    *,
    status: int | None = 200,
    fetch_error: str | None = None,
    body: bytes | None = PAGE,
    headlines: int = 45,
    parse_error: str | None = None,
    reference_bytes: int | None = None,
) -> OutletHealth:
    return outlet_health(
        BBC,
        status=status,
        fetch_error=fetch_error,
        body=body,
        headlines=headlines,
        parse_error=parse_error,
        reference_bytes=reference_bytes,
    )


def test_a_fetched_page_with_the_expected_headline_count_is_ok() -> None:
    assert _health().verdict == Verdict.OK


@pytest.mark.parametrize("headlines", [0, 10, 28, 66, 182])
def test_a_fetched_page_with_a_count_outside_the_band_is_an_xpath_error(
    headlines: int,
) -> None:
    # 47 expected: the band is [28.2, 65.8]
    assert _health(headlines=headlines).verdict == Verdict.XPATH


def test_an_extractor_that_raised_on_a_fetched_page_is_an_xpath_error() -> None:
    health = _health(headlines=0, parse_error="IndexError('list index out of range')")
    assert health.verdict == Verdict.XPATH


@pytest.mark.parametrize("status", [401, 403, 404, 500, 503])
def test_a_page_served_with_an_error_status_is_not_an_xpath_error(status: int) -> None:
    assert _health(status=status, headlines=0).verdict == Verdict.OTHER


def test_a_page_that_could_not_be_fetched_is_not_an_xpath_error() -> None:
    health = _health(status=None, fetch_error="TimeoutError()", body=None, headlines=0)
    assert health.verdict == Verdict.OTHER


@pytest.mark.parametrize(
    "body",
    [
        b"<html><head><title>Just a moment...</title></head></html>",
        b"<html><head><title>Access Denied</title></head></html>",
        b'<html><body><div id="px-captcha"></div></body></html>',
        b'<script src="https://ct.captcha-delivery.com/c.js"></script>',
        b'<form id="challenge-form" action="/?__cf_chl_f_tk=abc"></form>',
    ],
)
def test_a_bot_challenge_page_served_with_200_is_not_an_xpath_error(
    body: bytes,
) -> None:
    assert _health(body=body, headlines=0).verdict == Verdict.OTHER


def test_a_page_much_smaller_than_the_last_good_one_is_not_an_xpath_error() -> None:
    health = _health(body=PAGE, headlines=0, reference_bytes=len(PAGE) * 4)
    assert health.verdict == Verdict.OTHER


def test_a_tiny_page_is_not_an_xpath_error_even_without_a_last_good_one() -> None:
    # an outlet broken since before the health report began has no reference
    stub = b"<html><body>Service unavailable</body></html>"
    assert _health(body=stub, headlines=0).verdict == Verdict.OTHER


def test_a_page_about_as_big_as_the_last_good_one_is_an_xpath_error() -> None:
    health = _health(body=PAGE, headlines=0, reference_bytes=len(PAGE) * 2)
    assert health.verdict == Verdict.XPATH


def test_the_report_records_what_was_observed() -> None:
    health = _health(status=403, headlines=0)
    assert health.newspaper == "BBC"
    assert health.url == "https://bbc.com"
    assert health.status == 403
    assert health.headlines == 0
    assert health.expected == 47
    assert health.body_bytes == len(PAGE)


def test_health_is_stored_as_one_json_file_per_day(tmp_path: Path) -> None:
    report = [_health(), _health(status=403, headlines=0)]
    write_health(tmp_path, DAY, report)

    path = health_path(tmp_path, DAY)
    assert path == tmp_path / "health" / "2026" / "09" / "24.json"
    stored = json.loads(path.read_text())
    assert [outlet["verdict"] for outlet in stored] == ["ok", "other"]
    assert read_health(tmp_path, DAY) == report


def test_the_health_of_a_missing_day_is_empty(tmp_path: Path) -> None:
    assert read_health(tmp_path, DAY) == []


def test_reference_sizes_are_the_latest_ok_page_size_per_outlet(
    tmp_path: Path,
) -> None:
    write_health(tmp_path, DAY - 3 * ONE_DAY, [_health(body=b"a" * 100)])
    write_health(tmp_path, DAY - 2 * ONE_DAY, [_health(body=b"a" * 200)])
    # a broken day does not count as a reference
    write_health(tmp_path, DAY - ONE_DAY, [_health(body=b"a" * 5, headlines=0)])

    assert reference_sizes(tmp_path, DAY) == {"BBC": 200}


def test_reference_sizes_only_look_back_a_bounded_number_of_days(
    tmp_path: Path,
) -> None:
    write_health(tmp_path, DAY - 40 * ONE_DAY, [_health(body=b"a" * 100)])
    assert reference_sizes(tmp_path, DAY, lookback_days=30) == {}
