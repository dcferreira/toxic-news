# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the outlined-match screenshot helpers in ``toxic_news.outline``."""

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from toxic_news.outline import (
    CSP_META,
    Box,
    app,
    canary_script,
    count_matches,
    find_page,
    live_script,
    outlines_found,
    prepare_page,
)

PAGE = (
    "<html><head><title>t</title><script>document.body.remove()</script></head>"
    "<body><h2 class='story__headline'><a href='/a'>A</a></h2>"
    "<h2 class='story__headline'><a href='/b'>B</a></h2>"
    "<h2 class='other'><a href='/c'>C</a></h2></body></html>"
)
XPATH = "//h2[contains(@class, 'story__headline')][a]"


def test_prepare_page_puts_base_and_csp_first_in_head() -> None:
    out = prepare_page(PAGE, "https://nypost.com")
    head_start = out.index("<head>") + len("<head>")
    expected = f'<base href="https://nypost.com/">{CSP_META}'
    assert out[head_start : head_start + len(expected)] == expected
    # the inline script still comes after the CSP, so the browser never runs it
    assert out.index(CSP_META) < out.index("<script>")


def test_prepare_page_handles_a_head_with_attributes() -> None:
    out = prepare_page('<html><HEAD lang="en"><p>x</p></HEAD></html>', "https://x.com")
    assert out.startswith('<html><HEAD lang="en"><base href="https://x.com/">')


def test_prepare_page_without_head_prepends() -> None:
    out = prepare_page("<p>x</p>", "https://x.com")
    assert out == f'<base href="https://x.com/">{CSP_META}<p>x</p>'


def test_prepare_page_escapes_the_base_url() -> None:
    out = prepare_page("<head></head>", 'https://x.com/"><script>')
    assert '"><script>' not in out


def test_prepare_page_keeps_the_match_count() -> None:
    assert count_matches(prepare_page(PAGE, "https://nypost.com"), XPATH) == 2


def test_count_matches_rejects_a_non_node_set() -> None:
    with pytest.raises(TypeError, match="not a node-set"):
        count_matches(PAGE, "count(//h2)")


def test_count_matches_counts_raw_matches_not_deduped_headlines() -> None:
    page = "<body>" + "<h2 class='story__headline'><a href='/a'>A</a></h2>" * 3
    assert count_matches(page, XPATH) == 3


def test_canary_script_embeds_values_as_json() -> None:
    xpath = """//a[@title="it's"]"""
    script = canary_script("file:///tmp/page.html", xpath, "shot.png", "out.json")
    assert json.dumps(xpath) in script
    assert json.dumps("file:///tmp/page.html") in script
    assert json.dumps("shot.png") in script
    assert json.dumps("out.json") in script
    assert "document.evaluate" in script
    assert "getBoundingClientRect" in script
    assert "fullPage: true" in script


def test_live_script_opens_the_url() -> None:
    script = live_script("https://nypost.com")
    assert 'page.goto("https://nypost.com"' in script


def test_find_page_maps_the_file_stem_to_its_outlet(tmp_path: Path) -> None:
    (tmp_path / "nypost.com.html").write_text(PAGE)
    newspaper, path = find_page(tmp_path, "nypost.com")
    assert newspaper.name == "New York Post"
    assert path == tmp_path / "nypost.com.html"


def test_find_page_rejects_an_unknown_outlet(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no outlet"):
        find_page(tmp_path, "example.com")


def test_prepare_command_writes_page_script_and_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "nypost.com.html").write_text(PAGE)
    out = tmp_path / "out"
    args = ["prepare", "nypost.com", XPATH, "--raw-html-dir", str(raw)]
    result = CliRunner().invoke(app, [*args, "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    lxml_result = json.loads((out / "lxml.json").read_text())
    assert lxml_result == {"outlet": "nypost.com", "xpath": XPATH, "matches": 2}
    assert CSP_META in (out / "page.html").read_text()
    assert (out / "page.html").resolve().as_uri() in (out / "outline.js").read_text()
    assert '"https://nypost.com"' in (out / "live.js").read_text()


@pytest.mark.parametrize(
    ("browser", "exit_code"),
    [
        ({"matches": 2, "outlined": 2, "hidden": 0}, 0),
        # a hidden match has no box to outline, but it still counts
        ({"matches": 2, "outlined": 1, "hidden": 1}, 0),
        ({"matches": 3, "outlined": 3, "hidden": 0}, 1),
        ({"matches": 2, "outlined": 1, "hidden": 0}, 1),
    ],
)
def test_check_command_compares_the_counts(
    tmp_path: Path, browser: dict[str, int], exit_code: int
) -> None:
    lxml_file = tmp_path / "lxml.json"
    lxml_file.write_text(json.dumps({"outlet": "x", "xpath": XPATH, "matches": 2}))
    outlined_file = tmp_path / "outline.json"
    outlined_file.write_text(json.dumps(browser))
    result = CliRunner().invoke(app, ["check", str(lxml_file), str(outlined_file)])
    assert result.exit_code == exit_code, result.output


# --- the outlines, in the pixels -----------------------------------------------

PINK = (0xE4, 0x00, 0x7C)


def _screenshot(tmp_path: Path, drawn: list[Box], border: int = 3) -> Path:
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    for left, top, width, height in drawn:
        draw.rectangle(
            (left, top, left + width - 1, top + height - 1), outline=PINK, width=border
        )
    path = tmp_path / "shot.png"
    image.save(path)
    return path


def test_outlines_found_counts_only_boxes_drawn_in_the_pixels(tmp_path: Path) -> None:
    drawn: list[Box] = [(10, 10, 100, 30), (10, 60, 100, 30)]
    shot = _screenshot(tmp_path, drawn)
    # the third box lies past the bottom of the screenshot, as a clipped
    # full-page capture leaves it; the fourth was never painted
    boxes = [*drawn, (10, 290, 100, 30), (200, 10, 100, 30)]
    assert outlines_found(shot, boxes, scale=1) == [True, True, False, False]


def test_an_outline_cut_by_the_left_edge_still_counts(tmp_path: Path) -> None:
    # matches at x=0 get their outline 4px to the left, off the page
    shot = _screenshot(tmp_path, [(-4, 10, 100, 30)])
    assert outlines_found(shot, [(-4, 10, 100, 30)], scale=1) == [True]


def test_boxes_are_scaled_by_the_device_pixel_ratio(tmp_path: Path) -> None:
    shot = _screenshot(tmp_path, [(20, 20, 200, 60)], border=6)
    assert outlines_found(shot, [(10, 10, 100, 30)], scale=2) == [True]


def test_verify_command_writes_what_it_found(tmp_path: Path) -> None:
    shot = _screenshot(tmp_path, [(10, 10, 100, 30)])
    outlined = tmp_path / "outline.json"
    boxes = [[10, 10, 100, 30], [200, 10, 100, 30]]
    outlined.write_text(
        json.dumps(
            {"matches": 2, "outlined": 2, "hidden": 0, "scale": 1, "boxes": boxes}
        )
    )
    out = tmp_path / "pixels.json"
    result = CliRunner().invoke(app, ["verify", str(shot), str(outlined), str(out)])
    assert result.exit_code == 1, result.output
    assert json.loads(out.read_text()) == {"found": 1, "missing": [2]}
