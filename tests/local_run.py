# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Run the whole pipeline against the recorded front pages, with no network.

This is what `poe e2e` runs: the mock site serves `tests/assets/html` on an
ephemeral port, every outlet is pointed at it, and the day is scraped, scored,
stored, aggregated and rendered exactly as the scheduled job does — without
touching a live newspaper.
"""

from pathlib import Path

from loguru import logger

from tests.mock_site import serve
from toxic_news.main import update

ASSETS = Path(__file__).parent / "assets" / "html"


def main() -> None:
    """Serve the recorded front pages, run one `update` against them, and stop."""
    server, base_url = serve(ASSETS, port=0)
    logger.info(f"Serving the recorded front pages on {base_url}")
    try:
        update(data_dir=Path("data"), out_dir=Path("public"), base_url=base_url)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
