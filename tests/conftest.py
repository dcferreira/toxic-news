# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the ``toxic_news`` test suite."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.mock_site import serve


@pytest.fixture(scope="module")
def assets():
    """Path to the directory of recorded HTML assets the fetcher tests replay."""
    return Path(__file__).resolve().parent / "assets"


@pytest.fixture(scope="module")
def mock_site_url(assets) -> Iterator[str]:
    """The origin of a mock site serving the recorded front pages."""
    server, origin = serve(assets / "html")
    yield origin
    server.shutdown()
    server.server_close()
