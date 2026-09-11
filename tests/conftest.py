# SPDX-FileCopyrightText: 2023-present Daniel Ferreira <daniel.ferreira.1@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the ``toxic_news`` test suite."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def assets():
    """Path to the directory of recorded HTML assets the fetcher tests replay."""
    return Path(__file__).resolve().parent / "assets"
