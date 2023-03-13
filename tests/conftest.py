from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def assets():
    return Path(__file__).resolve().parent / "assets"
