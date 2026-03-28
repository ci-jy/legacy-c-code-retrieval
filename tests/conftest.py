from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "mini_c"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE


@pytest.fixture(scope="session")
def units(fixture_root):
    from codesearch.parser import parse_repository

    return parse_repository(fixture_root)


@pytest.fixture(scope="session")
def index(fixture_root):
    from codesearch.index import CodeIndex

    return CodeIndex.build(fixture_root)
