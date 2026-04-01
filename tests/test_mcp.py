import json

import anyio
import pytest

from mcp import Client

from codesearch.mcp_server import create_server

TOOLS = {"search_code", "get_function", "callers_of", "callees_of"}


def call(index, tool, args):
    async def run():
        async with Client(create_server(index)) as client:
            return await client.call_tool(tool, args)

    return anyio.run(run)


def payload(result):
    assert not result.is_error, result.content
    data = result.structured_content
    assert data is not None
    # the text content carries the same JSON for clients that ignore structured output
    assert json.loads(result.content[0].text) == data
    return data


def test_lists_the_four_tools(index):
    async def run():
        async with Client(create_server(index)) as client:
            return await client.list_tools()

    tools = {t.name: t for t in anyio.run(run).tools}
    assert set(tools) == TOOLS
    for t in tools.values():
        assert t.description and t.input_schema["type"] == "object"
        assert t.output_schema is not None


def test_search_code(index):
    data = payload(call(index, "search_code", {"query": "look up a key in the hash table", "k": 3}))
    assert data["query"] == "look up a key in the hash table"
    assert data["method"] == "hybrid_graph"
    assert [h["rank"] for h in data["results"]] == [1, 2, 3]
    assert data["results"][0]["id"] == "src/hash.c::table_get"
    for h in data["results"]:
        assert set(h) >= {"id", "name", "file", "start_line", "end_line", "signature", "score", "comment"}
        assert h["start_line"] <= h["end_line"]


def test_search_code_method_and_bounds(index):
    data = payload(call(index, "search_code", {"query": "hash", "k": 1000, "method": "bm25"}))
    assert data["method"] == "bm25"
    assert len(data["results"]) == 16  # capped by repository size
    bad = call(index, "search_code", {"query": "hash", "method": "nope"})
    assert bad.is_error


def test_get_function(index):
    data = payload(call(index, "get_function", {"name": "table_put"}))
    (m,) = data["matches"]
    assert m["id"] == "src/hash.c::table_put"
    assert m["code"].startswith("/* Insert or replace") and m["code"].rstrip().endswith("}")
    assert m["callees"] == ["src/hash.c::hash_bytes", "src/hash.c::table_resize"]
    assert m["callers"] == ["src/main.c::parse_line"]


def test_get_function_ambiguous_and_missing(index):
    data = payload(call(index, "get_function", {"name": "log_msg"}))
    assert [m["file"] for m in data["matches"]] == ["src/main.c", "src/strbuf.c"]
    missing = call(index, "get_function", {"name": "does_not_exist"})
    assert missing.is_error
    assert "does_not_exist" in missing.content[0].text


def test_callers_of(index):
    data = payload(call(index, "callers_of", {"name": "hash_bytes"}))
    assert data["relation"] == "callers"
    (m,) = data["matches"]
    assert m["function"]["id"] == "src/hash.c::hash_bytes"
    assert [r["name"] for r in m["related"]] == ["table_get", "table_put", "table_resize"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("src/main.c::main", ["table_free", "table_get", "dump_config", "load_config"]),
        ("sb_append", ["sb_len", "sb_grow"]),
        ("hash_bytes", []),
    ],
)
def test_callees_of(index, name, expected):
    data = payload(call(index, "callees_of", {"name": name}))
    assert data["relation"] == "callees"
    (m,) = data["matches"]
    assert [r["name"] for r in m["related"]] == expected
