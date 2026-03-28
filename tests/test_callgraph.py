from codesearch.callgraph import CallGraph

EXPECTED_EDGES = {
    ("src/strbuf.c::sb_grow", "src/strbuf.c::log_msg"),
    ("src/strbuf.c::sb_append", "src/strbuf.c::sb_grow"),
    ("src/strbuf.c::sb_append", "include/strbuf.h::sb_len"),
    ("src/hash.c::table_get", "src/hash.c::hash_bytes"),
    ("src/hash.c::table_put", "src/hash.c::table_resize"),
    ("src/hash.c::table_put", "src/hash.c::hash_bytes"),
    ("src/hash.c::table_resize", "src/hash.c::hash_bytes"),
    ("src/hash.c::free_chain", "src/hash.c::free_chain"),
    ("src/hash.c::table_free", "src/hash.c::free_chain"),
    ("src/main.c::parse_line", "src/hash.c::table_put"),
    ("src/main.c::load_config", "src/main.c::log_msg"),
    ("src/main.c::load_config", "src/main.c::parse_line"),
    ("src/main.c::dump_config", "src/strbuf.c::sb_append"),
    ("src/main.c::dump_config", "src/main.c::log_msg"),
    ("src/main.c::dump_config", "src/strbuf.c::sb_free"),
    ("src/main.c::main", "src/main.c::load_config"),
    ("src/main.c::main", "src/main.c::dump_config"),
    ("src/main.c::main", "src/hash.c::table_get"),
    ("src/main.c::main", "src/hash.c::table_free"),
}


def test_edges_match_exactly(units):
    assert CallGraph(units).edges() == EXPECTED_EDGES


def test_callers_and_callees(units):
    g = CallGraph(units)
    assert g.callers("src/hash.c::hash_bytes") == [
        "src/hash.c::table_get",
        "src/hash.c::table_put",
        "src/hash.c::table_resize",
    ]
    assert g.callees("src/main.c::main") == [
        "src/hash.c::table_free",
        "src/hash.c::table_get",
        "src/main.c::dump_config",
        "src/main.c::load_config",
    ]
    assert g.callers("src/main.c::main") == []
    assert g.callers("src/hash.c::free_chain") == ["src/hash.c::free_chain", "src/hash.c::table_free"]


def test_static_functions_resolve_within_their_file(units):
    g = CallGraph(units)
    assert g.callers("src/main.c::log_msg") == ["src/main.c::dump_config", "src/main.c::load_config"]
    assert g.callers("src/strbuf.c::log_msg") == ["src/strbuf.c::sb_grow"]


def test_neighbours_are_union(units):
    g = CallGraph(units)
    assert g.neighbours("src/hash.c::table_put") == [
        "src/hash.c::hash_bytes",
        "src/hash.c::table_resize",
        "src/main.c::parse_line",
    ]


def test_unknown_id_is_empty(units):
    g = CallGraph(units)
    assert g.callers("nope") == [] and g.callees("nope") == []
