from codesearch.parser import clean_comment, parse_source

EXPECTED_UNITS = {
    # id: (start_line, end_line, is_static)
    "include/strbuf.h::sb_len": (13, 16, True),
    "src/hash.c::hash_bytes": (8, 16, False),
    "src/hash.c::table_get": (19, 28, False),
    "src/hash.c::table_put": (31, 45, False),
    "src/hash.c::table_resize": (49, 71, True),
    "src/hash.c::free_chain": (74, 81, True),
    "src/hash.c::table_free": (83, 88, False),
    "src/main.c::log_msg": (6, 9, True),
    "src/main.c::parse_line": (12, 19, True),
    "src/main.c::load_config": (22, 38, False),
    "src/main.c::dump_config": (42, 52, True),
    "src/main.c::main": (56, 68, False),
    "src/strbuf.c::log_msg": (6, 9, True),
    "src/strbuf.c::sb_grow": (12, 25, True),
    "src/strbuf.c::sb_append": (28, 36, False),
    "src/strbuf.c::sb_free": (42, 47, False),
}

EXPECTED_COMMENTS = {
    "include/strbuf.h::sb_len": "Return the number of bytes currently stored in the buffer.",
    "src/hash.c::hash_bytes": "Compute the FNV-1a hash of a byte string.",
    "src/hash.c::table_get": "Look up a key in the hash table and return its stored value, or NULL when the key is missing.",
    "src/hash.c::table_put": "Insert or replace the value stored under a key, resizing the table when it becomes too full.",
    "src/hash.c::table_resize": "Double the number of buckets and rehash every entry.",
    "src/hash.c::free_chain": "Recursively free a chain of hash table entries.",
    "src/hash.c::table_free": "",
    "src/main.c::log_msg": "",
    "src/main.c::parse_line": 'Parse one "key=value" line and store it in the table.',
    "src/main.c::load_config": "Read configuration lines from a file into the table, skipping comments and blank lines.",
    "src/main.c::dump_config": "Write every configuration entry to standard error.",
    "src/main.c::main": "Entry point: load the config file named on the command line and print one value.",
    "src/strbuf.c::log_msg": "",
    "src/strbuf.c::sb_grow": "Grow the buffer so that it can hold at least need more bytes.",
    "src/strbuf.c::sb_append": "Append a NUL-terminated string to the end of the string buffer.",
    "src/strbuf.c::sb_free": "Release the memory owned by a string buffer and reset it to empty.",
}


def test_function_set_matches_exactly(units):
    assert {u.id for u in units} == set(EXPECTED_UNITS)
    assert len(units) == len(EXPECTED_UNITS)  # prototypes in headers are not units


def test_line_spans_and_storage(units):
    for u in units:
        assert (u.start_line, u.end_line, u.is_static) == EXPECTED_UNITS[u.id], u.id


def test_leading_comments(units):
    assert {u.id: u.comment for u in units} == EXPECTED_COMMENTS


def test_signature_and_body(units, fixture_root):
    by_id = {u.id: u for u in units}
    u = by_id["src/hash.c::table_get"]
    assert u.signature == "void *table_get(Table *t, const char *key)"
    lines = (fixture_root / u.file).read_text().splitlines()
    assert u.body == "\n".join(lines[u.start_line - 1 : u.end_line])
    assert u.body.startswith("void *table_get(") and u.body.endswith("}")
    assert "Look up a key" not in u.body
    assert u.text(include_comment=True).startswith("/* Look up a key")
    assert u.text(include_comment=False) == u.body


def test_function_inside_ifdef_is_found(units):
    assert any(u.name == "dump_config" for u in units)


def test_called_identifiers(units):
    by_id = {u.id: u for u in units}
    assert by_id["src/strbuf.c::sb_append"].calls == ["strlen", "sb_grow", "memcpy", "sb_len"]
    # t->on_resize(t) is a call through a pointer and records no name
    assert by_id["src/hash.c::table_resize"].calls == ["calloc", "hash_bytes", "free"]


def test_declarator_shapes():
    src = b"""
int **ptrs(void) { return 0; }
int (*pick(int which))(int) { return 0; }
EXPORT int with_macro (lua_State *L) { return helper(L); }
int proto_only(int x);
static const char *const_name(void) { return "x"; }
"""
    units = parse_source(src, "x.c")
    assert [u.name for u in units] == ["ptrs", "pick", "with_macro", "const_name"]
    assert units[2].calls == ["helper"]
    assert units[3].is_static


def test_comment_must_be_adjacent():
    src = b"""/* far away */



int a(void) { return 0; }

/* near */

int b(void) { return 0; }
"""
    a, b = parse_source(src, "x.c")
    assert a.comment == ""
    assert b.comment == "near"


def test_clean_comment_styles():
    assert clean_comment("/*\n** Lua style\n** two lines\n*/") == "Lua style two lines"
    assert clean_comment("/* ===== */\n/* real text */") == "real text"
    assert clean_comment("// a\n// b") == "a b"
    assert clean_comment("/** doxygen\n * @brief thing\n */") == "doxygen @brief thing"
    assert clean_comment("") == ""
