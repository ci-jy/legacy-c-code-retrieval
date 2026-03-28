from codesearch.tokenize import split_identifier, tokenize


def test_split_identifier():
    assert split_identifier("lua_newstate") == ["lua", "newstate"]
    assert split_identifier("luaH_getShortStr") == ["lua", "h", "get", "short", "str"]
    assert split_identifier("HTTPServerError") == ["http", "server", "error"]
    assert split_identifier("deflate_stored2") == ["deflate", "stored", "2"]
    assert split_identifier("__init__") == ["init"]


def test_tokenize_keeps_compounds_and_drops_noise():
    toks = tokenize("static int table_put(Table *t) { return 0; }")
    # keywords, numbers and one-letter names are dropped; the compound name is kept whole too
    assert toks == ["table_put", "table", "put", "table"]


def test_tokenize_english_drops_stopwords():
    assert tokenize("Look up a key in the hash table") == ["look", "up", "key", "hash", "table"]
