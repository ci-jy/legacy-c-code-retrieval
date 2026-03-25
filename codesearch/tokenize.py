"""Identifier-aware tokenisation for lexical search over C code and English queries."""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

C_KEYWORDS = frozenset(
    """auto break case char const continue default do double else enum extern float for goto if
    inline int long register restrict return short signed sizeof static struct switch typedef union
    unsigned void volatile while bool true false null define include ifdef ifndef endif""".split()
)

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was were
    will with which when where if then than not no it's we you they there these those can""".split()
)


def split_identifier(ident: str) -> list[str]:
    """Split `lua_newState2` into ['lua', 'new', 'state', '2']."""
    parts: list[str] = []
    for chunk in ident.split("_"):
        if chunk:
            parts.extend(p.lower() for p in _CAMEL.findall(chunk))
    return parts


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: sub-words of every identifier plus the whole identifier when compound."""
    out: list[str] = []
    for word in _WORD.findall(text):
        subs = split_identifier(word)
        whole = word.lower()
        if len(subs) > 1:
            out.append(whole)
        out.extend(subs)
    return [t for t in out if len(t) > 1 and t not in C_KEYWORDS and t not in STOPWORDS and not t.isdigit()]
