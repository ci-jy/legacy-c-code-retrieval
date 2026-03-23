"""Split C sources into function-level units with tree-sitter."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tree_sitter_c
from tree_sitter import Language, Node, Parser

C_EXTENSIONS = (".c", ".h")

# Directories that are almost never part of the code a user wants to search.
SKIP_DIRS = {".git", ".hg", ".svn", "build", "node_modules", ".codesearch"}


@dataclass
class FunctionUnit:
    """One C function definition and everything retrieval needs to know about it."""

    id: str
    name: str
    file: str  # path relative to the repository root, POSIX separators
    start_line: int  # 1-based, inclusive (the signature, not the comment)
    end_line: int  # 1-based, inclusive
    signature: str
    body: str  # full definition text: signature plus braces
    comment: str  # cleaned leading comment, "" when absent
    comment_raw: str = ""
    is_static: bool = False
    calls: list[str] = field(default_factory=list)  # called identifiers, in source order, deduplicated

    def text(self, include_comment: bool = True) -> str:
        if include_comment and self.comment_raw:
            return self.comment_raw + "\n" + self.body
        return self.body

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "comment": self.comment,
            "is_static": self.is_static,
        }


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tree_sitter_c.language()))


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _next_declarator(node: Node) -> Node | None:
    inner = node.child_by_field_name("declarator")
    if inner is None and node.type == "parenthesized_declarator" and node.named_children:
        inner = node.named_children[0]
    return inner


def _function_declarator(node: Node | None) -> Node | None:
    """Descend through pointer/parenthesized declarators to the function_declarator."""
    while node is not None and node.type != "function_declarator":
        node = _next_declarator(node)
    return node


def _function_name(defn: Node, src: bytes) -> str | None:
    fdecl = _function_declarator(defn.child_by_field_name("declarator"))
    if fdecl is None:
        return None
    ident = fdecl.child_by_field_name("declarator")
    while ident is not None and ident.type not in ("identifier", "field_identifier"):
        ident = _next_declarator(ident)
    if ident is None:
        return None
    return _text(ident, src)


_COMMENT_DECOR = re.compile(r"^[\s/*!<-]*")
_RULE_LINE = re.compile(r"^[\s=*#~_+-]*$")


def clean_comment(raw: str) -> str:
    """Strip C comment markers and decoration, returning plain prose on one line."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("//"):
            line = line[2:]
        line = line.replace("/*", " ").replace("*/", " ")
        line = _COMMENT_DECOR.sub("", line).strip()
        line = line.rstrip("*").strip()
        if not line or _RULE_LINE.match(line):
            continue
        lines.append(line)
    return " ".join(" ".join(lines).split())


def _leading_comment(defn: Node, src: bytes) -> str:
    """Collect the comment block directly above a definition (at most one blank line between)."""
    parts: list[Node] = []
    expected_row = defn.start_point[0]
    sib = defn.prev_sibling
    while sib is not None and sib.type == "comment":
        if expected_row - sib.end_point[0] > 2:
            break
        parts.append(sib)
        expected_row = sib.start_point[0]
        sib = sib.prev_sibling
    parts.reverse()
    return "\n".join(_text(p, src) for p in parts)


def _collect_calls(body: Node, src: bytes) -> list[str]:
    seen: dict[str, None] = {}
    stack = [body]
    while stack:
        node = stack.pop()
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                seen.setdefault(_text(fn, src), None)
        stack.extend(reversed(node.children))
    return list(seen)


def _iter_definitions(root: Node):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            yield node
            continue  # nested definitions are not valid C
        stack.extend(reversed(node.children))


def parse_source(src: bytes, rel_path: str) -> list[FunctionUnit]:
    """Extract function units from one C source buffer."""
    tree = _parser().parse(src)
    units: list[FunctionUnit] = []
    for defn in _iter_definitions(tree.root_node):
        name = _function_name(defn, src)
        body_node = defn.child_by_field_name("body")
        if name is None or body_node is None:
            continue
        sig = src[defn.start_byte : body_node.start_byte].decode("utf-8", errors="replace").strip()
        sig = " ".join(sig.split())
        is_static = any(
            c.type == "storage_class_specifier" and _text(c, src) == "static" for c in defn.children
        )
        raw = _leading_comment(defn, src)
        units.append(
            FunctionUnit(
                id="",
                name=name,
                file=rel_path,
                start_line=defn.start_point[0] + 1,
                end_line=defn.end_point[0] + 1,
                signature=sig,
                body=_text(defn, src),
                comment=clean_comment(raw),
                comment_raw=raw,
                is_static=is_static,
                calls=_collect_calls(body_node, src),
            )
        )
    return units


def iter_c_files(root: str | os.PathLike) -> list[Path]:
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(C_EXTENSIONS):
                out.append(Path(dirpath) / fn)
    return out


def parse_repository(root: str | os.PathLike) -> list[FunctionUnit]:
    """Parse every .c/.h file under root, assigning stable unique ids."""
    root = Path(root)
    units: list[FunctionUnit] = []
    for path in iter_c_files(root):
        rel = path.relative_to(root).as_posix()
        units.extend(parse_source(path.read_bytes(), rel))
    counts: dict[str, int] = {}
    for u in units:
        key = f"{u.file}::{u.name}"
        counts[key] = counts.get(key, 0) + 1
    for u in units:
        key = f"{u.file}::{u.name}"
        # Same name twice in one file happens with #ifdef alternatives; disambiguate by line.
        u.id = key if counts[key] == 1 else f"{key}@{u.start_line}"
    return units
