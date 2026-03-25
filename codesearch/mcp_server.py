"""MCP server exposing a CodeIndex to LLM agents.

Tools: search_code, get_function, callers_of, callees_of. Run with
`python3 -m codesearch.cli serve <repo>` (stdio transport).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .index import METHODS, CodeIndex
from .parser import FunctionUnit

MAX_K = 50

INSTRUCTIONS = (
    "Search and navigate a C codebase at function granularity. Start with search_code using a "
    "natural-language description of the behaviour you are looking for, read candidates with "
    "get_function, then follow the static call graph with callers_of / callees_of. Functions are "
    "identified by 'path/to/file.c::name'; a bare name is accepted when it is unambiguous."
)


class FunctionRef(BaseModel):
    id: str = Field(description="Stable identifier, 'relative/path.c::name'")
    name: str
    file: str
    start_line: int
    end_line: int
    signature: str


class SearchHit(FunctionRef):
    rank: int
    score: float
    comment: str = Field(description="Leading comment of the function, if any")


class SearchResult(BaseModel):
    query: str
    method: str
    results: list[SearchHit]


class FunctionDetail(FunctionRef):
    comment: str
    code: str = Field(description="Source of the definition, including its leading comment")
    callers: list[str]
    callees: list[str]


class FunctionLookup(BaseModel):
    query: str
    matches: list[FunctionDetail]


class Relation(BaseModel):
    function: FunctionRef
    related: list[FunctionRef]


class RelationResult(BaseModel):
    query: str
    relation: str = Field(description="'callers' or 'callees'")
    matches: list[Relation]


def _ref(u: FunctionUnit) -> FunctionRef:
    return FunctionRef(id=u.id, name=u.name, file=u.file, start_line=u.start_line, end_line=u.end_line,
                       signature=u.signature)


def create_server(index: CodeIndex) -> MCPServer:
    server = MCPServer("codesearch", instructions=INSTRUCTIONS)

    def resolve(name_or_id: str) -> list[FunctionUnit]:
        found = index.find(name_or_id.strip())
        if not found:
            raise ToolError(f"no function named {name_or_id!r}")
        return found

    @server.tool()
    def search_code(query: str, k: int = 10, method: str = "hybrid_graph") -> SearchResult:
        """Find functions matching a natural-language or identifier query.

        method is one of bm25, dense, hybrid, hybrid_graph (default: hybrid search with call-graph
        expansion). Returns up to k ranked functions with their location, signature and comment.
        """
        if method not in METHODS:
            raise ToolError(f"method must be one of {', '.join(METHODS)}")
        k = max(1, min(int(k), MAX_K))
        hits = [
            SearchHit(rank=r, score=round(score, 6), comment=u.comment, **_ref(u).model_dump())
            for r, (u, score) in enumerate(index.search(query, k=k, method=method), start=1)
        ]
        return SearchResult(query=query, method=method, results=hits)

    @server.tool()
    def get_function(name: str) -> FunctionLookup:
        """Return the full source of a function (by id 'file.c::name' or bare name) with its call edges."""
        matches = [
            FunctionDetail(
                comment=u.comment,
                code=u.text(include_comment=True),
                callers=index.graph.callers(u.id),
                callees=index.graph.callees(u.id),
                **_ref(u).model_dump(),
            )
            for u in resolve(name)
        ]
        return FunctionLookup(query=name, matches=matches)

    @server.tool()
    def callers_of(name: str) -> RelationResult:
        """List the functions in the repository that call the given function."""
        return RelationResult(
            query=name,
            relation="callers",
            matches=[Relation(function=_ref(u), related=[_ref(c) for c in index.callers(u.id)]) for u in resolve(name)],
        )

    @server.tool()
    def callees_of(name: str) -> RelationResult:
        """List the repository functions called by the given function (library calls are omitted)."""
        return RelationResult(
            query=name,
            relation="callees",
            matches=[Relation(function=_ref(u), related=[_ref(c) for c in index.callees(u.id)]) for u in resolve(name)],
        )

    return server
