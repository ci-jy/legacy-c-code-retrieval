"""The code index: function units, call graph, BM25 and dense vectors behind one search API."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .callgraph import CallGraph
from .dense import DEFAULT_MODEL, Embedder
from .parser import FunctionUnit, parse_repository
from .ranking import RRF_K, graph_expand, rank_order, rrf
from .tokenize import split_identifier, tokenize

METHODS = ("bm25", "dense", "hybrid", "hybrid_graph")
INDEX_DIRNAME = ".codesearch"
FORMAT_VERSION = 1

# How many candidates from each retriever take part in fusion.
FUSION_DEPTH = 100


def lexical_text(unit: FunctionUnit, include_comment: bool) -> str:
    return unit.file + " " + unit.text(include_comment)


def dense_text(unit: FunctionUnit, include_comment: bool) -> str:
    """Text fed to the embedding model: readable name and path first, since long bodies get truncated."""
    head = f"{' '.join(split_identifier(unit.name))} ({unit.file})"
    return head + "\n" + unit.text(include_comment)


class CodeIndex:
    def __init__(
        self,
        units: list[FunctionUnit],
        include_comments: bool = True,
        model_name: str = DEFAULT_MODEL,
        root: str | None = None,
        embeddings: np.ndarray | None = None,
        graph_seeds: int = 3,
        graph_weight: float = 0.5,
    ):
        self.units = units
        self.root = root
        self.include_comments = include_comments
        self.embedder = Embedder(model_name)
        self.by_id = {u.id: i for i, u in enumerate(units)}
        self.graph = CallGraph(units)
        self._nbr_idx = [[self.by_id[n] for n in self.graph.neighbours(u.id)] for u in units]
        self.bm25 = BM25([tokenize(lexical_text(u, include_comments)) for u in units])
        self._emb = embeddings
        self.graph_seeds = graph_seeds
        self.graph_weight = graph_weight

    # ---------------------------------------------------------------- building
    @classmethod
    def build(cls, root: str | os.PathLike, include_comments: bool = True, model_name: str = DEFAULT_MODEL,
              **kwargs) -> "CodeIndex":
        return cls(parse_repository(root), include_comments, model_name, root=str(root), **kwargs)

    @property
    def embeddings(self) -> np.ndarray:
        if self._emb is None:
            texts = [dense_text(u, self.include_comments) for u in self.units]
            self._emb = self.embedder.encode_documents(texts, show_progress=len(texts) > 500)
        return self._emb

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.embedder.model_name.encode())
        h.update(str(self.include_comments).encode())
        for u in self.units:
            h.update(dense_text(u, self.include_comments).encode("utf-8", "replace"))
        return h.hexdigest()

    # ---------------------------------------------------------------- persistence
    def save(self, directory: str | os.PathLike) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": FORMAT_VERSION,
            "root": self.root,
            "model": self.embedder.model_name,
            "include_comments": self.include_comments,
            "fingerprint": self.fingerprint(),
            "units": [asdict(u) for u in self.units],
        }
        (d / "units.json").write_text(json.dumps(meta))
        np.save(d / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: str | os.PathLike, **kwargs) -> "CodeIndex":
        d = Path(directory)
        meta = json.loads((d / "units.json").read_text())
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format in {d}")
        units = [FunctionUnit(**u) for u in meta["units"]]
        emb = np.load(d / "embeddings.npy")
        return cls(units, meta["include_comments"], meta["model"], root=meta["root"], embeddings=emb, **kwargs)

    @classmethod
    def open(cls, root: str | os.PathLike, model_name: str = DEFAULT_MODEL, rebuild: bool = False) -> "CodeIndex":
        """Load the saved index under root/.codesearch if it is still current, else build and save it."""
        root = Path(root)
        cache = root / INDEX_DIRNAME
        fresh = cls.build(root, model_name=model_name)
        if not rebuild and (cache / "units.json").exists():
            try:
                meta = json.loads((cache / "units.json").read_text())
                if meta.get("fingerprint") == fresh.fingerprint():
                    fresh._emb = np.load(cache / "embeddings.npy")
                    return fresh
            except (ValueError, OSError):
                pass
        fresh.save(cache)
        return fresh

    # ---------------------------------------------------------------- search
    def scores(self, query: str, method: str = "hybrid_graph") -> np.ndarray:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
        n = len(self.units)
        if n == 0:
            return np.zeros(0)
        bm = self.bm25.scores(tokenize(query)) if method != "dense" else None
        dn = self.embeddings @ self.embedder.encode_query(query) if method != "bm25" else None
        return self.combine(method, bm, dn)

    def combine(self, method: str, bm25_scores: np.ndarray | None, dense_scores: np.ndarray | None) -> np.ndarray:
        """Turn raw BM25 and dense scores into the final scores of `method`."""
        if method == "bm25":
            return bm25_scores
        if method == "dense":
            return dense_scores
        lex = rank_order(bm25_scores)
        den = rank_order(dense_scores)
        fused = rrf([lex, den], len(self.units), k=RRF_K, depth=FUSION_DEPTH)
        if method == "hybrid":
            return fused
        return graph_expand(fused, self._nbr_idx.__getitem__, seeds=self.graph_seeds, weight=self.graph_weight)

    def ranking(self, query: str, method: str = "hybrid_graph") -> list[int]:
        return rank_order(self.scores(query, method))

    def search(self, query: str, k: int = 10, method: str = "hybrid_graph") -> list[tuple[FunctionUnit, float]]:
        s = self.scores(query, method)
        return [(self.units[i], float(s[i])) for i in rank_order(s)[:k]]

    # ---------------------------------------------------------------- lookup
    def find(self, name_or_id: str) -> list[FunctionUnit]:
        if name_or_id in self.by_id:
            return [self.units[self.by_id[name_or_id]]]
        return [u for u in self.units if u.name == name_or_id]

    def callers(self, unit_id: str) -> list[FunctionUnit]:
        return [self.units[self.by_id[i]] for i in self.graph.callers(unit_id)]

    def callees(self, unit_id: str) -> list[FunctionUnit]:
        return [self.units[self.by_id[i]] for i in self.graph.callees(unit_id)]
