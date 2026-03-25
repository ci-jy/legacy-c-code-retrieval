"""A small, deterministic Okapi BM25 index."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {
            t: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        """Score every document; repeated query terms count once."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in sorted(set(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idx = np.fromiter((p[0] for p in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float64, count=len(plist))
            out[idx] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm[idx])
        return out
