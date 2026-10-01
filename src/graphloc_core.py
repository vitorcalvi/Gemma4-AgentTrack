"""
GraphLoc core engine.

Reproducible building blocks for function-level bug localization on
code-graph artifacts of the format emitted by the Gemma 4 Developer
Agent harness (graph JSON v1 + node embeddings NPZ).

Pure standard library + numpy. No network, no LLM, deterministic.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- tokenization

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]+|[a-z0-9]+")
# Generic vocabulary that carries no localization signal.
STOP = frozenset("""
a an the and or of to in is are was were be been being for on at by with as from that this
it its we you they he she i not no do does did done can could should would will shall may
might must have has had if then else when while but so such into over under about
error errors issue bug issue bugs python code line lines function functions class classes
file files repo repository traceback stack trace expected actual got when where which
what why how use used using new old version versions please thanks thank hi hello
""".split())


def ident_tokens(text: str) -> list[str]:
    """Split an identifier into subtokens, keeping the whole identifier too."""
    out: list[str] = []
    for tok in _IDENT.findall(text or ""):
        low = tok.lower()
        if low in STOP:
            continue
        out.append(low)
        parts = [p.lower() for p in _CAMEL.findall(tok) if len(p) > 1]
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOP)
    return out


def qualified_name(node_id: str) -> str:
    return node_id.split("::")[-1] if "::" in node_id else node_id


# ---------------------------------------------------------------------- BM25

class BM25:
    """Standard Okapi BM25 over an in-memory sparse term-document matrix."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, toks in enumerate(docs):
            lengths[i] = len(toks)
            counts = collections.Counter(toks)
            for term, c in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.lengths = lengths
        self.avgdl = float(lengths.mean()) if self.n and lengths.sum() else 1.0
        df = np.zeros(len(self.vocab), dtype=np.float32)
        np.add.at(df, np.asarray(cols, dtype=np.int64), 1.0)
        self.idf = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.vals = np.asarray(vals, dtype=np.float32)

    def score(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if self.n == 0:
            return out
        denom_len = self.k1 * (1.0 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(query):
            j = self.vocab.get(term)
            if j is None:
                continue
            m = self.cols == j
            r = self.rows[m]
            f = self.vals[m]
            out[r] += self.idf[j] * (f * (self.k1 + 1.0)) / (f + denom_len[r])
        return out


# ------------------------------------------------------------------- graph I/O

STRUCTURAL_EDGES = ("calls", "imports", "inherits")


@dataclass
class RepoGraph:
    instance_id: str
    repo: str
    base_commit: str
    node_ids: list[str]
    index: dict[str, int]
    kinds: list[str]
    is_test: np.ndarray
    files: list[str]
    sigs: list[str]
    adj: list[np.ndarray]
    out_edges: list[list[tuple[int, str]]]
    bm25: BM25
    file_bm25: BM25
    embeddings: np.ndarray | None = None
    emb_keys: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def bfs(self, seeds, max_hops: int = 3, cap: int | None = None) -> np.ndarray:
        """Minimum hop distance from `seeds` over the structural subgraph.

        Returns an int16 array of hop labels; -1 means unreached within
        `max_hops`. If `cap` is given, expansion stops as soon as the
        frontier exceeds `cap` nodes, so the cost stays bounded on
        large, densely connected repositories.
        """
        label = np.full(self.n_nodes, -1, dtype=np.int16)
        cur: list[int] = []
        for s in seeds:
            i = int(s)
            if 0 <= i < self.n_nodes and label[i] < 0:
                label[i] = 0
                cur.append(i)
        for h in range(1, max_hops + 1):
            if cap is not None and int((label >= 0).sum()) > cap:
                break
            nxt: list[int] = []
            for i in cur:
                for j in self.adj[i]:
                    if label[j] < 0:
                        label[j] = h
                        nxt.append(int(j))
            if not nxt:
                break
            cur = nxt
        return label


def load_graph(path: str, *, code_nodes_only: bool = True) -> RepoGraph:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    meta = raw.get("meta", {}) or {}
    nodes = raw.get("nodes", []) or []
    edges = raw.get("edges", raw.get("links", [])) or []

    keep = []
    for nd in nodes:
        if code_nodes_only and nd.get("kind") not in ("function", "class"):
            continue
        if code_nodes_only and nd.get("test"):
            continue
        keep.append(nd)
    node_ids = [nd["id"] for nd in keep]
    index = {nid: i for i, nid in enumerate(node_ids)}
    kinds = [nd.get("kind", "") for nd in keep]
    is_test = np.asarray([bool(nd.get("test")) for nd in keep], dtype=bool)
    files = [nd.get("file", "") for nd in keep]
    sigs = [nd.get("sig", "") or "" for nd in keep]

    out_edges: list[list[tuple[int, str]]] = [[] for _ in node_ids]
    undirected: list[set[int]] = [set() for _ in node_ids]
    for e in edges:
        if isinstance(e, (list, tuple)):
            u, v, t = e[0], e[1], e[2]
        else:
            u, v, t = e.get("from"), e.get("to"), e.get("type")
        if t not in STRUCTURAL_EDGES:
            continue
        iu, iv = index.get(u), index.get(v)
        if iu is None or iv is None or iu == iv:
            continue
        out_edges[iu].append((iv, t))
        undirected[iu].add(iv)
        undirected[iv].add(iu)
    adj = [np.fromiter(sorted(s), dtype=np.int64, count=len(s)) for s in undirected]

    docs = [ident_tokens(qualified_name(nid)) + ident_tokens(sig) for nid, sig in zip(node_ids, sigs)]
    file_names = sorted(set(files))
    fidx = {f: i for i, f in enumerate(file_names)}
    fdocs: list[list[str]] = [collections.Counter() for _ in file_names]
    for i, f in enumerate(files):
        for t in ident_tokens(f):
            fdocs[fidx[f]][t] += 1
    fdocs = [list(c) for c in fdocs]

    return RepoGraph(
        instance_id=str(meta.get("instance_id", os.path.basename(path)[:-5])),
        repo=str(meta.get("repo", "")),
        base_commit=str(meta.get("base_commit", "")),
        node_ids=node_ids,
        index=index,
        kinds=kinds,
        is_test=is_test,
        files=files,
        sigs=sigs,
        adj=adj,
        out_edges=out_edges,
        bm25=BM25(docs),
        file_bm25=BM25(fdocs),
    )


def load_embeddings(npz_path: str) -> tuple[list[str], np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.files)
        mat = np.stack([np.asarray(z[k], dtype=np.float32) for k in keys])
    return keys, mat
