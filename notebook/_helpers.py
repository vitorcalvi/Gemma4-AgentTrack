"""Shared helpers for the GraphLoc notebook cells."""
from __future__ import annotations
import collections, json, math, os, re
import numpy as np

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]+|[a-z0-9]+")
STOP = frozenset("""a an the and or of to in is are was were be been being for on at by with as from that this it its we you they i not no do does did done can could should would will shall may might must have has had if then else when while but so such into over under about error errors issue bug bugs python code line lines function functions class classes file files repo repository traceback stack trace expected actual got where which what why how use used using new old version versions please thanks thank""".split())

def ident_tokens(text: str) -> list[str]:
    out = []
    for tok in IDENT.findall(text or ""):
        low = tok.lower()
        if low in STOP:
            continue
        out.append(low)
        parts = [p.lower() for p in CAMEL.findall(tok) if len(p) > 1]
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOP)
    return out

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.vocab = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(self.n, dtype=np.float32)
        for i, toks in enumerate(docs):
            lengths[i] = len(toks)
            for term, c in collections.Counter(toks).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(c)
        self.lengths = lengths
        self.avgdl = float(lengths.mean()) if self.n and lengths.sum() else 1.0
        df = np.zeros(len(self.vocab), dtype=np.float32)
        np.add.at(df, np.asarray(cols, dtype=np.int64), 1.0)
        self.idf = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.vals = np.asarray(vals, dtype=np.float32)

    def score(self, query):
        out = np.zeros(self.n, dtype=np.float32)
        if self.n == 0:
            return out
        den = self.k1 * (1.0 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(query):
            j = self.vocab.get(term)
            if j is None:
                continue
            m = self.cols == j
            r, f = self.rows[m], self.vals[m]
            out[r] += self.idf[j] * (f * (self.k1 + 1.0)) / (f + den[r])
        return out

STRUCTURAL = ("calls", "imports", "inherits")

class RepoGraph:
    def __init__(self, path):
        raw = json.load(open(path, encoding="utf-8"))
        meta = raw.get("meta", {}) or {}
        self.instance_id = str(meta.get("instance_id", os.path.basename(path)[:-5]))
        self.repo = str(meta.get("repo", ""))
        self.base_commit = str(meta.get("base_commit", ""))
        nodes = [n for n in raw.get("nodes", []) if n.get("kind") in ("function", "class") and not n.get("test")]
        self.node_ids = [n["id"] for n in nodes]
        self.index = {v: i for i, v in enumerate(self.node_ids)}
        self.kinds = [n.get("kind", "") for n in nodes]
        self.files = [n.get("file", "") for n in nodes]
        self.sigs = [n.get("sig", "") or "" for n in nodes]
        self.is_test = np.zeros(len(nodes), dtype=bool)
        out = [[] for _ in nodes]
        und = [set() for _ in nodes]
        for e in raw.get("edges", raw.get("links", [])) or []:
            u, v, t = (e[0], e[1], e[2]) if isinstance(e, list) else (e["from"], e["to"], e.get("type"))
            if t not in STRUCTURAL:
                continue
            a, b = self.index.get(u), self.index.get(v)
            if a is None or b is None or a == b:
                continue
            out[a].append((b, t)); und[a].add(b); und[b].add(a)
        self.out_edges = out
        self.adj = [np.fromiter(sorted(s), dtype=np.int64, count=len(s)) for s in und]
        self.n_nodes = len(self.node_ids)
        docs = [ident_tokens(self.node_ids[i].split("::")[-1]) + ident_tokens(self.sigs[i]) for i in range(self.n_nodes)]
        self.bm25 = BM25(docs)

    def bfs(self, seeds, max_hops, cap=None):
        label = np.full(self.n_nodes, -1, dtype=np.int16)
        cur = [int(s) for s in seeds]
        for s in cur:
            label[s] = 0
        for h in range(1, max_hops + 1):
            if cap is not None and int((label >= 0).sum()) > cap:
                break
            nxt = []
            for i in cur:
                for j in self.adj[i]:
                    if label[j] < 0:
                        label[j] = h; nxt.append(int(j))
            if not nxt:
                break
            cur = nxt
        return label

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
