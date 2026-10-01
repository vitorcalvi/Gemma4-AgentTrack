#!/usr/bin/env python3
"""ADK skill entry point: rank functions by likelihood of being the patch target.

Runs inside the competition sandbox. Reads the repository code graph that
ships with the task, seeds it with BM25 over the issue text, and expands
across the call graph.

Standard library + numpy only.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]+|[a-z0-9]+")
STOP = frozenset("""
a an the and or of to in is are was were be been being for on at by with as from that this it its we you they i not no do does did done can could should would will shall may might must have has had if then else when while but so such into over under about error errors issue bug bugs python code line lines function functions class classes file files repo repository traceback stack trace expected actual got where which what why how use used using new old version versions please thanks thank
""".split())
STRUCTURAL = ("calls", "imports", "inherits")


def ident_tokens(text):
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
        self.lengths = [0.0] * self.n
        for i, toks in enumerate(docs):
            self.lengths[i] = float(len(toks))
            for term, c in collections.Counter(toks).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(c)
        self.lengths = self.lengths or [1.0]
        self.avgdl = sum(self.lengths) / len(self.lengths) or 1.0
        df = collections.Counter()
        for c in cols:
            df[c] += 1
        self.idf = {j: math.log(1.0 + (self.n - n + 0.5) / (n + 0.5))
                    for j, n in df.items()}
        self.rows, self.cols, self.vals = rows, cols, vals

    def score(self, query):
        s = [0.0] * self.n
        wanted = {self.vocab[w] for w in set(query) if w in self.vocab}
        for r, c, v in zip(self.rows, self.cols, self.vals):
            if c in wanted:
                s[r] += self.idf[c] * (v * (self.k1 + 1.0)) / (
                    v + self.k1 * (1.0 - self.b + self.b * self.lengths[r] / self.avgdl))
        return s


def find_graph(repo_name, search_dirs):
    """Locate the repository code graph in the workspace."""
    for d in search_dirs:
        for cand in (f"{repo_name}.json",
                     os.path.join(d, "graphs", f"{repo_name}.json"),
                     os.path.join(d, f"{repo_name.replace('/', '_')}.json")):
            if os.path.exists(cand):
                return cand
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".json") and "graph" in f.lower():
                    return os.path.join(root, f)
    return None


def localize(path, issue, seed_k=10, hops=2, cap=200, top_n=20):
    raw = json.load(open(path, encoding="utf-8"))
    nodes = [n for n in raw.get("nodes", []) if n.get("kind") in ("function", "class")]
    ids = [n["id"] for n in nodes]
    index = {v: i for i, v in enumerate(ids)}
    files = [n.get("file", "") for n in nodes]
    starts = [n.get("start", 0) for n in nodes]
    names = [v.split("::")[-1] for v in ids]
    sigs = [n.get("sig", "") or "" for n in nodes]
    is_test = [bool(n.get("test")) for n in nodes]

    adj = [set() for _ in ids]
    for e in raw.get("edges", raw.get("links", [])) or []:
        if isinstance(e, (list, tuple)):
            u, v, t = e[0], e[1], e[2]
        else:
            u, v, t = e.get("from"), e.get("to"), e.get("type")
        if t not in STRUCTURAL:
            continue
        a, b = index.get(u), index.get(v)
        if a is None or b is None or a == b:
            continue
        adj[a].add(b); adj[b].add(a)

    docs = [ident_tokens(names[i]) + ident_tokens(sigs[i]) for i in range(len(ids))]
    lex = BM25(docs).score(ident_tokens(issue))
    order = sorted(range(len(ids)), key=lambda i: -lex[i])

    seeds = order[:seed_k]
    label = [-1] * len(ids)
    for s in seeds:
        label[s] = 0
    frontier = list(seeds)
    for h in range(1, hops + 1):
        if cap and sum(1 for x in label if x >= 0) > cap:
            break
        nxt = []
        for i in frontier:
            for j in adj[i]:
                if label[j] < 0:
                    label[j] = h
                    nxt.append(j)
        if not nxt:
            break
        frontier = nxt

    out = []
    for h in range(0, max(label) + 1 if label else 0):
        layer = [i for i in range(len(ids)) if label[i] == h and not is_test[i]]
        layer.sort(key=lambda i: -lex[i])
        for i in layer[:top_n]:
            out.append((len(out) + 1, lex[i], h, files[i], starts[i], names[i], ids[i]))
            if len(out) >= top_n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default="")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--repo", default=os.environ.get("REPO_NAME", ""))
    ap.add_argument("--graph", default="")
    ap.add_argument("--seed-k", type=int, default=10)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=20)
    a = ap.parse_args()

    try:
        opts = json.loads(a.args) if a.args else {}
    except Exception:
        opts = {}
    issue = opts.get("issue", a.issue)
    seed_k = int(opts.get("seed_k", a.seed_k))
    hops = int(opts.get("hops", a.hops))
    cap = int(opts.get("cap", a.cap))
    top_n = int(opts.get("top_n", a.top_n))

    if not issue:
        print("ERROR: no issue text provided (args.issue)", file=sys.stderr)
        return 2

    search = [os.environ.get("GRAPH_DIR", "data/graphs"), "data", "/workspace", "."]
    path = a.graph or find_graph(a.repo, search)
    if not path:
        print("NOTE: no code graph found; falling back to name matching only.")
        return 0

    try:
        rows = localize(path, issue, seed_k, hops, cap, top_n)
    except Exception as exc:
        print(f"ERROR: localization failed: {exc}", file=sys.stderr)
        return 1

    mx = max((r[1] for r in rows), default=1.0) or 1.0
    print(f"# graph: {path}")
    print(f"# seeds={seed_k} hops={hops} cap={cap} candidates={len(rows)}")
    print("# rank score hops location symbol")
    for rank, sc, h, f, ln, nm, nid in rows:
        print(f"{rank:>4} {sc / mx:>6.3f} {h:>4} {f}:{ln} {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
