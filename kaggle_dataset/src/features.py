"""
Feature extraction for reranking the graph-expanded candidate pool.

Every feature is computable from (issue text, repository code graph)
alone. No LLM, no test execution, no gold information.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass

import numpy as np

from graphloc_core import ident_tokens, qualified_name

FEATURE_NAMES = [
    "lex_norm",        # normalized BM25 of the node
    "hop",             # hops from seed set (0,1,2,3)
    "consensus",       # number of distinct seeds reaching the node
    "seed_flag",       # node is itself a seed
    "degree_norm",     # log-degree in the call graph
    "in_degree",       # how many distinct callers
    "out_degree",      # how many distinct callees
    "same_file_seed",  # file contains a seed
    "name_in_issue",   # qualified name token appears verbatim in issue
    "sig_overlap",     # jaccard(name tokens, issue tokens)
    "is_private",      # leading underscore
    "is_dunder",       # __x__
    "is_test_adjacent",# file path looks like a test file
    "n_args",          # signature arity
    "size_norm",       # span length (proxy for function size)
    "is_method",       # nested under a class
    "reach_mass",      # sum of decaying seed mass reaching the node
    "rank_in_pool",    # normalized lexical rank inside the pool
]

_CODEY = re.compile(r"[`~]|:[0-9]+|\(\)|\bdef\b|\bclass\b")
_TESTPATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")


def _safe_log(x: float) -> float:
    return math.log1p(max(float(x), 0.0))


def node_features(g, i: int, issue_tokens: set[str], seed_set: set[int],
                  label: np.ndarray, cons: np.ndarray, reach: np.ndarray,
                  lex: np.ndarray, lex_max: float, pool_rank: dict[int, int],
                  pool_n: int) -> list[float]:
    nid = g.node_ids[i]
    qn = qualified_name(nid)
    name_toks = set(ident_tokens(qn))
    sig = g.sigs[i] or ""
    sig_toks = set(ident_tokens(sig))
    inter = name_toks & issue_tokens
    union = name_toks | issue_tokens
    deg = len(g.adj[i])
    out_n = len(g.out_edges[i])
    in_n = deg - out_n if deg > out_n else 0
    arity = sig.count(",") + 1 if sig.strip().startswith(("def ", "async def ")) else 0
    f = g.files[i]
    same_file = any(g.files[s] == f for s in seed_set) if seed_set else False
    return [
        float(lex[i]) / (lex_max or 1.0),
        float(label[i]),
        _safe_log(cons[i]),
        1.0 if label[i] == 0 else 0.0,
        _safe_log(deg) / 10.0,
        _safe_log(in_n) / 10.0,
        _safe_log(out_n) / 10.0,
        1.0 if same_file else 0.0,
        1.0 if (name_toks & issue_tokens) else 0.0,
        (len(inter) / len(union)) if union else 0.0,
        1.0 if qn.startswith("_") else 0.0,
        1.0 if (qn.startswith("__") and qn.endswith("__")) else 0.0,
        1.0 if _TESTPATH.search(f) else 0.0,
        min(arity, 12) / 12.0,
        min(_safe_log((g.adj[i].size if hasattr(g.adj[i],'size') else 0)), 6.0) / 6.0,
        1.0 if ("::" in nid and nid.count("::") >= 1 and not nid.split("::")[-1].startswith("__")) else 0.0,
        float(reach[i]),
        pool_rank.get(i, pool_n) / max(pool_n, 1),
    ]


def pool_features(g, issue: str, seed_k: int, hops: int, seed_order: np.ndarray,
                  label: np.ndarray, cons: np.ndarray, lex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pool_indices, feature_matrix) for the expanded candidate pool."""
    issue_tokens = set(ident_tokens(issue))
    seed_set = {int(s) for s in seed_order[:seed_k]}
    keep = (label >= 0) & (~g.is_test)
    pool = np.flatnonzero(keep)
    if pool.size == 0:
        return pool, np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    lex_max = float(lex[pool].max() or 1.0)
    # decaying reach mass from seeds
    reach = np.zeros(g.n_nodes, dtype=np.float32)
    for h in range(0, hops + 1):
        sel = label == h
        reach[sel] += 0.5 ** h
    order = np.argsort(-lex[pool], kind="stable")
    pool_rank = {int(pool[o]): r for r, o in enumerate(order)}
    X = np.asarray([node_features(g, int(i), issue_tokens, seed_set, label, cons,
                                  reach, lex, lex_max, pool_rank, pool.size)
                    for i in pool], dtype=np.float32)
    return pool, X
