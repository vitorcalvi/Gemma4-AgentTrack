"""
Disentangle two things that were previously conflated:

  (A) SEED SIZE  - how many lexical hits seed the expansion.
  (B) ORDERING   - whether expanded nodes are ranked before or after the
                   remaining lexical hits.

coverage.py showed 0.379 -> 0.510 at B=50 by expanding the top-10 seeds
and listing hop layers first. cross_repo.py showed the same ordering
HURTS when compared against plain lexical. These are consistent only if
the gain came from (A) the *restricted candidate set*, not from (B) the
hop-first ordering. This script tests that directly.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

BUDGETS = (5, 10, 20, 30, 50, 100, 200, 400, 800)
HOPS = (0, 1, 2, 3)


def expand_pool(g, issue, seed_k, hops=3, cap=None):
    lex = g.bm25.score(ident_tokens(issue))
    order = np.argsort(-lex, kind="stable")
    label = g.bfs(order[:seed_k], max_hops=hops, cap=cap)
    keep = (label >= 0) & (~g.is_test)
    return lex, order, label, np.flatnonzero(keep)


def main(inst_path, graph_dir, out_path):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    res = {}
    sizes = []
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst:
            continue
        gold = {x for x in inst[iid]["gold"]["entities"] if x}
        if not gold:
            continue
        try:
            g = load_graph(path)
        except Exception:
            continue
        gset = {x for x in gold if x in g.index}
        if not gset:
            continue
        sizes.append((iid, g, inst[iid]["problem_statement"], gset))

    print(f"n={len(sizes)}\n")
    # A) seed-size sweep, hop-first ordering
    print("(A) SEED SIZE, hop-layer-first ordering -- coverage@B")
    hdr = f"{'B':>5}" + "".join(f"{'k=' + str(k):>9}" for k in (1, 5, 10, 20, 50, 100))
    print(hdr)
    for k in (1, 5, 10, 20, 50, 100):
        acc = {b: [] for b in BUDGETS}
        for iid, g, ps, gset in sizes:
            lex, order, label, pool = expand_pool(g, ps, k)
            if pool.size == 0:
                continue
            ranked = []
            for h in range(0, HOPS[-1] + 1):
                sel = pool[label[pool] == h]
                ranked.extend(sel[np.argsort(-lex[sel], kind="stable")].tolist())
            names = [g.node_ids[int(i)] for i in ranked]
            for b in BUDGETS:
                acc[b].append(1.0 if set(names[:b]) & gset else 0.0)
        res[f"seedK{k}"] = {str(b): float(np.mean(acc[b])) for b in BUDGETS}
        print(f"{'':>5}" + "".join(f"{'':>9}" for _ in ()) if False else
              f"{k:>5}" + "".join(f"{res[f'seedK{k}'][str(b)]:>9.3f}" for b in (10, 20, 50, 100, 200)))
    print("     (columns: B=10, 20, 50, 100, 200)")

    # B) ordering ablation at fixed seed k=10
    print("\n(B) ORDERING at fixed seed_k=10 -- coverage@B")
    for name, hopfirst in (("lexical_first", False), ("hop_first", True)):
        acc = {b: [] for b in BUDGETS}
        for iid, g, ps, gset in sizes:
            lex, order, label, pool = expand_pool(g, ps, 10)
            if pool.size == 0:
                continue
            if hopfirst:
                seq = []
                for h in range(0, HOPS[-1] + 1):
                    sel = pool[label[pool] == h]
                    seq.extend(sel[np.argsort(-lex[sel], kind="stable")].tolist())
            else:
                rest = np.setdiff1d(order, pool, assume_unique=False)
                seq = list(rest) + list(pool[np.argsort(-lex[pool], kind="stable")])
            names = [g.node_ids[int(i)] for i in seq]
            for b in BUDGETS:
                acc[b].append(1.0 if set(names[:b]) & gset else 0.0)
        res[f"order_{name}"] = {str(b): float(np.mean(acc[b])) for b in BUDGETS}
        print(f"  {name:16s}" + "".join(f"{res[f'order_{name}'][str(b)]:>9.3f}"
                                        for b in (10, 20, 50, 100, 200)))
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
