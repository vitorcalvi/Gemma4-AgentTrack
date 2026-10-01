"""
Full evaluation suite: seed expansion vs. lexical-only baseline.

Produces the exact numbers reported in the paper, plus a hop-depth
ablation and paired significance tests.
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens
from graphloc import localize, Config

KS = (1, 5, 10, 20, 50, 100)

def acc_at(ranked_names, gold, k):
    s = set(ranked_names[:k])
    return 1.0 if any(g in s for g in gold) else 0.0

def boot(d, e, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d, e = np.asarray(d), np.asarray(e)
    m = len(d); g = []
    for _ in range(n_boot):
        ix = rng.integers(0, m, m)
        g.append(e[ix].mean() - d[ix].mean())
    g = np.asarray(g)
    return float(g.mean()), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))

def mcnemar(d, e):
    from math import comb
    b = sum(1 for x, y in zip(d, e) if y > x)
    c = sum(1 for x, y in zip(d, e) if x > y)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return b, c, p

def evaluate(inst_path, graph_dir, out_path, ablate=True):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    rows = []
    t0 = time.time()
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst:
            continue
        gold = [x for x in inst[iid]["gold"]["entities"]]
        if not gold:
            continue
        try:
            g = load_graph(path)
        except Exception as e:
            print("ERR", iid, e, file=sys.stderr); continue
        gset = {x for x in gold if x in g.index}
        if not gset:
            continue
        lex = g.bm25.score(ident_tokens(inst[iid]["problem_statement"]))
        order = np.argsort(-lex, kind="stable")
        lex_names = [g.node_ids[int(i)] for i in order]
        r = {"instance_id": iid, "repo": g.repo, "n_nodes": g.n_nodes,
             "load_s": 0.0, "localize_s": 0.0}
        for k in KS:
            r[f"lex_acc@{k}"] = acc_at(lex_names, gset, k)
        if ablate:
            for h in (1, 2, 3):
                for sk in (5, 10, 20):
                    tl = time.time()
                    p = localize(g, inst[iid]["problem_statement"], Config(seed_k=sk, hops=h))
                    if (sk, h) == (10, 2):
                        r["localize_s"] = time.time() - tl
                    names = [x["node"] for x in p.ranked]
                    for k in KS:
                        r[f"gl_h{h}k{sk}@{k}"] = acc_at(names, gset, k)
        rows.append(r); print(".", end="", flush=True)
    el = time.time() - t0
    n = len(rows)
    print(f"\ninstances={n}  wall={el:.0f}s  mean_localize={np.mean([r['localize_s'] for r in rows]):.4f}s\n")

    main = "gl_h2k10"
    hdr = f"{'k':>5} {'lexical':>9} {'graphloc':>9} {'delta':>9}  {'95% CI of delta':>22}  {'McNemar p':>11}"
    print(hdr); print("-" * len(hdr))
    summary = {"n": n, "mean_localize_s": float(np.mean([r["localize_s"] for r in rows]))}
    for k in KS:
        d = [r[f"lex_acc@{k}"] for r in rows]
        e = [r[f"{main}@{k}"] for r in rows]
        a, b = float(np.mean(d)), float(np.mean(e))
        m, lo, hi = boot(d, e, 4000)
        _, _, p = mcnemar(d, e)
        summary[f"lex_acc@{k}"] = a; summary[f"gl_acc@{k}"] = b
        print(f"{k:>5} {a:>9.3f} {b:>9.3f} {b-a:>+9.3f}  {f'[{lo:+.3f}, {hi:+.3f}]':>22}  {p:>11.2e}")

    if ablate:
        print("\nhop-depth x seed-size ablation (acc@20):")
        for h in (1, 2, 3):
            line = [f"  h={h}"]
            for sk in (5, 10, 20):
                v = float(np.mean([r[f"gl_h{h}k{sk}@20"] for r in rows]))
                line.append(f"k={sk}:{v:.3f}")
                summary[f"gl_h{h}k{sk}@20"] = v
            print("  ".join(line))
        print(f"  lexical  acc@20: {float(np.mean([r['lex_acc@20'] for r in rows])):.3f}")

    json.dump({"rows": rows, "summary": summary}, open(out_path, "w"), indent=1)
    return summary

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
