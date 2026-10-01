"""
Coverage-vs-budget curve.

For an agent that can only afford to look at B functions, the metric that
matters is: what fraction of bugs have the true target inside the
budgeted candidate set? This is a *retrieval* metric, deliberately
separate from ranking precision, because a human- or agent-in-the-loop
reviewer can filter a small set but cannot search a large one.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

BUDGETS = (1, 3, 5, 10, 20, 30, 50, 100, 200, 400, 800)
HOPS = (0, 1, 2, 3, 4)

def bfs(g, seeds, hops):
    label = np.full(g.n_nodes, -1, dtype=np.int16)
    cur = [int(s) for s in seeds]
    for s in cur:
        label[s] = 0
    for h in range(1, hops + 1):
        nxt = []
        for i in cur:
            for j in g.adj[i]:
                if label[j] < 0:
                    label[j] = h; nxt.append(int(j))
        if not nxt: break
        cur = nxt
    return label

def main(inst_path, graph_dir, out_path, seed_k=10):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    rows = []
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst: continue
        gold = {x for x in inst[iid]["gold"]["entities"]}
        if not gold: continue
        try: g = load_graph(path)
        except Exception: continue
        gset = {x for x in gold if x in g.index}
        if not gset: continue
        lex = g.bm25.score(ident_tokens(inst[iid]["problem_statement"]))
        order = np.argsort(-lex, kind="stable")
        seeds = order[:seed_k]
        r = {"instance_id": iid, "repo": g.repo, "n_nodes": g.n_nodes, "n_pool": 0}
        # lexical-only coverage (rank by BM25)
        lex_names = [g.node_ids[int(i)] for i in order]
        for b in BUDGETS:
            r[f"lex_cov@{b}"] = 1.0 if set(lex_names[:b]) & gset else 0.0
        for h in HOPS:
            label = bfs(g, seeds, h)
            keep = (label >= 0) & (~g.is_test)
            pool = np.flatnonzero(keep)
            r[f"n_pool_h{h}"] = int(pool.size)
            # depth-prioritized coverage: seeds first (BM25 order), then hop1, hop2...
            ordered = []
            for hh in range(0, h + 1):
                sel = pool[label[pool] == hh]
                ordered.extend(sel[np.argsort(-lex[sel], kind="stable")].tolist())
            names = [g.node_ids[int(i)] for i in ordered]
            for b in BUDGETS:
                r[f"exp{h}_cov@{b}"] = 1.0 if set(names[:b]) & gset else 0.0
        rows.append(r); print(".", end="", flush=True)
    n = len(rows)
    print(f"\nn = {n}\n")
    hdr = f"{'budget':>7} " + "".join(f"{'h='+str(h):>8}" for h in HOPS)
    print("BM25-seed coverage (target inside budgeted candidate set)"); print(hdr); print("-"*len(hdr))
    summary = {"n": n, "seed_k": seed_k, "budgets": list(BUDGETS)}
    for b in BUDGETS:
        line = f"{b:>7} "
        for h in HOPS:
            v = float(np.mean([r[f"exp{h}_cov@{b}"] for r in rows]))
            summary[f"exp{h}_cov@{b}"] = v
            line += f"{v:>8.3f}"
        print(line)
    print("\nmedian pool size by hop depth:")
    for h in HOPS:
        print(f"  h={h}: {np.median([r[f'n_pool_h{h}'] for r in rows]):.0f}")
        summary[f"n_pool_h{h}"] = float(np.median([r[f"n_pool_h{h}"] for r in rows]))
    json.dump({"summary": summary, "rows": rows}, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], seed_k=int(sys.argv[4]) if len(sys.argv)>4 else 10)
