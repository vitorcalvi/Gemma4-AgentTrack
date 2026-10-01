"""
Is there ONE hop depth that works across repository sizes?

A fixed hop radius over-expands in big repositories. We test a
size-adaptive rule: expand until the pool exceeds a cap, i.e. adaptive
radius. Compared against fixed radius under equal pool budget.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

def bfs_limited(g, seeds, max_hops, cap):
    label = np.full(g.n_nodes, -1, dtype=np.int16)
    cur = [int(s) for s in seeds]
    for s in cur:
        label[s] = 0
    for h in range(1, max_hops + 1):
        if len(np.flatnonzero(label >= 0)) > cap:
            break
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
        r = {"instance_id": iid, "repo": g.repo, "n_nodes": g.n_nodes}
        for cap in (50, 100, 200, 400, 800, 10 ** 9):
            lab = bfs_limited(g, seeds, 12, cap)
            keep = (lab >= 0) & (~g.is_test)
            pool = np.flatnonzero(keep)
            ordered = []
            for hh in range(0, int(lab.max()) + 1):
                sel = pool[lab[pool] == hh]
                ordered.extend(sel[np.argsort(-lex[sel], kind="stable")].tolist())
            names = [g.node_ids[int(i)] for i in ordered]
            key = "inf" if cap > 10 ** 8 else str(cap)
            r[f"pool@{key}"] = int(pool.size)
            r[f"maxhop@{key}"] = int(lab.max())
            for b in (20, 50, 100, 200):
                r[f"cov@{key}@{b}"] = 1.0 if set(names[:b]) & gset else 0.0
        rows.append(r); print(".", end="", flush=True)
    n = len(rows)
    print(f"\nn = {n}\nadaptive pool cap -> coverage")
    caps = ["50","100","200","400","800","inf"]
    hdr = f"{'budget':>7} " + "".join(f"{'cap='+c:>9}" for c in caps)
    print(hdr); print("-"*len(hdr))
    summary = {"n": n, "seed_k": seed_k}
    for b in (20,50,100,200):
        line=f"{b:>7} "
        for c in caps:
            v=float(np.mean([r[f"cov@{c}@{b}"] for r in rows])); summary[f"cov@{c}@{b}"]=v
            line+=f"{v:>9.3f}"
        print(line)
    print("\nmean realized hop radius / pool size:")
    for c in caps:
        print(f"  cap={c:>4}: maxhop={np.mean([r[f'maxhop@{c}'] for r in rows]):.2f} "
              f"pool={np.mean([r[f'pool@{c}'] for r in rows]):.0f}")
    json.dump({"summary":summary,"rows":rows}, open(out_path,"w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], seed_k=int(sys.argv[4]) if len(sys.argv)>4 else 10)
