"""
Hop-distance profile of the patch target from lexical seeds.

Answers: how deep does an agent have to look in the call graph to reach
the function a maintainer actually edited? This is the quantity that
determines whether a fixed-depth neighborhood expansion is sufficient.
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

def bfs_multi(g, seeds, max_hops=8):
    dist = np.full(g.n_nodes, -1, dtype=np.int16)
    fr = []
    for s in seeds:
        i = g.index.get(s)
        if i is not None and dist[i] < 0:
            dist[i] = 0; fr.append(i)
    for h in range(1, max_hops + 1):
        nxt = []
        for i in fr:
            for j in g.adj[i]:
                if dist[j] < 0:
                    dist[j] = h; nxt.append(int(j))
        if not nxt: break
        fr = nxt
    return dist

def main(inst_path, graph_dir, out_path, k=10):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    hist = collections.Counter()
    per_repo = collections.defaultdict(collections.Counter)
    rows = []
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst: continue
        g = load_graph(path)
        gix = [g.index[x] for x in inst[iid]["gold"]["entities"] if x in g.index]
        if not gix: continue
        order = np.argsort(-g.bm25.score(ident_tokens(inst[iid]["problem_statement"])), kind="stable")
        d = bfs_multi(g, [g.node_ids[i] for i in order[:k]], max_hops=8)
        gd = int(min(d[i] for i in gix))
        lab = "unreachable" if gd < 0 else str(gd)
        hist[lab] += 1
        per_repo[g.repo][lab] += 1
        rows.append({"instance_id": iid, "repo": g.repo, "hops": gd})
        print(".", end="", flush=True)
    n = sum(hist.values())
    print(f"\nseed set = BM25 top-{k}\nn = {n} instances")
    print("\nhops from lexical seeds to true patch target:")
    for h in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "unreachable"]:
        c = hist.get(h, 0)
        if c: print(f"  {h:>12}: {c:4d}  ({100*c/n:5.1f}%)")
    cum = 0
    print("\ncumulative reach:")
    for h in range(0, 9):
        cum += hist.get(str(h), 0)
        print(f"  <= {h} hops: {100*cum/n:5.1f}%")
    print("\nmedian hops (reachable only):", np.median([r["hops"] for r in rows if r["hops"] >= 0]))
    json.dump(rows, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], k=int(sys.argv[4]) if len(sys.argv) > 4 else 10)
