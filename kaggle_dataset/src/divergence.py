"""
Is the patch target lexically findable at all?

For each task we take the BM25 top-k functions as a *proxy* for what the
issue text points at ("the API surface the reporter talks about"), and ask
how far the true patch target sits from that set in the call graph.
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

def bfs(g, seeds, max_hops):
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
        fr = nxt
        if not fr: break
    return dist

def main(inst_path, graph_dir, out_path):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    rows = []
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(f)[:-5]
        if iid not in inst: continue
        try: g = load_graph(f)
        except Exception as e:
            print("ERR", iid, e, file=sys.stderr); continue
        gold = inst[iid]["gold"]["entities"]
        gix = [g.index[x] for x in gold if x in g.index]
        if not gix: continue
        q = ident_tokens(inst[iid]["problem_statement"])
        order = np.argsort(-g.bm25.score(q), kind="stable")
        r = dict(instance_id=iid, repo=g.repo, n=g.n_nodes)
        for k in (1, 3, 5, 10, 20, 50, 100):
            d = bfs(g, [g.node_ids[i] for i in order[:k]], max_hops=6)
            gd = int(min(d[i] for i in gix))
            r[f"h@{k}"] = gd if gd >= 0 else -1
        # how many gold targets are in the same file as any top-5 hit
        top5 = {g.files[i] for i in order[:5]}
        r["file_hit5"] = 1.0 if any(g.files[i] in top5 for i in gix) else 0.0
        rows.append(r); print(".", end="", flush=True)
    print(f"\ninstances={len(rows)}")
    for k in (1, 3, 5, 10, 20, 50, 100):
        h = np.array([r[f"h@{k}"] for r in rows])
        reach = float(np.mean((h >= 0) & (h <= 2)))
        reach6 = float(np.mean(h >= 0))
        never = float(np.mean(h < 0))
        print(f"k={k:4d}  within2hops={reach:.3f}  reachable={reach6:.3f}  UNREACHABLE={never:.3f}")
    print("gold file in BM25 top-5 files:", np.mean([r["file_hit5"] for r in rows]))
    json.dump(rows, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
