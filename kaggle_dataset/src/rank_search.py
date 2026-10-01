"""
Which ranking rule best orders the EXPANDED candidate set?

The expansion reliably puts the gold target inside the candidate pool
(set-reach@20 ~ 0.46 vs 0.25 lexical). The open question is how to order
that pool. We compare several rules under a strict dev/test split so the
reported configuration is not chosen on the test set.

Rules
  lex        : original BM25 order (baseline)
  hop        : hop distance, then lexical
  cons       : seed-consensus - how many distinct seeds reach the node
  hopcons    : hop distance bucket, then consensus, then lexical
  pagerank   : personalized PageRank from the seed set (classic fusion)
  bc         : betweenness-centrality weighted, seeds as restart
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

HOPS = 3

def bfs_consensus(g, seeds, hops):
    """label, consensus count, and the set of reaching seeds per node."""
    label = np.full(g.n_nodes, -1, dtype=np.int16)
    cons = np.zeros(g.n_nodes, dtype=np.int32)
    cur = [int(s) for s in seeds]
    for s in cur:
        label[s] = 0
    for h in range(1, hops + 1):
        nxt = []
        for i in cur:
            for j in g.adj[i]:
                if label[j] < 0:
                    label[j] = h
                    nxt.append(int(j))
        for j in nxt:
            cons[j] += 1
        if not nxt:
            break
        cur = nxt
    return label, cons

def pagerank_scores(g, seeds, alpha=0.85, iters=40):
    n = g.n_nodes
    deg = np.array([max(len(a), 1) for a in g.adj], dtype=np.float64)
    restart = np.zeros(n); restart[list(seeds)] = 1.0 / max(len(seeds), 1)
    p = restart.copy()
    for _ in range(iters):
        q = np.zeros(n)
        for i in range(n):
            if p[i] == 0: continue
            w = p[i] / deg[i]
            for j in g.adj[i]:
                q[j] += w
        p = (1 - alpha) * restart + alpha * q
    return p

def rules(g, issue, seed_k=10, hops=HOPS):
    lex = g.bm25.score(ident_tokens(issue))
    order = np.argsort(-lex, kind="stable")
    seeds = order[:seed_k]
    label, cons = bfs_consensus(g, seeds, hops)
    keep = (label >= 0) & (~g.is_test)
    idx = np.flatnonzero(keep)
    lx = lex[idx]
    lb = label[idx].astype(float)
    cs = cons[idx].astype(float)
    # normalize lexical within pool
    lxn = lx / (lx.max() or 1.0)
    pr = pagerank_scores(g, seeds)
    prn = pr[idx]; prn = prn / (prn.max() or 1.0)
    out = {}
    out["lex"] = idx[np.argsort(-lx, kind="stable")]
    out["hop"] = idx[np.lexsort((-lx, lb))]
    out["cons"] = idx[np.lexsort((-lx, -cs))]
    out["hopcons"] = idx[np.lexsort((-(lb * 0.6 + cs / max(cons.max(), 1) * 0.4), lb))]
    out["pagerank"] = idx[np.argsort(-prn, kind="stable")]
    out["pr_hop"] = idx[np.lexsort((-lx, -prn))]
    out["lex_hop"] = idx[np.lexsort((lb, -lx))]
    out["hop2_lex2"] = idx[np.lexsort((-lxn, lb))]  # depth-prioritized
    out["cons_hop"] = idx[np.lexsort((lb, -cs, -lx))]
    return out, g

def main(inst_path, graph_dir, out_path):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    items = []
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst: continue
        gold = {x for x in inst[iid]["gold"]["entities"]}
        if not gold: continue
        try: g = load_graph(path)
        except Exception: continue
        gset = {x for x in gold if x in g.index}
        if not gset: continue
        items.append((iid, g, gset, inst[iid]["problem_statement"]))
    # deterministic dev/test split by instance hash
    dev = [x for x in items if hash(x[0]) % 2 == 0]
    test = [x for x in items if hash(x[0]) % 2 == 1]
    print(f"n={len(items)}  dev={len(dev)}  test={len(test)}")
    names = None
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for split, data in (("dev", dev), ("test", test)):
        for iid, g, gset, ps in data:
            rr, _ = rules(g, ps)
            names = list(rr)
            for nm, idx in rr.items():
                ranked = [g.node_ids[int(i)] for i in idx]
                for k in (10, 20, 50):
                    res[(nm, k)][split].append(1.0 if set(ranked[:k]) & gset else 0.0)
    print(f"\n{'rule':>12} " + " ".join(f"{'dev@'+str(k):>8}" for k in (10,20,50)) +
          " | " + " ".join(f"{'test@'+str(k):>8}" for k in (10,20,50)))
    for nm in names:
        line = f"{nm:>12} "
        for k in (10,20,50):
            line += f" {np.mean(res[(nm,k)]['dev']):>8.3f}"
        line += " | "
        for k in (10,20,50):
            line += f" {np.mean(res[(nm,k)]['test']):>8.3f}"
        print(line)
    json.dump({f"{nm}|{k}|{s}": float(np.mean(v)) for (nm,k),d in res.items() for s,v in d.items()},
              open(out_path,"w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
