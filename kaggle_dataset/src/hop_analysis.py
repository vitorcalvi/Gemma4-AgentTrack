"""Lexical-graph divergence and graph-reachable localization ceiling."""
from __future__ import annotations
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import RepoGraph, load_graph, ident_tokens, qualified_name

def bfs_hops(g: RepoGraph, seeds, max_hops=3, cap=20000):
    """Return min hop distance from any seed to each node (BFS, undirected)."""
    dist = np.full(g.n_nodes, -1, dtype=np.int16)
    frontier = []
    for s in seeds:
        i = g.index.get(s)
        if i is not None and dist[i] < 0:
            dist[i] = 0; frontier.append(i)
    seen = len(frontier)
    for h in range(1, max_hops + 1):
        nxt = []
        for i in frontier:
            for j in g.adj[i]:
                if dist[j] < 0:
                    dist[j] = h; nxt.append(int(j)); seen += 1
        frontier = nxt
        if seen > cap or not frontier:
            break
    return dist

def rank_of(order, idx):
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order)] = np.arange(len(order))
    return pos

def analyze(path, inst, topk=(1,3,5,10,20,50)):
    g = load_graph(path)
    gold = inst["gold"]["entities"]
    if not gold:
        return None
    q = ident_tokens(inst["problem_statement"])
    s = g.bm25.score(q)
    order = np.argsort(-s, kind="stable")
    pos = rank_of(order, np.arange(g.n_nodes))
    gold_ix = [g.index[x] for x in gold if x in g.index]
    if not gold_ix:
        return None
    best = int(min(pos[i] for i in gold_ix))
    # same file?
    gfiles = set(inst["gold"]["files"])
    file_of_gold = {g.files[i] for i in gold_ix}
    # file-level rank
    fq = ident_tokens(inst["problem_statement"])
    fs = g.file_bm25.score(fq)
    forder = np.argsort(-fs, kind="stable")
    fpos = rank_of(forder, np.arange(len(g.file_bm25.vocab)) if g.file_bm25.n else np.arange(0))
    fnames = sorted(set(g.files))
    fr = {fn: r for r, fn in enumerate(sorted(fnames, key=lambda f: -fs[g.file_bm25.vocab.get(f, -1)]))}
    gf_rank = min(fr.get(f, 10 ** 9) for f in file_of_gold)
    out = dict(instance_id=g.instance_id, repo=g.repo, n_nodes=g.n_nodes,
               gold_file_lex_rank=gf_rank,
               same_file_at_1=1.0 if gf_rank == 0 else 0.0)
    for k in topk:
        seeds = [g.node_ids[i] for i in order[:k]]
        hit = any(g.node_ids[i] == x for i in order[:k] for x in gold)
        out[f"bm25_acc@{k}"] = 1.0 if hit else 0.0
        d = bfs_hops(g, seeds, max_hops=3)
        gd = int(min(d[i] for i in gold_ix))
        out[f"hop@{k}"] = gd if gd >= 0 else 99
        out[f"reach3@{k}"] = 1.0 if 0 <= gd <= 3 else 0.0
    return out

if __name__ == "__main__":
    import glob
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(sys.argv[1])}
    rows = []
    for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.json"))):
        iid = os.path.basename(f)[:-5]
        if iid not in inst:
            continue
        try:
            r = analyze(f, inst[iid])
        except Exception as e:
            print("ERR", iid, e, file=sys.stderr); continue
        if r:
            rows.append(r); print(".", end="", flush=True)
    print(f"\ninstances={len(rows)}")
    ks = (1, 3, 5, 10, 20, 50)
    for k in ks:
        a = np.mean([r[f"bm25_acc@{k}"] for r in rows])
        r3 = np.mean([r[f"reach3@{k}"] for r in rows])
        hm = [r[f"hop@{k}"] for r in rows]
        m = np.mean([x for x in hm if x < 99]) if any(x < 99 for x in hm) else float("nan")
        print(f"k={k:3d}  bm25_acc={a:.3f}  graph_reach<=3={r3:.3f}  (+{100*(r3-a):.1f}pt)  mean_hops={m:.2f}")
    print("same-file@1 (lexical file rank 0):", np.mean([r["same_file_at_1"] for r in rows]))
    print("median gold file lexical rank:", np.median([r["gold_file_lex_rank"] for r in rows]))
    json.dump(rows, open(os.path.join(os.path.dirname(sys.argv[2]), "hop_rows.json"), "w"), indent=1)
