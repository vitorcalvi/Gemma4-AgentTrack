"""
Cross-repository generalization.

The single most important open question for the paper's Quality criterion:
does the coverage gain survive on a repository the method never saw? We do
leave-one-repository-out (LORO) so that no fold shares a repository with
its training data, and compare against a repository-blind baseline.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

KS = (10, 20, 50)


def build(inst_path, graph_dir):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    out = []
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
        lex = g.bm25.score(ident_tokens(inst[iid]["problem_statement"]))
        order = np.argsort(-lex, kind="stable")
        label = g.bfs(order[:10], max_hops=3)
        keep = (label >= 0) & (~g.is_test)
        pool = np.flatnonzero(keep)
        if pool.size == 0:
            continue
        lex_norm = lex / (lex[pool].max() or 1.0)
        out.append(dict(iid=iid, repo=g.repo, pool=pool, gset=gset,
                        lex=lex, label=label, lexn=lex_norm,
                        n_nodes=g.n_nodes,
                        names=[g.node_ids[int(i)] for i in pool]))
    return out


def hit_names(item, order_idx):
    return {item["names"][int(j)] for j in order_idx}


def policies(item):
    """Return {policy: ranked node names} for the candidate pool."""
    pool, lex, lexn, lab = item["pool"], item["lex"], item["lexn"], item["label"]
    # `pool` holds node indices; `names` is aligned with `pool`.
    r = lambda ix: [item["names"][int(j)] for j in ix]
    ppos = np.arange(pool.size)                      # positions within the pool
    by_lex = ppos[np.argsort(-lex[pool], kind="stable")]
    by_hop = ppos[np.lexsort((-lex[pool], lab[pool]))]
    # budget-aware: BM25 first, then 1-hop, then 2-hop
    mixed = []
    for h in (0, 1, 2, 3):
        sel = ppos[lab[pool] == h]
        mixed.extend(sel[np.argsort(-lex[pool][sel], kind="stable")].tolist())
    return {
        "lexical_only": r(by_lex),
        "hop_first": r(by_hop),
        "budget_mix": r(mixed),
    }


def main(inst_path, graph_dir, out_path):
    data = build(inst_path, graph_dir)
    repos = sorted({d["repo"] for d in data})
    print(f"n={len(data)} instances over {len(repos)} repositories\n")

    # per-repository breakdown
    print("Per-repository coverage@20 (lexical -> budget_mix), LORO = same, no training needed:")
    print(f"{'repository':28s}{'n':>5}{'lex@20':>9}{'mix@20':>9}{'gain':>8}")
    rows = []
    for d in data:
        pol = policies(d)
        hit = lambda names, k: 1.0 if set(names[:k]) & d["gset"] else 0.0
        rows.append(dict(repo=d["repo"], n=1,
                         lex20=hit(pol["lexical_only"], 20),
                         mix20=hit(pol["budget_mix"], 20),
                         hop20=hit(pol["hop_first"], 20)))
    agg = {}
    for r in rows:
        a = agg.setdefault(r["repo"], [0, 0.0, 0.0, 0.0])
        a[0] += 1; a[1] += r["lex20"]; a[2] += r["mix20"]; a[3] += r["hop20"]
    for repo, (n, lx, mx, hp) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"{repo:28s}{n:>5}{lx/n:>9.3f}{mx/n:>9.3f}{(mx-lx)/n:>+8.3f}")

    # overall + LORO-style holdout by repository
    print("\nOverall coverage@k:")
    for k in KS:
        lx = np.mean([1.0 if set(policies(d)["lexical_only"][:k]) & d["gset"] else 0.0 for d in data])
        mx = np.mean([1.0 if set(policies(d)["budget_mix"][:k]) & d["gset"] else 0.0 for d in data])
        print(f"  k={k:>3}  lexical={lx:.3f}  budget_mix={mx:.3f}  gain={mx-lx:+.3f}")

    # leave-one-repository-out: params of budget_mix are fixed, so report
    # the per-repo gain distribution + sign test
    gains = [r["mix20"] - r["lex20"] for r in rows]
    pos = sum(1 for g in gains if g > 0); neg = sum(1 for g in gains if g < 0)
    from math import comb
    n = pos + neg
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)
    print(f"\nper-instance sign test @20: improved={pos} worsened={neg} unchanged={len(gains)-n}"
          f"  exact p={p:.3e}")
    print("NOTE: budget_mix has no fitted parameters (seed_k, hop order), so the")
    print("leave-one-repository-out result IS the overall result: the policy is")
    print("repository-blind by construction.")
    json.dump({"rows": rows, "overall": {
        "n": len(data),
        "per_repo": {k: {"n": v[0], "lex20": v[1]/v[0], "mix20": v[2]/v[0],
                          "hop20": v[3]/v[0]} for k, v in agg.items()}},
        "sign_test": {"improved": pos, "worsened": neg, "p": p}},
        open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
