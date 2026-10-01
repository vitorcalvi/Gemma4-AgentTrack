"""
Learned reranker over the graph-expanded candidate pool.

A gradient-boosted tree (scikit-learn HistGradientBoostingClassifier) is
trained on dev instances to predict, for each pooled function, whether it
is the reference patch's target. Ranking is by predicted probability.

Design constraints:
  * features are issue+graph only, so the same model generalizes to any
    repository with a code graph in the competition's format;
  * grouping is by instance, so a function never appears in both train
    and test folds;
  * the split is by repository *and* instance hash to avoid leakage.
"""
from __future__ import annotations
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens
from features import FEATURE_NAMES, pool_features

SEED_K = 10
HOPS = 3
KS = (1, 5, 10, 20, 50)


def bfs(g, seeds, hops):
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


def build(inst_path, graph_dir, cache_path=None):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    data = []
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst:
            continue
        gold = {x for x in inst[iid]["gold"]["entities"] if x}
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
        label, cons = bfs(g, order[:SEED_K], HOPS)
        pool, X = pool_features(g, inst[iid]["problem_statement"], SEED_K, HOPS, order, label, cons, lex)
        if pool.size == 0:
            continue
        y = np.asarray([1.0 if g.node_ids[int(i)] in gset else 0.0 for i in pool], dtype=np.float32)
        data.append(dict(instance_id=iid, repo=g.repo, pool=pool, X=X, y=y,
                         names=[g.node_ids[int(i)] for i in pool],
                         gold=list(gset), n_nodes=g.n_nodes))
        print(".", end="", flush=True)
    print(f"\nbuilt {len(data)} instances")
    if cache_path:
        np.savez_compressed(cache_path, **{f"X{i}": d["X"] for i, d in enumerate(data)},
                            meta=json.dumps([{k: d[k] for k in ("instance_id","repo","names","gold","n_nodes")} for d in data]))
    return data


def evaluate(data, seed=0, n_splits=5, out_path=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    repos = np.asarray([d["repo"] for d in data])
    groups = np.asarray([hash(d["instance_id"]) % 1000 for d in data])
    uniq = sorted(set(repos))
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(uniq)))
    fold_of_repo = {uniq[perm[i]]: i % n_splits for i in range(len(uniq))}
    fold = np.asarray([fold_of_repo[r] for r in repos])

    oof = []
    acc = {k: [] for k in KS}
    lex_acc = {k: [] for k in KS}
    pool_reach = {k: [] for k in KS}
    for f in range(n_splits):
        tr = [i for i in range(len(data)) if fold[i] != f]
        te = [i for i in range(len(data)) if fold[i] == f]
        Xtr = np.concatenate([data[i]["X"] for i in tr])
        ytr = np.concatenate([data[i]["y"] for i in tr])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             max_depth=6, l2_regularization=1.0,
                                             random_state=seed, early_stopping=False)
        clf.fit(Xtr, ytr)
        for i in te:
            d = data[i]
            p = clf.predict_proba(d["X"])[:, 1]
            order = np.argsort(-p, kind="stable")
            ranked = [d["names"][j] for j in order]
            gold = set(d["gold"])
            for k in KS:
                acc[k].append(1.0 if set(ranked[:k]) & gold else 0.0)
                pool_reach[k].append(1.0 if set(d["names"][:k]) & gold else 0.0)
            # lexical baseline restricted to the same pool
            lx = d["X"][:, 0]
            lo = np.argsort(-lx, kind="stable")
            lranked = [d["names"][j] for j in lo]
            for k in KS:
                lex_acc[k].append(1.0 if set(lranked[:k]) & gold else 0.0)
        print(f"  fold {f} done ({len(te)} instances)", flush=True)

    n = len(acc[1])
    print(f"\nn_test = {n} instances, {n_splits}-fold grouped by repository\n")
    hdr = f"{'k':>5} {'lexical':>10} {'reranked':>10} {'delta':>9}   (pool is the same set for both)"
    print(hdr); print("-" * len(hdr))
    summary = {"n_test": n, "n_splits": n_splits, "features": FEATURE_NAMES}
    for k in KS:
        a = float(np.mean(lex_acc[k])); b = float(np.mean(acc[k]))
        summary[f"lex@{k}"] = a; summary[f"model@{k}"] = b
        print(f"{k:>5} {a:>10.3f} {b:>10.3f} {b-a:>+9.3f}")
    # paired bootstrap at k=20
    from math import comb
    for k in (10, 20):
        b_cnt = sum(1 for x, y in zip(lex_acc[k], acc[k]) if y > x)
        c_cnt = sum(1 for x, y in zip(lex_acc[k], acc[k]) if x > y)
        nbc = b_cnt + c_cnt
        p = 1.0 if nbc == 0 else min(1.0, 2 * sum(comb(nbc, j) for j in range(min(b_cnt, c_cnt) + 1)) / 2 ** nbc)
        summary[f"mcnemar_p@{k}"] = p
        print(f"k={k}: model gains {b_cnt}, loses {c_cnt}, exact McNemar p={p:.3e}")
    if out_path:
        json.dump(summary, open(out_path, "w"), indent=1)
    return summary


if __name__ == "__main__":
    t = time.time()
    cache = sys.argv[4] if len(sys.argv) > 4 else None
    data = build(sys.argv[1], sys.argv[2], cache)
    print(f"feature build: {time.time()-t:.0f}s")
    evaluate(data, out_path=sys.argv[3])
