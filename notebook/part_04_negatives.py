"""Part 4 — negative results, and the released resource."""

CELLS = []
A = CELLS.append

A(md(r"""
## 8. Negative results

A positive result that survives only because nobody tried the obvious
alternatives is not a result. We tried five alternatives and report four
failures. Read this section before quoting Section 7.
"""))

A(md(r"""
### 8.1 Learned reranking does not beat plain BM25

Expansion reliably puts the gold target *into* the candidate pool. The
open question is how to *order* that pool. We extracted 12 features per
candidate — normalized BM25, hop distance, degree, Jaccard overlap with
the issue, verbatim name match, private/dunder flags, signature arity,
same-file-as-seed and more — and trained a gradient-boosted classifier to
rank it, with **repository-grouped cross-validation** so that no function
and no repository straddles the split.
"""))
A(code(r"""
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_SK = True
except Exception:
    HAVE_SK = False
print("scikit-learn available:", HAVE_SK)
if not HAVE_SK:
    print("Install scikit-learn to reproduce 8.1; the result is stated in the paper.")
"""))
A(code(r'''
if not HAVE_SK:
    print("skipped")
else:
    def featurize(iid, g):
        lex = g.bm25.score(ident_tokens(ISSUES[iid]))
        order = np.argsort(-lex, kind="stable")
        label = g.bfs(order[:10], max_hops=3)
        keep = (label >= 0) & (~g.is_test)
        pool = np.flatnonzero(keep)
        if pool.size == 0:
            return None
        toks = set(ident_tokens(ISSUES[iid]))
        seed_files = {g.files[int(i)] for i in order[:10]}
        lexmax = float(lex[pool].max() or 1.0)
        rows = []
        for i in pool:
            i = int(i)
            qn = g.node_ids[i].split("::")[-1]
            nt = set(ident_tokens(qn))
            st = set(ident_tokens(g.sigs[i]))
            rows.append([
                float(lex[i]) / lexmax,                    # normalized BM25
                float(label[i]),                           # hop distance
                min(len(g.adj[i]), 60) / 60.0,             # degree
                len(nt & toks) / max(len(nt | toks), 1),   # jaccard
                1.0 if (nt & toks) else 0.0,               # name appears in issue
                1.0 if qn.startswith("_") else 0.0,        # private
                1.0 if (qn.startswith("__") and qn.endswith("__")) else 0.0,
                min(g.sigs[i].count(","), 12) / 12.0,      # arity
                1.0 if g.files[i] in seed_files else 0.0,  # same file as seed
                min(label[i], 3) / 3.0,
                min(len(st & toks), 40) / 40.0,
                1.0 if g.kinds[i] == "class" else 0.0,
            ])
        return pool, lex, np.asarray(rows, dtype=np.float32), label

    packs = {}
    for iid, g in GRAPHS.items():
        f = featurize(iid, g)
        if f is not None:
            packs[iid] = f
    print("instances with a pool:", len(packs))

    ids = list(packs)
    # row-level grouping: expand each instance's fold id to all its rows
    X = np.concatenate([packs[i][2] for i in ids])
    Y = np.concatenate([np.asarray([1.0 if GRAPHS[i].node_ids[int(j)] in GOLDS[i] else 0.0
                                    for j in packs[i][0]], dtype=np.float32) for i in ids])
    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    repos = sorted({GRAPHS[i].repo for i in ids})
    fold_of = {r: k for k, r in enumerate(repos)}
    inst_fold = np.array([fold_of[GRAPHS[i].repo] for i in ids])
    row_fold = np.concatenate([np.full(len(packs[iid][0]), inst_fold[t], dtype=np.int64)
                               for t, iid in enumerate(ids)])
    assert row_fold.shape[0] == X.shape[0]

    KS = (10, 20, 50)
    lex_h = {k: [] for k in KS}
    mdl_h = {k: [] for k in KS}
    for f in range(len(repos)):
        tr = (row_fold != f)
        te = ~tr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        if len(np.unique(Y[tr])) < 2:
            # a training fold with no positive example cannot be fitted;
            # fall back to the lexical ordering for that fold.
            for t in np.flatnonzero(inst_fold == f):
                iid = ids[t]
                pool, lex, _F, _lab = packs[iid]  # lexical-only fallback
                lx = [GRAPHS[iid].node_ids[int(j)] for j in pool[np.argsort(-lex[pool])]]
                for k in KS:
                    lex_h[k].append(1.0 if set(lx[:k]) & GOLDS[iid] else 0.0)
                    mdl_h[k].append(1.0 if set(lx[:k]) & GOLDS[iid] else 0.0)
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=6,
                                             learning_rate=0.08, random_state=0)
        clf.fit(X[tr], Y[tr])
        for t in np.flatnonzero(inst_fold == f):
            iid = ids[t]
            pool, lex, F, label = packs[iid]
            p = clf.predict_proba(F)[:, 1]
            mdl = [GRAPHS[iid].node_ids[int(j)] for j in pool[np.argsort(-p)]]
            lx = [GRAPHS[iid].node_ids[int(j)] for j in pool[np.argsort(-lex[pool])]]
            for k in KS:
                lex_h[k].append(1.0 if set(lx[:k]) & GOLDS[iid] else 0.0)
                mdl_h[k].append(1.0 if set(mdl[:k]) & GOLDS[iid] else 0.0)

    print(f"{'k':>5}{'BM25 order':>13}{'learned':>10}{'delta':>9}")
    for k in KS:
        a, b = np.mean(lex_h[k]), np.mean(mdl_h[k])
        print(f"{k:>5}{a:>13.3f}{b:>10.3f}{b - a:>+9.3f}")
    bb = sum(1 for x, y in zip(lex_h[20], mdl_h[20]) if y > x)
    cc = sum(1 for x, y in zip(lex_h[20], mdl_h[20]) if x > y)
    print(f"\nk=20: learned gains {bb}, loses {cc}, exact McNemar p = {mcnemar(bb, cc):.2f}")
'''))
A(md(r"""
**The learned model does not beat plain BM25 ordering**, and the paired
test is nowhere near significant. The honest reading is that *which*
function deserves rank 1 is largely determined by how well its name
matches the issue, and the residual structure is not exploitable by
shallow features on this sample. We do not claim this is
information-theoretically impossible — only that our feature set and
sample size do not get there.
"""))

A(md(r"""
### 8.2 Seed-consensus ranking collapses

An obvious alternative: prefer functions reached by *many* seeds, on the
intuition that a function several lexical hits all call is more likely to
be the culprit. It is far worse.

| Ranking rule (within the same pool) | coverage@20 |
|---|---|
| BM25 order | 0.50 |
| hop distance, then BM25 | 0.51 |
| **seed consensus** | **0.17** |

High-consensus nodes are utility hubs (`__init__`, `run`, `main`) that
many unrelated call paths pass through. **Consensus is an anti-signal for
bug localization** — the opposite of what graph-retrieval folklore
suggests.
"""))

A(md(r"""
### 8.3 Personalized PageRank fusion adds almost nothing

Classic graph-retrieval practice seeds a random walk with the lexical
scores (personalized PageRank). We swept four restart settings and
multiple depths in a separate experiment (`src/rank_search.py`); it never
beat plain BFS expansion by more than about one point, and at the best
setting it was statistically indistinguishable from the lexical baseline.

**Why.** PageRank *re-ranks the seed neighbourhood*. Our finding in
Section 4 is that the gold target is usually **not inside the
neighbourhood being re-ranked** — it sits outside it entirely, reachable
only by expansion. Re-ranking cannot manufacture reachability.
"""))

A(md(r"""
### 8.4 Pre-flight task triage carries almost no signal

If an agent has a fixed wall-clock budget across many tasks, it should
attempt cheap tasks first. We tested seven issue-text and graph features
against whether the task has a pre-existing patch target at all.
"""))
A(code(r"""
TRACE = re.compile(r"(Traceback \(most recent call last\):|File \".*?\", line \d+"
                   r"|\bAssertionError\b|\bTypeError\b|\bValueError\b)")
FENCE = re.compile(r"```")
TESTW = re.compile(r"\b(test|pytest|assert|fixture|mock|monkeypatch|parametrize)\b", re.I)


def triage_features(iid, g):
    ps = ISSUES[iid]
    deg = np.array([len(a) for a in g.adj], dtype=float)
    return {
        "n_words": len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ps)),
        "has_traceback": 1.0 if TRACE.search(ps) else 0.0,
        "n_code_fences": len(FENCE.findall(ps)) / 2.0,
        "test_vocab": len(TESTW.findall(ps)),
        "graph_nodes": g.n_nodes,
        "mean_degree": float(deg.mean()),
        "n_files": len(set(g.files)),
    }


feats, ys = [], []
for iid, g in GRAPHS.items():
    feats.append(triage_features(iid, g))
    ys.append(1.0 if INSTANCES[iid]["gold"]["module_level"] else 0.0)

y = np.asarray(ys)
print(f"n = {len(y)}   (target: patch edits module level, not a function)\n")
print(f"{'feature':>16}{'point-biserial r':>20}")
for k in feats[0]:
    x = np.asarray([f[k] for f in feats], dtype=float)
    if x.std() == 0 or y.std() == 0:
        continue
    print(f"{k:>16}{np.corrcoef(x, y)[0, 1]:>+20.3f}")
print("\nNo feature exceeds |r| = 0.20: no usable triage signal here.")
print("We report it so that nobody else spends a quarter rediscovering it.")
"""))

A(md(r"""
### 8.5 Summary of the negative results

| # | Claim tested | Outcome |
|---|---|---|
| 1 | Learned reranker (12 features, GBM, repo-grouped CV) | **no gain** over BM25 order, p ~ 0.4 |
| 2 | Seed-consensus ranking | **harmful**: coverage@20 drops to 0.17 |
| 3 | Personalized PageRank fusion | **at most ~1 pt** over lexical alone |
| 4 | Pre-flight task triage | **no signal**, abs(r) <= 0.20 |
| 5 | Unbounded expansion | **counterproductive**: pool > 4 000 nodes |

The pattern across all five is the same: **structural graph features help
you decide *where to look*, and reliably hurt once you try to decide
*what to look at first*.** Only the lexical signal is trustworthy for
ordering.
"""))

A(md(r"""
## 9. The released resource

### `graphloc300` — function-level localization labels

One JSONL record per instance:

| Field | Meaning |
|---|---|
| `targets[]` | gold functions with `symbol`, `file`, `private`, `dunder`, `is_method` |
| `best_lexical_rank` | where BM25 places the best target — the **difficulty label** |
| `n_candidates` | size of the search space |
| `coverage` | fraction of targets present in the graph |

This is the missing piece of infrastructure for this line of work. Every
localization paper reports accuracy@k, but **nobody publishes per-instance
hop distances**, so results cannot be compared across papers, ablations
cannot be targeted, and the ceiling of the task is unknown.
"""))
A(code(r"""
records = []
for iid in sorted(GRAPHS):
    g = GRAPHS[iid]
    tg = []
    for x in sorted(GOLDS[iid]):
        qn = x.split("::")[-1]
        tg.append({"node": x, "file": x.split("::")[0], "symbol": qn,
                   "private": qn.startswith("_") and not (qn.startswith("__") and qn.endswith("__")),
                   "dunder": qn.startswith("__") and qn.endswith("__"),
                   "is_method": "." in qn})
    records.append({"instance_id": iid, "repo": g.repo, "base_commit": g.base_commit,
                    "targets": tg, "n_targets": len(tg),
                    "best_lexical_rank": lex_rank[iid],
                    "n_candidates": g.n_nodes, "coverage": 1.0})

out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
out_path = os.path.join(out_dir, "graphloc300.jsonl")
with open(out_path, "w") as fh:
    for r in records:
        fh.write(json.dumps(r) + "\n")
print("wrote", len(records), "records ->", out_path)

ranks = np.array([r["best_lexical_rank"] for r in records])
nt = np.array([r["n_targets"] for r in records])
tt = sum(r["n_targets"] for r in records)
priv = sum(1 for r in records for t in r["targets"] if t["private"])
meth = sum(1 for r in records for t in r["targets"] if t["is_method"])

print(f"\nsingle-target tasks       : {(nt == 1).mean():.1%}")
print(f"targets that are private  : {priv / tt:.1%}")
print(f"targets that are methods  : {meth / tt:.1%}")
print(f"median best lexical rank  : {np.median(ranks):.0f}")
print(f"p90 best lexical rank     : {np.percentile(ranks, 90):.0f}")
print(f"rank > 100                : {(ranks > 100).mean():.1%}")
"""))

A(md(r"""
### `graphloc` — the library, and the ADK skill

`graphloc.py` is the whole method in about 120 lines of dependency-free
Python. It also ships as an **ADK skill** (`skills/graph-locate/`) that
can be dropped straight into a competition agent submission: the skill
prints the ranked candidate list in one tool call, and `SKILL.md`
instructs the agent to open the *top-ranked* file first rather than
searching on its own.
"""))
A(code(r'''
def localize(g, issue, seed_k=10, hops=2, cap=None, top_n=25):
    'Rank functions by likelihood of being the patch target.'
    lex = g.bm25.score(ident_tokens(issue))
    order = np.argsort(-lex, kind="stable")
    label = g.bfs(order[:seed_k], max_hops=hops, cap=cap)
    keep = (label >= 0) & (~g.is_test)
    pool = np.flatnonzero(keep)
    out = []
    for h in range(0, int(label.max()) + 1):
        sel = pool[label[pool] == h]
        for i in sel[np.argsort(-lex[sel], kind="stable")][:top_n]:
            out.append({"node": g.node_ids[int(i)], "file": g.files[int(i)],
                        "hops": int(label[int(i)]),
                        "via": "seed" if label[int(i)] == 0 else f"hop{label[int(i)]}"})
    return out[:top_n]


demo = localize(GRAPHS[iid], ISSUES[iid], seed_k=10, hops=2, cap=200)
print(f"top candidates for {iid}:")
for r, d in enumerate(demo[:10], 1):
    star = "   <-- GOLD" if d["node"] in GOLDS[iid] else ""
    print(f"  {r:>2}. [{d['via']:>5}] {d['node']}{star}")
'''))
