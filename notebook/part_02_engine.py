"""Part 2 — engine, task structure, coverage, lexical difficulty."""

CELLS = []
A = CELLS.append

A(md(r"""
### The engine

`_helpers.py` holds the complete implementation — identifier tokenizer,
BM25, graph loader, capped breadth-first expansion, and the two
statistics (Wilson intervals, exact McNemar) used throughout. It is pure
Python + numpy with no third-party dependency, so it runs identically on
Kaggle, Colab and a laptop.
"""))
A(code(r"""
import os as _os


def _load_helpers():
    # Prefer an attached dataset copy, then a local checkout, then inline.
    cands = [_os.environ.get("GRAPHLOC_HELPERS"),
             "/kaggle/input/graphloc-300-labels/graphloc_helpers.py",
             "/kaggle/input/graphloc-src/_helpers.py",
             _os.path.join(_os.getcwd(), "_helpers.py"),
             _os.path.join(_os.getcwd(), "notebook", "_helpers.py"),
             "/tmp/_helpers.py"]
    for c in cands:
        if c and _os.path.isfile(c):
            return c
    raise FileNotFoundError("_helpers.py not found")


HELPERS = _load_helpers()
exec(compile(open(HELPERS, encoding="utf-8").read(), HELPERS, "exec"))
print("engine loaded from:", HELPERS)

g0 = RepoGraph(_os.path.join(DATASET_GRAPH, "astropy__astropy-12907.json"))
print("loaded:", g0.instance_id, "|", g0.repo)
print("non-test function/class nodes:", g0.n_nodes)
print("vocabulary size:", len(g0.bm25.vocab))
print("mean degree:", round(float(np.mean([len(a) for a in g0.adj])), 2))
"""))

A(md(r"""
## 3. What does a bug-fix patch actually touch?

Before asking how to *find* the target, we characterize the target.
This is the first piece of evidence that the problem is closer to
single-target search than to multi-site editing.
"""))
A(code(r"""
n_ent = [len(i["gold"]["entities"]) for i in INSTANCES.values()]
n_file = [len(i["gold"]["files"]) for i in INSTANCES.values()]
N = len(n_ent)


def pct(cond):
    return f"{100 * sum(cond) / N:.1f} %"


print(f"N = {N} instances\n")
print(f"exactly one gold function            : {pct([v == 1 for v in n_ent])}")
print(f"exactly one gold file                : {pct([v == 1 for v in n_file])}")
print(f"more than three gold functions       : {pct([v > 3 for v in n_ent])}")
print(f"NO pre-existing function (add-only)  : {pct([v == 0 for v in n_ent])}")
print(f"\nmean gold functions per task         : {np.mean(n_ent):.2f}")
print(f"median                               : {np.median(n_ent):.0f}")
print(f"max                                  : {max(n_ent)}")
"""))

A(md(r"""
**Result.** Localization here is overwhelmingly a **single-target
needle-in-a-haystack** problem: four out of five patches modify exactly
one function in exactly one file. A method built for "find all the
relevant places" is solving a harder problem than the benchmark poses.
"""))

A(md(r"""
### Graph coverage: is the target even present?

If the gold function were missing from the provided graph, no graph
method could work and the premise collapses. We verify that it is not.
"""))
A(code(r"""
cov_rows = []
for iid in sorted(INSTANCES):
    p = os.path.join(DATASET_GRAPH, iid + ".json")
    if not os.path.exists(p):
        continue
    gold = INSTANCES[iid]["gold"]["entities"]
    if not gold:
        continue
    g = RepoGraph(p)
    present = [x for x in gold if x in g.index]
    cov_rows.append((iid, len(present) / len(gold)))

cov = np.array([c for _, c in cov_rows])
print(f"instances with a graph on disk     : {len(cov_rows)}")
print(f"mean target coverage               : {cov.mean():.4f}")
print(f"instances fully covered            : {(cov == 1.0).mean():.1%}")
print(f"instances with NO in-graph target  : {(cov == 0.0).mean():.1%}")
"""))

A(md(r"""
**Result.** Coverage is essentially complete. The organizers' graphs are
not the bottleneck — the gold function is present in the graph
essentially always. What is missing is a way to *choose* it.
"""))

A(md(r"""
## 4. Lexical retrieval alone is a weak localizer

We build a standard BM25 index over identifier-subtokenized function
names and signatures — the natural first thing any agent does, and the
control condition for everything that follows. Identifiers are split on
camelCase and snake_case so that `separability_matrix` and
`separabilityMatrix` match.
"""))
A(code(r"""
def load_all():
    graphs, issues, golds = {}, {}, {}
    for iid in sorted(INSTANCES):
        p = os.path.join(DATASET_GRAPH, iid + ".json")
        if not os.path.exists(p):
            continue
        g = RepoGraph(p)
        gold = [x for x in INSTANCES[iid]["gold"]["entities"] if x in g.index]
        if not gold:
            continue
        graphs[iid] = g
        issues[iid] = INSTANCES[iid]["problem_statement"]
        golds[iid] = set(gold)
    return graphs, issues, golds


GRAPHS, ISSUES, GOLDS = load_all()
print("usable instances :", len(GRAPHS))
print("median candidates:", int(np.median([g.n_nodes for g in GRAPHS.values()])))

lex_rank = {}
for iid, g in GRAPHS.items():
    lex = g.bm25.score(ident_tokens(ISSUES[iid]))
    order = np.argsort(-lex, kind="stable")
    pos = {g.node_ids[int(i)]: r for r, i in enumerate(order)}
    lex_rank[iid] = min(pos.get(x, 10 ** 9) for x in GOLDS[iid])

vals = np.array(list(lex_rank.values()))
print("\nlexical rank of the gold function (1 = top hit):")
for q in (10, 25, 50, 75, 90, 99):
    print(f"  p{q:<3d} {np.percentile(vals, q):>9.0f}")
for t in (10, 100, 1000, 5000):
    print(f"  fraction ranked worse than {t:>5d}: {(vals > t).mean():.1%}")
"""))

A(md(r"""
The distribution is **bimodal in effect if not in shape**: most tasks
are easy for BM25 (median rank 17) but the tail is catastrophic. Roughly
a third of tasks have their target outside the top 100, and 7.7 % outside
the top 5 000.

**This motivates everything that follows.** A hard threshold on lexical
rank either keeps almost everything — flooding the agent with candidates
— or discards the hard third entirely.
"""))
