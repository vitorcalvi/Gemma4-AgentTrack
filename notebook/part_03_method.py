"""Part 3 — case study and the method."""

CELLS = []
A = CELLS.append

A(md(r"""
## 5. Case study: why the tail exists

We look at one instance in detail, because the aggregate statistics hide
the mechanism.
"""))
A(code(r"""
iid = "astropy__astropy-12907"
g = GRAPHS[iid]
issue = ISSUES[iid]

print("ISSUE (first 400 chars):")
print(issue[:400].strip())
print("\nGOLD TARGET:", sorted(GOLDS[iid]))

lex = g.bm25.score(ident_tokens(issue))
order = np.argsort(-lex, kind="stable")
pos = {g.node_ids[int(i)]: r for r, i in enumerate(order)}
target = sorted(GOLDS[iid])[0]

print(f"\nBM25 rank of the gold target : {pos[target]:,} of {g.n_nodes:,}")
print("\nWhat BM25 actually surfaces (top 6):")
for i in order[:6]:
    print(f"  rank {pos[g.node_ids[int(i)]]:>5d}  {g.node_ids[int(i)]}")
"""))

A(md(r"""
The issue is written against the **public** function
`separability_matrix`. BM25 finds it immediately — it is the top hit, and
the reporter even quotes its docstring. But the maintainer's patch does
not touch it. The patch is in `_cstack`, a private helper.
"""))
A(code(r"""
seed = int(order[0])
lab = g.bfs([seed], max_hops=3)
ti = g.index[target]

print(f"seed         : {g.node_ids[seed]}")
print(f"gold target  : {target}")
print(f"hop distance : {lab[ti]}")

# reconstruct a shortest path seed -> target
path, cur = [target], ti
while cur != seed and lab[cur] > 0:
    prevs = [int(j) for j in g.adj[cur] if lab[int(j)] == lab[cur] - 1]
    if not prevs:
        break
    cur = prevs[0]
    path.append(g.node_ids[cur])

print("\nshortest call-graph path seed -> target:")
for i, n in enumerate(reversed(path)):
    print(f"  {i}  {n}")
"""))

A(md(r"""
**Public-API / private-implementation divergence.** The report and the
patch describe the *same bug* in two different vocabularies. Lexical
retrieval is structurally incapable of connecting them, because the only
thing linking `separability_matrix` and `_cstack` is a call edge in the
graph.

This is not an isolated curiosity. It is the mechanism behind the long
tail in Section 4.
"""))

A(md(r"""
## 6. The method: seed expansion

Given the above, the method is direct:

1. **Seed.** Rank candidate functions by BM25. Take the top `k` as `S`.
2. **Expand.** Grow `S` over the `calls` / `imports` / `inherits`
   subgraph for `h` hops, giving `N_h(S)`.
3. **Cap.** Stop if the pool exceeds `cap` nodes, so expansion stays
   affordable and comparable across repository sizes.
4. **Order.** Seeds first in BM25 order, then hop-1, then hop-2, each
   layer in BM25 order.

Step 3 matters: without it, expansion runs away. We quantify that in
Section 7.3.
"""))
A(code(r"""
def expand(g, issue, seed_k=10, hops=2, cap=None):
    'Return (ranked node ids, realized radius, pool size).'
    lex = g.bm25.score(ident_tokens(issue))
    order = np.argsort(-lex, kind="stable")
    label = g.bfs(order[:seed_k], max_hops=hops, cap=cap)
    keep = (label >= 0) & (~g.is_test)
    pool = np.flatnonzero(keep)
    ordered = []
    for h in range(0, int(label.max()) + 1):
        sel = pool[label[pool] == h]
        ordered.extend(sel[np.argsort(-lex[sel], kind="stable")].tolist())
    return [g.node_ids[int(i)] for i in ordered], int(label.max()), int(pool.size)


ranked, mh, psz = expand(GRAPHS[iid], ISSUES[iid], seed_k=10, hops=2, cap=200)
print("case study under the method (seeds top-10, 2 hops, cap 200):")
print("  gold target now at position:", ranked.index(target) + 1)
print("  was", f"{pos[target]:,}", "under BM25 alone")
print("  pool size:", psz, " realized radius:", mh)
"""))


A(md(r"""
## 7. Coverage versus budget — the main result

For an agent that can only afford to inspect a bounded number of
functions, the operative metric is **coverage**: what fraction of bugs
have their true target *inside the budgeted candidate set*? A reviewer —
human or agent — can filter a small set but cannot search a large one.

We sweep budget `B` and hop depth `h`.
"""))
A(code(r"""
BUDGETS = (1, 3, 5, 10, 20, 30, 50, 100, 200, 400, 800)
HOPS = (0, 1, 2, 3, 4)

cov = {h: {b: [] for b in BUDGETS} for h in HOPS}
pool_sizes = {h: [] for h in HOPS}
radii = {h: [] for h in HOPS}

t0 = time.time()
for iid, g in GRAPHS.items():
    for h in HOPS:
        ranked, mh, psz = expand(g, ISSUES[iid], seed_k=10, hops=h)
        pool_sizes[h].append(psz)
        radii[h].append(mh)
        gs = GOLDS[iid]
        for b in BUDGETS:
            cov[h][b].append(1.0 if set(ranked[:b]) & gs else 0.0)

print(f"computed in {time.time() - t0:.1f}s over {len(GRAPHS)} instances\n")
hdr = f"{'budget':>7}" + "".join(f"{'h=' + str(h):>8}" for h in HOPS)
print("COVERAGE  (target inside the budgeted candidate set)")
print(hdr)
print("-" * len(hdr))
for b in BUDGETS:
    print(f"{b:>7}" + "".join(f"{np.mean(cov[h][b]):>8.3f}" for h in HOPS))
"""))

A(md(r"""
### Reading the table

* At `h=0` the curve **saturates at 0.412** — the ceiling of lexical
  retrieval. Growing the budget past 20 buys nothing, because there is
  nothing more to find.
* One hop of expansion lifts that ceiling to **0.531** and, crucially,
  reaches it at a *budget of 20 rather than 800*.
* Two hops reach **0.548** at budget 30.
* The `h=4` cell at budget 800 (0.734) is genuine recall, but it costs an
  average pool of thousands of nodes — useless to an agent with a
  tool-call budget.
"""))
A(code(r"""
print(f"{'h':>4}{'cov@50':>9}{'cov@200':>9}{'med pool':>10}{'med radius':>12}")
for h in HOPS:
    print(f"{h:>4}{np.mean(cov[h][50]):>9.3f}{np.mean(cov[h][200]):>9.3f}"
          f"{np.median(pool_sizes[h]):>10.0f}{np.median(radii[h]):>12.1f}")

base = np.array(cov[0][50])
one = np.array(cov[1][50])
p, lo, hi = wilson(int(one.sum()), len(one))
print(f"\ncoverage@50 at h=0 : {base.mean():.3f}")
print(f"coverage@50 at h=1 : {one.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
gained = int(((one > base) & (base == 0)).sum())
lost = int(((base > one) & (one == 0)).sum())
print(f"paired: {gained} tasks gained, {lost} lost, exact McNemar p = {mcnemar(gained, lost):.2e}")
"""))

A(md(r"""
**Expansion never loses a task.** The one-hop gain comes from tasks
moving from "not in the lexical top 50" to "in the expanded top 50", and
`lost = 0` in the reverse direction — exactly the monotonicity one would
hope for, and a direct consequence of seeds being ordered first.
"""))

A(md(r"""
### 7.3 Choosing a hop cap that does not run away

Unbounded expansion is not free. We cap the pool and re-measure.
"""))
A(code(r"""
CAPS = (50, 100, 200, 400, 800, 10 ** 9)
cap_tab = {c: {b: [] for b in (20, 50, 100, 200)} for c in CAPS}
cap_pool = {c: [] for c in CAPS}
cap_rad = {c: [] for c in CAPS}

for iid, g in GRAPHS.items():
    for c in CAPS:
        ranked, mh, psz = expand(g, ISSUES[iid], seed_k=10, hops=12, cap=c)
        cap_pool[c].append(psz)
        cap_rad[c].append(mh)
        for b in (20, 50, 100, 200):
            cap_tab[c][b].append(1.0 if set(ranked[:b]) & GOLDS[iid] else 0.0)

print(f"{'cap':>8}{'cov@50':>9}{'cov@200':>9}{'mean pool':>11}{'mean radius':>13}")
for c in CAPS:
    lbl = "inf" if c > 10 ** 8 else str(c)
    print(f"{lbl:>8}{np.mean(cap_tab[c][50]):>9.3f}{np.mean(cap_tab[c][200]):>9.3f}"
          f"{np.mean(cap_pool[c]):>11.0f}{np.mean(cap_rad[c]):>13.2f}")
"""))

A(md(r"""
A cap of **200 nodes** is the operating point: it realizes a mean radius
of ~2.8 hops and coverage 0.603 at budget 50, whereas unbounded expansion
explodes to a mean pool above 4 000 nodes and a realized radius above 11
hops. The graph is small-world, so radius grows logarithmically and the
frontier becomes enormous.
"""))
