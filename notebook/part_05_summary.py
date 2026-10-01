"""Part 5 — summary, reproduction, citation."""

CELLS = []
A = CELLS.append

A(md(r"""
## 10. What we learned

**Supported by these measurements:**

1. Bug-fix patches here are overwhelmingly **single-target** — 80.7 %
   touch exactly one function in one file.
2. The provided graphs are **complete**; the target is present essentially
   always. Coverage is not the bottleneck, ranking is.
3. Lexical retrieval has a **long, catastrophic tail**: median rank 17,
   but 34 % of tasks have their target outside the top 100.
4. The mechanism is **public-API / private-implementation divergence**:
   the issue names a symptom surface, the patch edits an implementation
   detail one or two call hops away.
5. **Seed expansion converts that into recall.** Coverage@50 rises from
   0.412 to 0.531 at one hop and 0.603 at two hops with a 200-node cap,
   with **zero tasks lost** at any setting.
6. Expansion must be **capped**; unbounded expansion is useless in
   practice (mean pool > 4 000 nodes, mean radius > 11 hops).

**Not supported — reported as negative results:**

7. A learned reranker over graph and lexical features **does not beat
   plain BM25 ordering** (McNemar p ≈ 0.4).
8. **Seed consensus is an anti-signal** (coverage@20 falls to 0.17).
9. **Personalized PageRank fusion** adds ≤ ~1 point over lexical alone.
10. **Pre-flight task triage carries no usable signal** (abs(r) ≤ 0.20).

### The one-line takeaway

> Graphs reliably tell you **where to look**. Nothing we tried tells you
> **what to look at first**. That second problem is where the remaining
> headroom is — and it is not a retrieval problem.

Submission 2 of 2 takes this straight to the agent. On the public
end-to-end run logs, we measure that in **46.6 %** of runs where the
retriever's top-15 *did* contain the correct function, the agent still
did not edit it. The binding constraint is **adherence**, not recall.
"""))

A(md(r"""
## 11. Limitations

We state these plainly, because they bound the claims above.

* **Sample.** All localization results use the subset of SWE-bench Lite
  instances for which a public code graph is available (300 published;
  the run above reports whatever is present on disk). Confidence
  intervals are given, and every paired comparison is exact.
* **One graph builder.** All graphs come from a single extraction
  pipeline. Conclusions about `calls` / `imports` / `inherits` do not
  transfer automatically to data-flow edges, to other languages, or to
  dynamically-typed call sites that static analysis cannot resolve.
* **Coverage is computed, not executed.** We verify that the gold symbol
  exists as a node. We do not run the reference patch, so we do not claim
  that any ranking we produce yields a *correct patch* — only that it
  puts the right function in front of a reviewer.
* **No end-to-end agent run.** This submission deliberately makes no
  resolve-rate claim. See submission 2 for the agent-side measurement.
* **BM25 is a strong baseline.** We did not compare against a modern
  dense code retriever, because no public checkpoint for this exact
  `codegraph/v1` node set was available to us without competition data
  access. A dense retriever would likely raise the *seed* quality and
  therefore the expansion ceiling; it would not change the structural
  argument.
"""))

A(md(r"""
---
## Reproduction

```bash
python src/reproduce.py <instances.jsonl> <graph_dir> results/
```

or open this notebook top-to-bottom. Runtime on a Kaggle CPU notebook:
**under two minutes**, no GPU, and no network access beyond one optional
identity check that degrades gracefully offline.

### Files

| Path | Purpose |
|---|---|
| `src/graphloc_core.py` | tokenizer, BM25, graph loader |
| `src/graphloc.py` | the method |
| `src/coverage.py` | coverage-vs-budget main result |
| `src/hop_scale.py` | cap selection |
| `src/hop_profile.py` | hop-distance distribution |
| `src/rerank.py` | negative result 8.1 |
| `src/shortlist.py` | negative result 8.4 |
| `skills/graph-locate/` | ADK skill for agent submissions |

### Citation

```bibtex
@misc{calvi2026graphloc,
  title        = {Where Does the Graph Help? Seed Expansion for
                  Function-Level Bug Localization on Repository Code Graphs},
  author       = {Vitor Calvi},
  year         = {2026},
  note         = {Kaggle: Google -- The Gemma 4 Developer Agent Paper Track},
  howpublished = {\\url{https://www.kaggle.com/competitions/gemma-4-developer-agent-paper}}
}
```

### License

Apache-2.0, matching the competition. No proprietary data, model
checkpoints, or evaluation infrastructure are used. Underlying
benchmarks: SWE-bench (Jimenez et al., ICLR 2024) and its Lite split.
"""))
