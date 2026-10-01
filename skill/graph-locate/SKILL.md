---
name: graph-locate
description: Rank repository functions by likelihood of containing a bug fix, using BM25 seeds over the issue text expanded across the code call graph. Use this FIRST, before grep or opening files, to decide which file to edit.
---

# graph-locate

## When to use this

Call this **before** you `run_command("grep ...")` or open any file. It
answers "which function is the patch probably in?" in one call.

## How to call it

```
run_skill_script(
  skill="graph-locate",
  script="localize.py",
  args={"issue": "<the full issue text>", "seed_k": 10, "hops": 2, "cap": 200, "top_n": 20}
)
```

`args["issue"]` must be the **raw issue text**, not a summary. The
retriever is lexical; a paraphrase destroys the signal.

## What you get back

One line per candidate:

```
<rank> <score> <hops> <file>:<line> <symbol>
```

* `hops = 0` — the issue text lexically matches this function
* `hops = 1` — it is called by, or calls, a lexical match
* `hops = 2` — two call hops out

## How you MUST use the output

1. **Open the `hops = 0` file first.** That is where the reporter
   observed the symptom.
2. **If no `hops = 0` candidate reproduces the symptom, open the
   top-ranked `hops = 1` file.** Maintainers frequently fix the private
   helper that the public API delegates to. This is the case the plain
   graph tools miss.
3. **Do not open more than three candidate files before you have edited
   something.** Budget your tool calls.
4. **Run the reproduction before you edit**, so you know whether you are
   in the right file at all.

## Why hops matter

On SWE-bench Lite, 80.7% of patches touch exactly one function. That
function is outside the lexical top-100 in 34% of tasks, because issues
name the public API while patches edit the private helper underneath.
Walking one or two call hops is what closes that gap: it lifts the
fraction of tasks whose target is inside a 50-function review budget
from 0.41 to 0.60.

Ranking within the expanded pool is a genuinely open problem — a
learned reranker did not beat plain BM25 ordering. So trust the *order
within* a hop layer, and use `hops` to decide which layer to work in.
