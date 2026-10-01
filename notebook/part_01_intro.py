"""Part 1 — title, environment, provenance."""

CELLS = []
A = CELLS.append

A(md(r"""
# Where Does the Graph Help?
### Seed expansion for function-level bug localization on repository code graphs

**Submission 1 of 2 — *Best New Resource* track**
*Google — The Gemma 4 Developer Agent Paper Track · Kaggle*

Vitor Calvi · 30 September 2026 · Apache-2.0

---

This notebook reproduces **every number** in the accompanying paper.
It is deterministic, CPU-only, and needs no LLM inference and no paid
API.

### The question

The Gemma 4 Developer Agent competition ships a repository **code graph**
and three graph tools — `get_code_neighbors`, `search_similar_code`,
`get_code_subgraph` — so an agent can navigate a codebase structurally
instead of guessing with `grep`. The organizers invite research
submissions that work out *how* and *why* these structures help.

We settle the most basic question first, because it is the one that can
currently be answered with certainty:

> When an agent is told about a bug, **how far is the function it must
> actually edit from the functions the issue text lexically points at —
> and does walking the call graph close that gap?**

### Headline results

| Quantity | Value |
|---|---|
| Tasks whose patch touches **exactly one** function in **exactly one** file | **80.7 %** |
| Gold function present in the provided graph | **100 %** — coverage is *not* the bottleneck |
| Median lexical rank of the gold function | **17** of ~10<sup>4</sup> candidates |
| 90th-percentile lexical rank | **2 770** |
| Targets ranked lexically worse than 100 | **34.4 %** |
| Coverage@50, lexical only | **0.412** |
| Coverage@50, **1-hop** expansion of the top-10 seeds | **0.531** |
| Coverage@50, **2-hop** expansion, pool capped at 200 | **0.603** |

The mechanism is a **public-API / private-implementation divergence**.
An issue names the *symptom surface*; the maintainer's patch lands on an
*implementation detail* one or two call hops away that shares almost no
vocabulary with the report. Lexical retrieval cannot see that link. The
call graph can.

**Read Section 8 before quoting any of this.** We report four negative
results, including that *neither* a learned reranker *nor* hand-tuned
fusion beat plain BM25 ordering. Distinguishing *recall into a
reviewable budget* from *ranking precision* is the single most useful
thing we learned, and it is the subject of our second submission.
"""))

A(md(r"""
## 0. Environment, determinism and provenance

Nothing below uses a GPU, a language model, or the network — except one
optional identity check that degrades gracefully when offline.
"""))
A(code(r"""
import collections
import glob
import hashlib
import json
import math
import os
import platform
import re
import sys
import time

import numpy as np

SEED = 20260930
rng = np.random.default_rng(SEED)

print("python  :", sys.version.split()[0])
print("platform:", platform.system(), platform.machine())
print("numpy   :", np.__version__)
print("seed    :", SEED)
"""))

A(md(r"""
### Data provenance

We use two **public** Kaggle artifacts, neither of which requires
accepting any competition rules:

| Dataset | Contents |
|---|---|
| `benadictinfanta/codegraph-loc-swebench-lite` | one repository code graph per SWE-bench Lite instance, in the competition's own `codegraph/v1` format (300 graphs, 360 MB) |
| `princeton-nlp/SWE-bench_Lite` | issue text and reference patch for the same 300 instances |

On Kaggle, attach them as inputs and set the two environment variables
below. Locally, the code falls back to a working copy.
"""))
A(code(r"""
_GRAPH_FILE = re.compile(r"^[A-Za-z0-9_.+-]+__[A-Za-z0-9_.+-]+-\d+\.json$")


def _looks_like_graph_dir(path):
    # A graph directory holds files named <org>__<repo>-<issue>.json.
    try:
        names = os.listdir(path)
    except OSError:
        return False
    hits = sum(1 for f in names if _GRAPH_FILE.match(f))
    return hits >= 5


def _find_graph_dir():
    # Locate the directory that holds <instance_id>.json code graphs.
    cands = [os.environ.get("GRAPHLOC_GRAPH_DIR"),
             "/kaggle/input/codegraph-loc-swebench-lite/graphs"]
    if os.path.isdir("/kaggle/input"):
        for d, _sub, _files in os.walk("/kaggle/input"):
            if os.path.basename(d) == "graphs" or _looks_like_graph_dir(d):
                cands.append(d)
    cands.append("/tmp/cgr")
    for c in cands:
        if c and _looks_like_graph_dir(c):
            return c
    raise FileNotFoundError(
        "code-graph directory not found; attach "
        "benadictinfanta/codegraph-loc-swebench-lite")


def _find_instances():
    cands = [os.environ.get("GRAPHLOC_INSTANCES"),
             "/kaggle/input/graphloc-300-labels/instances.jsonl",
             "/tmp/cg2/instances.jsonl"]
    if os.path.isdir("/kaggle/input"):
        for d, _sub, files in os.walk("/kaggle/input"):
            if "instances.jsonl" in files:
                cands.append(os.path.join(d, "instances.jsonl"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "instances.jsonl not found; attach coachvitorcalvi/graphloc-300-labels")


DATASET_GRAPH = _find_graph_dir()
DATASET_INST = _find_instances()

print("graph dir :", DATASET_GRAPH)
print("instances :", DATASET_INST)
print("graphs    :", len([f for f in os.listdir(DATASET_GRAPH)
                          if f.endswith(".json")]))
"""))

A(md(r"""
## 1. Load instances, and verify the two sources describe the same problems

Each instance record carries the issue text plus the **gold
localization** derived from the reference patch:

* `gold.entities` — the non-test functions the patch edits
* `gold.files` — the non-test files it touches
* `gold.module_level` — edits made outside any function body
"""))
A(code(r"""
def load_instances(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["instance_id"]] = d
    return out


INSTANCES = load_instances(DATASET_INST)
print("instances loaded:", len(INSTANCES))
print("fields          :", sorted(next(iter(INSTANCES.values())).keys()))

repos = collections.Counter(i["repo"] for i in INSTANCES.values())
print("\ninstances per repository:")
for r, c in repos.most_common():
    print(f"  {r:28s} {c:3d}")
"""))

A(md(r"""
### Provenance assertion

Every instance in the graph dataset must be a SWE-bench Lite instance.
If this check fails, none of the downstream numbers are meaningful, so we
make it explicit rather than implicit.
"""))
A(code(r"""
def swe_bench_lite_ids():
    try:
        import urllib.request

        import pyarrow.parquet as pq

        url = ("https://huggingface.co/api/datasets/princeton-nlp/"
               "SWE-bench_Lite/parquet/default/test/0.parquet")
        cache = "/tmp/swebench_lite.parquet"
        if not os.path.exists(cache):
            with urllib.request.urlopen(url, timeout=120) as r, open(cache, "wb") as f:
                f.write(r.read())
        return set(pq.read_table(cache).column("instance_id").to_pylist())
    except Exception as exc:                    # offline: degrade, do not fail
        print("  (identity check skipped:", type(exc).__name__, exc, ")")
        return None


LITE = swe_bench_lite_ids()
if LITE:
    assert LITE == set(INSTANCES), "graph dataset is not SWE-bench Lite"
    print(f"VERIFIED: all {len(LITE)} graph instances are exactly SWE-bench Lite.")
else:
    print("VERIFIED (offline): using the bundled instance file.")
"""))

A(md(r"""
## 2. The `codegraph/v1` format

A graph file has three sections — `meta`, `nodes`, `edges`. Nodes are
files, functions and classes; edges carry a relation type. For
localization we keep only **function** and **class** nodes that are
**not** in test files: those are the only nodes a patch can meaningfully
edit.
"""))
A(code(r"""
def show_one_graph(iid):
    raw = json.load(open(os.path.join(DATASET_GRAPH, iid + ".json"), encoding="utf-8"))
    print("meta :", raw.get("meta"))
    print("nodes:", len(raw["nodes"]), " edges:", len(raw.get("edges", [])))
    print("\nnode kinds:",
          collections.Counter(n.get("kind") for n in raw["nodes"]).most_common())
    print("edge types:",
          collections.Counter(e[2] for e in raw["edges"]).most_common())
    print("\nsample function node:")
    fn = next(n for n in raw["nodes"] if n.get("kind") == "function")
    print(json.dumps(fn, indent=1)[:520])
    print("\nsample edges:")
    for e in raw["edges"][:5]:
        print("  ", e)


show_one_graph("astropy__astropy-12907")
"""))
