"""
Task shortlisting: can cheap structural signals predict which tasks a
budget-limited coding agent should attempt first?

The end-to-end audit shows agent runs die on 'no_patch | time budget'.
This module asks whether a pre-flight triage score, computable from the
issue text + graph alone (no LLM, no test execution), can rank tasks by
the likelihood that a patch is needed at all.
"""
from __future__ import annotations
import collections, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens, qualified_name

TRACE = re.compile(r"(Traceback \(most recent call last\):|File \".*?\", line \d+|\bAssertionError\b|\bTypeError\b|\bValueError\b|\bKeyError\b|\bAttributeError\b|\bImportError\b)")
CODE_FENCE = re.compile(r"```")
TESTWORD = re.compile(r"\b(test|pytest|assert|fixture|mock|monkeypatch|parametrize)\b", re.I)
MOCKWORD = re.compile(r"\b(mock|MagicMock|Fake|mocker|monkeypatch)\b", re.I)
NUMPUNCT = re.compile(r"[()\[\]{}<>:;=]")

def issue_features(ps: str) -> dict:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", ps)
    return dict(
        n_words=len(words),
        has_trace=1.0 if TRACE.search(ps) else 0.0,
        has_fence=1.0 if CODE_FENCE.search(ps) else 0.0,
        n_fence=len(CODE_FENCE.findall(ps)) / 2.0,
        test_words=len(TESTWORD.findall(ps)),
        mock_words=len(MOCKWORD.findall(ps)),
        punct=len(NUMPUNCT.findall(ps)),
        upper_ratio=sum(1 for c in ps if c.isupper()) / max(len(ps), 1),
    )

def graph_features(g) -> dict:
    n = g.n_nodes
    if n == 0:
        return dict(n_nodes=0, mean_deg=0.0, cross_file=0.0, n_files=0)
    deg = np.array([len(a) for a in g.adj], dtype=np.float32)
    files = np.array([hash(f) % 10 ** 6 for f in g.files])
    uf = len(set(files))
    # cross-file edge fraction: share of structural edges that leave the file
    cross = tot = 0
    for i, outs in enumerate(g.out_edges):
        fi = files[i]
        for j, _t in outs:
            tot += 1
            if files[j] != fi:
                cross += 1
    return dict(n_nodes=n, mean_deg=float(deg.mean()), n_files=uf,
                cross_file=cross / max(tot, 1))

def gold_features(inst) -> dict:
    g = inst["gold"]
    return dict(n_gold=len(g["entities"]), n_gold_files=len(g["files"]),
                module_level=len(g["module_level"]), added_files=len(g["added_files"]))

def main(inst_path, graph_dir, out_path):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    rows = []
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(f)[:-5]
        if iid not in inst: continue
        try: gg = load_graph(f)
        except Exception as e:
            print("ERR", iid, e, file=sys.stderr); continue
        r = {"instance_id": iid, "repo": gg.repo}
        r.update(issue_features(inst[iid]["problem_statement"]))
        r.update(graph_features(gg))
        r.update(gold_features(inst[iid]))
        rows.append(r); print(".", end="", flush=True)
    print(f"\ninstances={len(rows)}")
    json.dump(rows, open(out_path, "w"), indent=1)
    # quick correlations
    keys = [k for k in rows[0] if k not in ("instance_id", "repo")]
    y = np.array([r["n_gold"] > 0 for r in rows], dtype=float)
    print("feature vs 'has pre-existing gold target' (point-biserial):")
    for k in keys:
        x = np.array([r[k] for r in rows], dtype=float)
        if x.std() == 0: continue
        print(f"  {k:12s} r={np.corrcoef(x, y)[0,1]:+.3f}")
    return rows

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
