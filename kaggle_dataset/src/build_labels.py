"""
Build the GraphLoc-300 label resource.

For every SWE-bench Lite instance we emit function-level localization
labels derived from the reference patch, together with the structural
features of the target. This is the reusable artifact: it lets anyone
train or audit a localizer without re-deriving patch semantics.
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

# Map the function-level gold ids used by the graph back to a stable
# module::qualified name and record whether the symbol is private/public.
def classify(nid: str) -> dict:
    qn = nid.split("::")[-1] if "::" in nid else nid
    return {
        "node": nid,
        "file": nid.split("::")[0] if "::" in nid else nid,
        "symbol": qn,
        "private": qn.startswith("_") and not (qn.startswith("__") and qn.endswith("__")),
        "dunder": qn.startswith("__") and qn.endswith("__"),
        "is_method": qn.count(".") >= 1,
    }

def main(inst_path, graph_dir, out_path, seed_k=10, cap=200):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    recs = []
    missing = 0
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst: continue
        gold = [x for x in inst[iid]["gold"]["entities"]]
        rec = {"instance_id": iid, "repo": inst[iid]["repo"],
               "base_commit": inst[iid]["base_commit"],
               "n_gold_files": len(inst[iid]["gold"]["files"]),
               "gold_files": inst[iid]["gold"]["files"],
               "module_level_lines": len(inst[iid]["gold"]["module_level"]),
               "targets": [classify(g) for g in gold]}
        try:
            g = load_graph(path)
            rec["n_graph_nodes"] = g.n_nodes
            rec["n_graph_files"] = len(set(g.files))
            gset = {x for x in gold if x in g.index}
            rec["targets_in_graph"] = len(gset)
            rec["coverage"] = len(gset) / max(len(gold), 1)
            # lexical rank of the best target
            lex = g.bm25.score(ident_tokens(inst[iid]["problem_statement"]))
            order = np.argsort(-lex, kind="stable")
            pos = {g.node_ids[int(i)]: r for r, i in enumerate(order)}
            rec["best_lexical_rank"] = min((pos.get(x, 10 ** 9) for x in gset), default=10 ** 9)
            rec["n_candidates"] = g.n_nodes
        except Exception as e:
            rec["error"] = str(e)
        if not rec.get("targets_in_graph"):
            missing += 1
        recs.append(rec)
    print(f"instances labelled: {len(recs)}  (no in-graph target: {missing})")
    with open(out_path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    # dataset-level summary
    cov = [r.get("coverage", 0.0) for r in recs]
    rk = [r.get("best_lexical_rank", 10 ** 9) for r in recs]
    npv = sum(1 for r in recs for t in r["targets"] if t["private"])
    ntg = sum(len(r["targets"]) for r in recs)
    summ = {
        "instances": len(recs),
        "graph_coverage": float(np.mean(cov)),
        "pct_targets_private": npv / max(ntg, 1),
        "median_best_lexical_rank": float(np.median(rk)),
        "pct_lexical_rank_gt_1000": float(np.mean([r > 1000 for r in rk])),
        "pct_lexical_rank_gt_100": float(np.mean([r > 100 for r in rk])),
    }
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(out_path.replace(".jsonl", "_summary.json"), "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
