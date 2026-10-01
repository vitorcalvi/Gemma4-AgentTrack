"""
The adherence gap.

Localization research measures P(target in candidate set). Agent
research measures P(task resolved). This script joins the two on the
public end-to-end run log and quantifies the gap between a target being
*retrievable* and the agent actually *acting* on it.
"""
from __future__ import annotations
import csv, json, sys
import numpy as np
from math import comb

def num(v):
    v = (v or "").strip()
    return float(v) if v not in ("", "nan", "None") else np.nan

def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)

def main(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path)))
    out = {"n_rows": len(rows)}
    print(f"loaded {len(rows)} agent runs\n")

    base = [r for r in rows if r["arm"] == "base"]
    skill = [r for r in rows if r["arm"] == "skill"]

    # --- 1. Localization is necessary but not sufficient -------------------
    loc = []
    for arm, rs in (("base", base), ("skill", skill)):
        for r in rs:
            edited = num(r["edits_gold_function"]) == 1
            res = r["resolved"] == "1"
            loc.append((edited, res, arm))
    e1 = [x for x in loc if x[0]]; e0 = [x for x in loc if not x[0]]
    p1, l1, h1 = wilson(sum(x[1] for x in e1), len(e1))
    p0, l0, h0 = wilson(sum(x[1] for x in e0), len(e0))
    print("P(resolved | agent edited the gold function)")
    print(f"   edited gold fn : {sum(x[1] for x in e1):4d}/{len(e1):4d} = {p1:.3f}  95% CI [{l1:.3f},{h1:.3f}]")
    print(f"   did not        : {sum(x[1] for x in e0):4d}/{len(e0):4d} = {p0:.3f}  95% CI [{l0:.3f},{h0:.3f}]")
    b = sum(1 for x in e0 if x[1]) ; c = sum(1 for x in e1 if not x[1])
    ptest = binom_two_sided(b, c)
    print(f"   risk ratio     : {p1/max(p0,1e-9):.1f}x   McNemar p={ptest:.2e}\n")
    out["resolved_given_edited"] = {"p": p1, "ci": [l1, h1], "n": len(e1)}
    out["resolved_given_not_edited"] = {"p": p0, "ci": [l0, h0], "n": len(e0)}
    out["mcnemar_localization"] = ptest

    # --- 2. The adherence gap ---------------------------------------------
    has = [r for r in skill if num(r["skill_top15_has_gold"]) == 1]
    gap = [r for r in has if num(r["edits_gold_function"]) == 0]
    adhered = [r for r in has if num(r["edits_gold_function"]) == 1]
    pg, lg, hg = wilson(len(gap), len(has))
    pa, la, ha = wilson(len(adhered), len(has))
    print("Adherence: gold WAS in the retriever's top-15")
    print(f"   runs                : {len(has)}")
    print(f"   agent edited it     : {len(adhered)} = {pa:.3f}  95% CI [{la:.3f},{ha:.3f}]")
    print(f"   agent IGNORED it    : {len(gap)} = {pg:.3f}  95% CI [{lg:.3f},{hg:.3f}]  <-- adherence gap")
    locs = {}
    fails = {}
    for r in gap:
        locs[r["location"]] = locs.get(r["location"], 0) + 1
        fails[r["failure"]] = fails.get(r["failure"], 0) + 1
    print(f"   where they went     : {locs}")
    print(f"   why they failed     : {fails}")
    print(f"   mean tool calls     : {np.mean([num(r['tool_calls']) for r in gap]):.1f}")
    print(f"   mean minutes        : {np.mean([num(r['minutes']) for r in gap]):.1f}")
    print(f"   ran their tests     : {np.mean([num(r['ran_tests']) for r in gap]):.1%}\n")
    out["adherence"] = {"n_retrievable": len(has), "p_adhered": pa,
                        "p_gap": pg, "gap_locations": locs, "gap_failures": fails}

    # --- 3. Top-5 vs top-15 headroom --------------------------------------
    t5 = [r for r in skill if num(r["skill_top5_has_gold"]) == 1]
    t15 = [r for r in skill if num(r["skill_top15_has_gold"]) == 1]
    t15m = [r for r in skill if np.isnan(num(r["skill_top15_has_gold"]))]
    print("Retriever headroom")
    print(f"   gold in top-5  : {len(t5)}/{len(skill)} = {len(t5)/len(skill):.3f}")
    print(f"   gold in top-15 : {len(t15)}/{len(skill)} = {len(t15)/len(skill):.3f}")
    print(f"   gold in top-15 but not top-5 (recoverable by depth): {len(t15)-len(t5)}")
    out["top5"] = len(t5) / len(skill)
    out["top15"] = len(t15) / len(skill)

    # --- 4. Do agents use their own tests? ---------------------------------
    for arm, rs in (("base", base), ("skill", skill)):
        ran = [r for r in rs if num(r["ran_tests"]) == 1]
        nr = [r for r in rs if num(r["ran_tests"]) == 0]
        pr = np.mean([r["resolved"] == "1" for r in ran]) if ran else float("nan")
        pn = np.mean([r["resolved"] == "1" for r in nr]) if nr else float("nan")
        print(f"\n{arm}: ran tests {len(ran)} ({len(ran)/len(rs):.1%}) -> resolved {pr:.1%}; "
              f"skipped {len(nr)} -> {pn:.1%}")
        out[f"{arm}_ran_tests"] = len(ran) / len(rs)

    json.dump(out, open(out_path, "w"), indent=1)
    return out

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
