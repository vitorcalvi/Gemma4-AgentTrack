"""
Where does graph expansion actually start to pay off?

cross_repo.py found that at k=20 expansion does NOT help (-0.014). This
script locates the crossover budget B* where depth-aware expansion starts
to beat pure lexical ordering, and reports it per repository so we can say
exactly when the method is worth its cost.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cross_repo import build, policies

BUDGETS = (5, 10, 20, 30, 50, 100, 200, 400, 800)
KS = (1, 3, 5, 10, 20, 50)


def main(inst_path, graph_dir, out_path):
    data = build(inst_path, graph_dir)
    repos = sorted({d["repo"] for d in data})
    print(f"n={len(data)} over {len(repos)} repositories\n")

    hit = lambda names, k: 1.0 if set(names[:k]) & d_gset else 0.0

    curves = {}
    for d in data:
        pol = policies(d)
        for pn, names in pol.items():
            for b in BUDGETS:
                curves.setdefault((pn, b), []).append(
                    1.0 if set(names[:b]) & d["gset"] else 0.0)
    for d in data:
        d_gset = d["gset"]

    print(f"{'B':>5}{'lexical':>10}{'budget_mix':>13}{'gain':>9}   (coverage@B, n=%d)" % len(data))
    rows = {}
    for b in BUDGETS:
        lx = float(np.mean(curves[("lexical_only", b)]))
        mx = float(np.mean(curves[("budget_mix", b)]))
        rows[b] = {"lexical": lx, "budget_mix": mx, "gain": mx - lx}
        print(f"{b:>5}{lx:>10.3f}{mx:>13.3f}{mx-lx:>+9.3f}")

    # crossover
    first = next((b for b in BUDGETS if rows[b]["gain"] > 0), None)
    print(f"\nfirst budget with positive gain: {first}")

    # sign test at the chosen operating point
    from math import comb
    for b in (50, 100):
        g = np.array(curves[("budget_mix", b)]) - np.array(curves[("lexical_only", b)])
        pos, neg = int((g > 0).sum()), int((g < 0).sum())
        n = pos + neg
        p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)
        print(f"B={b:>4}: improved={pos:3d} worsened={neg:3d} unchanged={len(g)-n:3d}  exact p={p:.3e}")

    # per-repo at B=50
    print(f"\nper-repository at B=50 (lexical -> budget_mix):")
    per = {}
    for d in data:
        pol = policies(d)
        a = per.setdefault(d["repo"], [0, 0, 0])
        a[0] += 1
        a[1] += 1.0 if set(pol["lexical_only"][:50]) & d["gset"] else 0.0
        a[2] += 1.0 if set(pol["budget_mix"][:50]) & d["gset"] else 0.0
    for repo, (n, lx, mx) in sorted(per.items(), key=lambda x: -x[1][0]):
        print(f"  {repo:26s}{n:>4}  {lx/n:.3f} -> {mx/n:.3f}  {(mx-lx)/n:+.3f}")
    json.dump({"n": len(data), "curves": {f"{k[0]}@{k[1]}": float(np.mean(v))
                                          for k, v in curves.items()},
               "crossover": first, "per_repo_B50": {k: {"n": v[0], "lex": v[1]/v[0],
                                                        "mix": v[2]/v[0]} for k, v in per.items()}},
              open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
