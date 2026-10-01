"""Bootstrap confidence intervals for the seed-expansion effect."""
from __future__ import annotations
import json, sys
import numpy as np

def boot(direct, exp1, exp2, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    d, e1, e2 = map(np.asarray, (direct, exp1, exp2))
    m = len(d)
    out = {"direct": [], "exp1": [], "exp2": [], "gain1": [], "gain2": []}
    for _ in range(n_boot):
        ix = rng.integers(0, m, m)
        dd, ee1, ee2 = d[ix].mean(), e1[ix].mean(), e2[ix].mean()
        out["direct"].append(dd); out["exp1"].append(ee1); out["exp2"].append(ee2)
        out["gain1"].append(ee1 - dd); out["gain2"].append(ee2 - dd)
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items()}

if __name__ == "__main__":
    rows = json.load(open(sys.argv[1]))
    print(f"n = {len(rows)} instances (95% bootstrap CI, 20k resamples)\n")
    hdr = f"{'k':>4} {'direct':>14} {'+1 hop':>14} {'+2 hops':>14} {'gain@2':>16}"
    print(hdr); print("-" * len(hdr))
    for k in (1, 5, 20, 50):
        d = [r[f"direct@{k}"] for r in rows]
        e1 = [r[f"exp1@{k}"] for r in rows]
        e2 = [r[f"exp2@{k}"] for r in rows]
        s = boot(d, e1, e2, n_boot=4000)
        def f(k2):
            m, lo, hi = s[k2]; return f"{m:.3f} [{lo:.3f},{hi:.3f}]"
        m, lo, hi = s["gain2"]
        print(f"{k:>4} {f('direct'):>14} {f('exp1'):>14} {f('exp2'):>14} {f'{100*m:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]':>16}")
    # McNemar-style paired test at k=5 and k=20
    from math import comb
    for k in (5, 20, 50):
        b = sum(1 for r in rows if r[f"exp2@{k}"] > r[f"direct@{k}"])
        c = sum(1 for r in rows if r[f"direct@{k}"] > r[f"exp2@{k}"])
        n = b + c
        p = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2 if n else 1.0
        print(f"k={k:3d}  gained={b:4d}  lost={c:3d}  exact McNemar p={min(p,1.0):.2e}")
