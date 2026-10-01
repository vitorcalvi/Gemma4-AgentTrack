"""
One-shot reproduction driver.

Runs every experiment in the paper and writes a single
results/summary.json plus per-experiment tables. Deterministic: no
randomness except seeded bootstrap, no network, no LLM.
"""
from __future__ import annotations
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def run(script, *args):
    print(f"\n{'='*70}\n$ python {os.path.basename(script)} {' '.join(args)}\n{'='*70}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args])
    print(f"[{os.path.basename(script)}] exit={r.returncode} in {time.time()-t:.0f}s", flush=True)
    return r.returncode

def main(instances, graphs, outdir):
    os.makedirs(outdir, exist_ok=True)
    J = lambda n: os.path.join(outdir, n)
    run("build_labels.py", instances, graphs, J("graphloc300.jsonl"))
    run("coverage.py",     instances, graphs, J("coverage.json"))
    run("hop_scale.py",    instances, graphs, J("hop_scale.json"), "10")
    run("hop_profile.py",  instances, graphs, J("hop_profile.json"), "10")
    run("eval_suite.py",   instances, graphs, J("eval_suite.json"))
    run("rank_search.py",  instances, graphs, J("rank_search.json"))
    run("rerank.py",       instances, graphs, J("rerank.json"), J("features.npz"))
    run("shortlist.py",    instances, graphs, J("shortlist.json"))
    print("\nAll experiments finished. Results in", outdir)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
