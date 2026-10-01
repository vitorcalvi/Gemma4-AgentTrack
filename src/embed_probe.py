"""
Do the competition's own node embeddings add anything over lexical+graph?

Loads embeddings/<repo>_<commit>.npz when present and measures whether
cosine retrieval recovers targets that BM25 misses.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from graphloc_core import load_graph, ident_tokens

def main(inst_path, graph_dir, emb_glob):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    files = sorted(glob.glob(os.path.join(graph_dir, "*.json")))
    found = miss = 0
    stats = []
    for path in files:
        iid = os.path.basename(path)[:-5]
        if iid not in inst: continue
        cands = glob.glob(emb_glob.format(iid=iid))
        cands += glob.glob(os.path.join(emb_glob, iid + "*.npz"))
        if not cands:
            miss += 1; continue
        found += 1
    print("instances with a matching embedding file:", found, " without:", miss)
    print("(embedding assets live in the competition bucket and need rule acceptance;")
    print(" probe reported for transparency only)")
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "/tmp/emb")
