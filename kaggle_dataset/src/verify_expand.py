"""
Independent re-implementation of the seed-expansion result.

Deliberately does NOT reuse graphloc_core's BM25: it uses a plain
python-scored inverted index, so a bug in the shared code cannot
manufacture the headline number.
"""
from __future__ import annotations
import collections, glob, json, math, os, re, sys
import numpy as np

def words(s):
    out = []
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s or ""):
        for piece in ([t] + re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]+|[a-z0-9]+", t)):
            if len(piece) > 2:
                out.append(piece.lower())
    return out

def main(inst_path, graph_dir, out_path):
    inst = {json.loads(l)["instance_id"]: json.loads(l) for l in open(inst_path)}
    rows = []
    for path in sorted(glob.glob(os.path.join(graph_dir, "*.json"))):
        iid = os.path.basename(path)[:-5]
        if iid not in inst:
            continue
        raw = json.load(open(path, encoding="utf-8"))
        nodes = [n for n in raw["nodes"] if n.get("kind") in ("function", "class") and not n.get("test")]
        ids = [n["id"] for n in nodes]
        pos = {v: i for i, v in enumerate(ids)}
        # simple TF-IDF cosine over name+sig
        docs = [words(ids[i].split("::")[-1]) + words(nodes[i].get("sig", "")) for i in range(len(nodes))]
        df = collections.Counter()
        for d in docs:
            df.update(set(d))
        import math as _m
        vecs = []
        for d in docs:
            c = collections.Counter(d)
            v = {w: (1 + _m.log(t)) * _m.log(len(docs) / (1 + df[w])) for w, t in c.items()}
            nrm = _m.sqrt(sum(x * x for x in v.values())) or 1.0
            vecs.append({w: x / nrm for w, x in v.items()})
        q = collections.Counter(words(inst[iid]["problem_statement"]))
        qn = _m.sqrt(sum(t * t for t in q.values())) or 1.0
        scores = np.zeros(len(docs))
        for i, v in enumerate(vecs):
            s = 0.0
            for w, t in q.items():
                if w in v:
                    s += (t / qn) * v[w]
            scores[i] = s
        order = list(np.argsort(-scores, kind="stable"))
        # adjacency (calls/imports/inherits), undirected
        adj = collections.defaultdict(set)
        for e in (raw.get("edges") or []):
            u, v, t = (e[0], e[1], e[2]) if isinstance(e, list) else (e["from"], e["to"], e.get("type"))
            if t not in ("calls", "imports", "inherits"):
                continue
            a, b = pos.get(u), pos.get(v)
            if a is None or b is None or a == b:
                continue
            adj[a].add(b); adj[b].add(a)
        gix = [pos[x] for x in inst[iid]["gold"]["entities"] if x in pos]
        if not gix:
            continue
        r = {"instance_id": iid, "n": len(ids), "repo": raw.get("meta", {}).get("repo", "")}
        gset = set(gix)
        for k in (1, 5, 20, 50):
            seeds = set(order[:k])
            r[f"direct@{k}"] = 1.0 if seeds & gset else 0.0
            seen, frontier = set(seeds), set(seeds)
            hop1 = set()
            for s in frontier: hop1 |= adj[s]
            hop1 -= seen
            seen2 = seen | hop1
            hop2 = set()
            for s in hop1: hop2 |= adj[s]
            hop2 -= seen2
            r[f"exp1@{k}"] = 1.0 if (seeds | hop1) & gset else 0.0
            r[f"exp2@{k}"] = 1.0 if (seen2 | hop2) & gset else 0.0
        rows.append(r); print(".", end="", flush=True)
    print(f"\ninstances={len(rows)}")
    for k in (1, 5, 20, 50):
        d = np.mean([r[f"direct@{k}"] for r in rows])
        e1 = np.mean([r[f"exp1@{k}"] for r in rows])
        e2 = np.mean([r[f"exp2@{k}"] for r in rows])
        print(f"k={k:3d} direct={d:.3f} +1hop={e1:.3f} ({100*(e1-d):+.1f}) +2hop={e2:.3f} ({100*(e2-d):+.1f})")
    json.dump(rows, open(out_path, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
