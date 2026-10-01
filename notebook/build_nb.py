"""Assemble the GraphLoc research notebook from cell sources.

Run:  python build_nb.py [output.ipynb]
"""
import glob
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cells):
    return {"cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python", "version": "3.12.0",
                                  "mimetype": "text/x-python",
                                  "file_extension": ".py",
                                  "codemirror_mode": {"name": "ipython", "version": 3},
                                  "pygments_lexer": "ipython3",
                                  "nbconvert_exporter": "python"},
                "accelerator": "none"},
            "nbformat": 4, "nbformat_minor": 5}


def _lines(s):
    """Notebook source as a list of lines, each keeping its newline."""
    out = s.strip("\n").split("\n")
    return [ln + "\n" for ln in out[:-1]] + [out[-1]]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(s)}


def load_parts():
    cells = []
    for path in sorted(glob.glob(os.path.join(HERE, "part_*.py"))):
        ns = {"md": md, "code": code}
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
        got = ns.get("CELLS") or []
        cells.extend(got)
        print(f"  {os.path.basename(path):22s} {len(got):3d} cells")
    return cells


if __name__ == "__main__":
    cells = load_parts()
    out = sys.argv[1] if len(sys.argv) > 1 else "graphloc_seed_expansion.ipynb"
    nb = build(cells)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    ncode = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {out}: {len(nb['cells'])} cells ({ncode} code)")

