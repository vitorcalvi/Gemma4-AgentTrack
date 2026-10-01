import nbformat, sys, time
from nbclient import NotebookClient
src, out = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "."}},
                        allow_errors=True)
t=time.time()
client.execute()
nbformat.write(nb, out)
errs=0
for i,c in enumerate(nb.cells):
    if c.cell_type!="code": continue
    for o in c.get("outputs",[]):
        if o.get("output_type")=="error":
            errs+=1
            print(f"--- ERROR cell {i}: {o['ename']}: {o['evalue']}")
            print("   ", "\n    ".join(o.get("traceback",[])[-4:])[:600])
print(f"\nexecuted in {time.time()-t:.0f}s, errors={errs}")
