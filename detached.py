import subprocess, sys, os
log = sys.argv[1]
cmd = sys.argv[2:]
with open(log, "wb") as fh:
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=os.getcwd())
print("pid", p.pid)
