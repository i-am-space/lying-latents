"""Detached scheduler for Step 2b: encodes each SAE as soon as its download is verified and a GPU
is free; runs the layer-20 16k reproduction check after that width. Rerun-safe (skips done keys).
Usage: setsid nohup python3 concept/scripts/cache_queue.py > logs/queue.log 2>&1 &"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "concept" / "src"))
from cseeds import load_concept_config  # noqa: E402

cfg = load_concept_config()
cc = cfg["concept"]
cdir = ROOT / cc["cache_dir"]
logs = cdir / "logs"
order = sorted(cc["saes"], key=lambda k: cc["saes"][k]["width"])          # every SAE in the config, narrowest first
def resid_ready(layer):
    f = cdir / "resid.done"
    return f.exists() and layer in __import__("json").loads(f.read_text())["layers"] and (cdir / f"resid_L{layer}.npy").exists()
gpus = [int(g) for g in os.environ.get("GPUS", "0,1").split(",")]
running = {}                                   # gpu -> (key, Popen)
done = {k for k in order if (cdir / f"lat_{k}.json").exists()}
failed = set()
attempts = {}
print(time.strftime("%T"), "start; already done:", sorted(done), flush=True)
while len(done | failed) < len(order):
    for g, (k, pr) in list(running.items()):
        if pr.poll() is not None:
            del running[g]
            if pr.returncode == 0 and (cdir / f"lat_{k}.json").exists():
                done.add(k); print(time.strftime("%T"), "DONE", k, flush=True)
                if k == "L20_16k":
                    r = subprocess.run([sys.executable, str(ROOT / "concept/src/check_reproduction.py")],
                                       capture_output=True, text=True)
                    print(time.strftime("%T"), "REPRODUCTION CHECK", r.stdout.strip()[-400:], r.stderr[-300:], flush=True)
            else:
                log = (logs / f"lat_{k}.log").read_text()[-3000:]
                attempts[k] = attempts.get(k, 0) + 1
                if (pr.returncode == 75 or "CUDA driver initialization failed" in log) and attempts[k] < 30:
                    print(time.strftime("%T"), "CUDA init failure, requeue", k, attempts[k], flush=True)
                else:
                    failed.add(k); print(time.strftime("%T"), "FAILED", k, "rc", pr.returncode, flush=True)
    busy = {k for k, _ in running.values()}
    ready = [k for k in order if k not in done | failed | busy and resid_ready(cc["saes"][k]["layer"])
             and (cdir / "sae" / cc["saes"][k]["path"] / ".verified").exists()]
    for g in gpus:
        if g not in running and ready:
            k = ready.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            pr = subprocess.Popen([sys.executable, "-u", str(ROOT / "concept/src/cache_sparse_latents.py"), "--key", k],
                                  env=env, stdout=open(logs / f"lat_{k}.log", "w"), stderr=subprocess.STDOUT)
            running[g] = (k, pr); print(time.strftime("%T"), f"launch {k} on GPU {g}", flush=True)
    time.sleep(15)
print(time.strftime("%T"), "ALL DONE; failed:", sorted(failed), flush=True)
