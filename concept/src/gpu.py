"""Robust CUDA start-up. This host's driver intermittently fails cuInit (CUDA_ERROR_NOT_INITIALIZED,
about half the time during bad spells); a process that initialises successfully then runs normally.
Initialise the driver directly, with retries, before torch touches CUDA. Exit code 75 (EX_TEMPFAIL)
tells the schedulers to relaunch."""
import ctypes
import sys
import time

CUDA_TEMPFAIL = 75


def init_cuda(device: str = "cuda", tries: int = 60, wait: float = 5.0):
    import torch
    if not str(device).startswith("cuda"):
        return torch.device(device)
    lib = ctypes.CDLL("libcuda.so.1")
    for _ in range(tries):
        if lib.cuInit(0) == 0:
            try:
                torch.zeros(1, device=device)
                return torch.device(device)
            except RuntimeError:
                pass
        time.sleep(wait)
    print("CUDA unavailable after retries; exiting for relaunch", flush=True)
    sys.exit(CUDA_TEMPFAIL)
