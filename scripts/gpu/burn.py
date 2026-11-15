"""Load test of one NVIDIA card: bf16 matmuls for N seconds; every 10 s the speed (TFLOPS) and what nvidia-smi says
(temperature, SM clock, power, fan, throttle reasons). Stops early at STOP_C degrees.
Run on the host:  CUDA_VISIBLE_DEVICES=<i> python burn.py <i> [seconds]   (i = the physical index for nvidia-smi)."""
import subprocess
import sys
import time

import torch

STOP_C = 95
idx = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 300
q = "temperature.gpu,clocks.sm,power.draw,fan.speed,clocks_throttle_reasons.active,utilization.gpu"


def smi() -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--id={idx}", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [x.strip() for x in out.split(",")]


n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
print("time_s  TFLOPS  temp_C  sm_MHz  power_W  fan_%  throttle  util_%", flush=True)
print("   0      -     " + "  ".join(smi()), flush=True)
t0 = last = time.time()
ops = 0
while time.time() - t0 < seconds:
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    ops += 20
    now = time.time()
    if now - last >= 10:
        s = smi()
        tflops = ops * 2 * n ** 3 / (now - last) / 1e12
        print(f"{now - t0:5.0f}  {tflops:6.1f}   " + "  ".join(s), flush=True)
        ops, last = 0, now
        if s[0].isdigit() and int(s[0]) >= STOP_C:
            print(f"stopped: {s[0]} C >= {STOP_C} C", flush=True)
            break
del a, b
torch.cuda.empty_cache()
time.sleep(20)
print("20 s after:  " + "  ".join(smi()), flush=True)
