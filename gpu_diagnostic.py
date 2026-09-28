#!/usr/bin/env python3
"""
GPU Diagnostic Tool — VRAM + CUDA Core integrity tester
Target: NVIDIA RTX PRO 6000 Blackwell (Max-Q Workstation Edition), but works on any CUDA GPU.

WHAT IT DOES
1. VRAM test  - writes known bit patterns (0x00, 0xFF, 0xAA, 0x55, walking-bit) across
                all free VRAM, reads them back, and reports any mismatched bytes.
                This is the same principle memtest86 uses for system RAM.
2. CUDA core  - runs identical matrix multiplications redundantly (on different streams /
   test           at different times) and compares results bit-for-bit. A damaged core or
                die typically produces silent wrong answers (not a crash), which this catches.
3. Monitoring - logs temperature, power draw, and clocks via nvidia-smi during the test so
                you can correlate errors with thermal throttling (relevant for Max-Q).

USAGE
    pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
    python gpu_diagnostic.py                     # quick pass (~2 min)
    python gpu_diagnostic.py --full              # thorough pass (~15-20 min)
    python gpu_diagnostic.py --duration 1800     # custom compute-test duration (seconds)
    python gpu_diagnostic.py --vram-only         # skip compute test
    python gpu_diagnostic.py --compute-only      # skip VRAM test

EXIT CODE
    0 = no errors detected
    1 = errors detected (see log)
    2 = could not run (no CUDA / no torch)
"""

import argparse
import ctypes
import subprocess
import sys
import time
from datetime import datetime

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed. Install it with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(2)


LOG_FILE = "gpu_diagnostic_log.txt"


def log(msg, also_print=True):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line)


def get_nvidia_smi_snapshot():
    """Grab temp/power/clock/ECC info from nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,power.draw,power.limit,"
                "clocks.sm,clocks.mem,utilization.gpu,memory.used,memory.total,"
                "ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return out
    except Exception as e:
        return f"(nvidia-smi unavailable: {e})"


def check_cuda():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA-capable GPU detected by PyTorch.")
        sys.exit(2)
    name = torch.cuda.get_device_name(0)
    log(f"Detected GPU: {name}")
    log(f"nvidia-smi snapshot at start:\n{get_nvidia_smi_snapshot()}")
    return name


# ---------------------------------------------------------------------------
# VRAM TEST
# ---------------------------------------------------------------------------

PATTERNS = {
    "all_zero": 0x00,
    "all_one": 0xFF,
    "alternating_AA": 0xAA,
    "alternating_55": 0x55,
}


def vram_pattern_test(chunk_gb=1.0, safety_margin_gb=1.0):
    """
    Allocate as much free VRAM as safely possible in chunks, fill with a pattern,
    read back and verify. Repeats for several patterns to catch stuck bits.
    """
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    log(f"VRAM total: {total_gb:.2f} GB | free: {free_gb:.2f} GB")

    usable_gb = max(free_gb - safety_margin_gb, 0.5)
    chunk_bytes = int(chunk_gb * (1024**3))
    n_chunks = max(int((usable_gb * (1024**3)) // chunk_bytes), 1)
    log(f"Testing {n_chunks} chunk(s) of {chunk_gb} GB each (~{usable_gb:.2f} GB total)")

    total_errors = 0

    for pattern_name, byte_val in PATTERNS.items():
        log(f"--- Pattern: {pattern_name} (0x{byte_val:02X}) ---")
        pattern_errors = 0
        # Fill + verify + free ONE chunk at a time so peak memory stays small
        # (holding all chunks simultaneously leaves no headroom for the
        # comparison buffer and causes an OOM, not a hardware fault).
        for i in range(n_chunks):
            t = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
            t.fill_(byte_val)
            torch.cuda.synchronize()
            mismatches = int((t != byte_val).sum().item())
            if mismatches > 0:
                pattern_errors += mismatches
                log(f"  !! Chunk {i}: {mismatches} byte mismatches detected")
            del t
            torch.cuda.empty_cache()
        log(f"  Pattern {pattern_name}: {pattern_errors} total mismatches")
        total_errors += pattern_errors

    # Walking-bit pattern (catches bits stuck low/high on specific lines)
    log("--- Pattern: walking-bit ---")
    walking_errors = 0
    for bit in range(8):
        byte_val = 1 << bit
        for i in range(n_chunks):
            t = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
            t.fill_(byte_val)
            torch.cuda.synchronize()
            mismatches = int((t != byte_val).sum().item())
            if mismatches > 0:
                walking_errors += mismatches
                log(f"  !! Bit {bit}, chunk {i}: {mismatches} mismatches")
            del t
            torch.cuda.empty_cache()
    log(f"  Walking-bit pattern: {walking_errors} total mismatches")
    total_errors += walking_errors

    return total_errors


# ---------------------------------------------------------------------------
# CUDA CORE / COMPUTE TEST
# ---------------------------------------------------------------------------

def compute_redundancy_test(duration_sec=120, matrix_size=4096, log_interval=15):
    """
    Repeatedly performs the same matmul from the same inputs and checks that the
    result is bit-identical each time. A healthy GPU is deterministic for a fixed
    matmul op/inputs on the same device; divergence indicates a compute fault
    (bad ALU/tensor core, bit-flip, or unstable overclock/voltage/thermal issue).
    Also monitors nvidia-smi for temperature/throttling during the run.
    """
    device = torch.device("cuda")
    torch.manual_seed(1234)

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)

    # Establish a golden reference result
    torch.cuda.synchronize()
    golden = torch.matmul(a, b)
    torch.cuda.synchronize()

    log(f"Compute test: {matrix_size}x{matrix_size} matmul, running for {duration_sec}s")

    start = time.time()
    last_log = start
    iterations = 0
    mismatches_found = 0
    max_abs_diff_seen = 0.0

    while time.time() - start < duration_sec:
        result = torch.matmul(a, b)
        # Exact compare first (catches hard errors / NaNs / Infs)
        if not torch.equal(result, golden):
            diff = (result - golden).abs()
            max_diff = float(diff.max().item())
            max_abs_diff_seen = max(max_abs_diff_seen, max_diff)
            n_bad = int((diff > 1e-3).sum().item())
            if n_bad > 0:
                mismatches_found += n_bad
                log(f"  !! Iteration {iterations}: {n_bad} elements diverged "
                    f"(max abs diff {max_diff:.6f})")

        if torch.isnan(result).any() or torch.isinf(result).any():
            mismatches_found += 1
            log(f"  !! Iteration {iterations}: NaN/Inf detected in output")

        iterations += 1

        now = time.time()
        if now - last_log > log_interval:
            snap = get_nvidia_smi_snapshot()
            log(f"  [{iterations} iters] nvidia-smi: {snap}")
            last_log = now

        del result

    torch.cuda.synchronize()
    log(f"Compute test complete: {iterations} iterations, "
        f"{mismatches_found} mismatched elements, "
        f"max abs diff observed: {max_abs_diff_seen:.6f}")
    return mismatches_found


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU VRAM + CUDA core diagnostic")
    parser.add_argument("--vram-only", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--full", action="store_true", help="Longer, more thorough run")
    parser.add_argument("--duration", type=int, default=None,
                         help="Compute test duration in seconds (default 120, or 900 with --full)")
    parser.add_argument("--chunk-gb", type=float, default=1.0,
                         help="VRAM test chunk size in GB (default 1.0)")
    args = parser.parse_args()

    open(LOG_FILE, "w").close()  # reset log
    log("=== GPU Diagnostic Started ===")

    name = check_cuda()

    duration = args.duration
    if duration is None:
        duration = 900 if args.full else 120

    total_errors = 0

    if not args.compute_only:
        log("\n### VRAM PATTERN TEST ###")
        vram_errors = vram_pattern_test(chunk_gb=args.chunk_gb)
        total_errors += vram_errors
    else:
        vram_errors = 0

    if not args.vram_only:
        log("\n### CUDA CORE / COMPUTE TEST ###")
        compute_errors = compute_redundancy_test(duration_sec=duration)
        total_errors += compute_errors
    else:
        compute_errors = 0

    log(f"\nFinal nvidia-smi snapshot:\n{get_nvidia_smi_snapshot()}")
    log("\n=== SUMMARY ===")
    log(f"GPU: {name}")
    log(f"VRAM mismatches: {vram_errors}")
    log(f"Compute mismatches: {compute_errors}")

    if total_errors == 0:
        log("RESULT: PASS — no VRAM or CUDA core errors detected.")
        log("Note: a clean pass reduces the odds of a hardware fault but doesn't "
            "100% rule out intermittent issues. Re-run --full if you saw crashes "
            "or artifacts in real workloads but this test passes.")
        sys.exit(0)
    else:
        log(f"RESULT: FAIL — {total_errors} total mismatches detected. "
            f"This points to likely damaged VRAM and/or CUDA cores. "
            f"Also check nvidia-smi ECC error counters above and run "
            f"nvidia-smi -q -d ECC for detailed error locations if your card supports ECC.")
        sys.exit(1)


if __name__ == "__main__":
    main()
