import time
import platform
import subprocess
import json
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def get_sys_info():
    """Gathers basic hardware identification."""
    info = {
        "os": platform.system(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    if info["os"] == "Darwin":
        try:
            info["model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            info["cores_logical"] = int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).decode().strip())
        except:
            pass
    return info

def benchmark_cpu(matrix_size=4000):
    """Benchmarks CPU using heavy matrix multiplication."""
    print(f"[*] Starting CPU Benchmark (Matrix Size: {matrix_size}x{matrix_size})...")
    start_time = time.time()
    
    # Create large matrices
    a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    
    # Perform multiplication
    result = np.dot(a, b)
    
    end_time = time.time()
    duration = end_time - start_time
    tflops = (2.0 * matrix_size**3) / duration / 1e12
    print(f"[-] CPU Benchmark Completed in {duration:.2f}s (~{tflops:.4f} TFLOPS)")
    return {"duration_sec": duration, "tflops": tflops}

def benchmark_gpu_mps(matrix_size=10000):
    """Benchmarks GPU using PyTorch and Metal Performance Shaders (MPS)."""
    if torch is None:
        return "PyTorch not installed. Skipping GPU test."
    
    if not torch.backends.mps.is_available():
        return "MPS (Metal) not available. This test requires Apple Silicon or a compatible Mac GPU."

    device = torch.device("mps")
    print(f"[*] Starting GPU Benchmark on {device} (Matrix Size: {matrix_size}x{matrix_size})...")
    
    # Warmup
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    _ = torch.mm(a, b)
    
    torch.mps.synchronize()
    start_time = time.time()
    
    # Stress test
    iterations = 5
    for _ in range(iterations):
        c = torch.mm(a, b)
    
    torch.mps.synchronize()
    end_time = time.time()
    
    duration = (end_time - start_time) / iterations
    tflops = (2.0 * matrix_size**3) / duration / 1e12
    print(f"[-] GPU Benchmark Completed in {duration:.4f}s per iteration (~{tflops:.4f} TFLOPS)")
    return {"duration_per_iter_sec": duration, "tflops": tflops}

def benchmark_neural_capacity():
    """
    Simulates an NPU/Neural workload using a deep learning inference simulation.
    On Mac, this often utilizes the ANE (Apple Neural Engine) via MPS/CoreML.
    """
    if torch is None or not torch.backends.mps.is_available():
        return "NPU simulation unavailable (requires PyTorch/MPS)."

    device = torch.device("mps")
    print("[*] Starting Neural Workload Simulation (Conv2D Layer Stress)...")
    
    # Simulate a typical high-density neural network layer (Convolution)
    # Large batch and channel size to fill the pipeline
    input_data = torch.randn(64, 512, 64, 64, device=device)
    weights = torch.randn(512, 512, 3, 3, device=device)
    
    # Warmup
    _ = torch.nn.functional.conv2d(input_data, weights)
    torch.mps.synchronize()
    
    start_time = time.time()
    iters = 50
    for _ in range(iters):
        _ = torch.nn.functional.conv2d(input_data, weights)
    
    torch.mps.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iters
    print(f"[-] Neural Simulation Completed. Avg Latency: {avg_latency*1000:.2f}ms")
    return {"avg_latency_ms": avg_latency * 1000}

def generate_report():
    sys_info = get_sys_info()
    print(f"\nHardware Report for: {sys_info.get('model', 'Unknown Device')}")
    print("=" * 50)
    
    results = {
        "system": sys_info,
        "benchmarks": {}
    }
    
    results["benchmarks"]["cpu"] = benchmark_cpu()
    
    gpu_res = benchmark_gpu_mps()
    if isinstance(gpu_res, dict):
        results["benchmarks"]["gpu"] = gpu_res
    else:
        print(f"[!] GPU Warning: {gpu_res}")

    npu_res = benchmark_neural_capacity()
    if isinstance(npu_res, dict):
        results["benchmarks"]["neural_engine_sim"] = npu_res
    else:
        print(f"[!] NPU Warning: {npu_res}")

    # Save to file
    report_filename = "hardware_performance_report.json"
    with open(report_filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print("=" * 50)
    print(f"✅ Report generated successfully: {report_filename}")
    
    # Summary output
    if "gpu" in results["benchmarks"]:
        cpu_perf = results["benchmarks"]["cpu"]["tflops"]
        gpu_perf = results["benchmarks"]["gpu"]["tflops"]
        ratio = gpu_perf / cpu_perf if cpu_perf > 0 else 0
        print(f"Summary: GPU is {ratio:.1f}x faster than CPU for matrix operations.")

if __name__ == "__main__":
    if torch is None:
        print("Please install dependencies: pip install numpy torch")
    else:
        try:
            generate_report()
        except KeyboardInterrupt:
            print("\nBenchmark cancelled by user.")