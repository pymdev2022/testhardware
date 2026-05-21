import platform
import os
import subprocess
import time
import math
import sys
from concurrent.futures import ProcessPoolExecutor

# Try importing torch for GPU benchmarking
try:
    import torch
except ImportError:
    torch = None

def run_command(command: list[str]) -> str | None:
    """Helper function to run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    except Exception as e:
        print(f"Error running command {command}: {e}", file=sys.stderr)
        return None

def get_system_info() -> dict:
    """Gathers general system information."""
    info = {
        "OS": platform.system(),
        "OS Release": platform.release(),
        "OS Version": platform.version(),
        "Architecture": platform.machine(),
        "Node Name": platform.node(),
        "Computer Model": "N/A"
    }

    if info["OS"] == "Windows":
        model = run_command(["wmic", "csproduct", "get", "name"])
        if model:
            info["Computer Model"] = model.split('\n')[-1].strip()
    elif info["OS"] == "Linux":
        # Removed 'sudo dmidecode' as it requires a password/login.
        model = run_command(["hostnamectl", "--static"])
        if model:
            info["Computer Model"] = model
    elif info["OS"] == "Darwin": # macOS
        model = run_command(["sysctl", "-n", "hw.model"])
        if model:
            info["Computer Model"] = model

    return info

def get_cpu_info() -> dict:
    """Gathers CPU information."""
    info = {
        "Processor": platform.processor(),
        "Logical Cores": os.cpu_count(),
        "CPU Version": "N/A"
    }

    if platform.system() == "Windows":
        # wmic output can have multiple lines, take the last non-empty one
        wmic_output = run_command(["wmic", "cpu", "get", "name"])
        if wmic_output:
            lines = [line.strip() for line in wmic_output.split('\n') if line.strip()]
            if len(lines) > 1: # Skip the header "Name"
                info["CPU Version"] = lines[-1]
    elif platform.system() == "Linux":
        # Try reading /proc/cpuinfo first for a more direct approach
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["CPU Version"] = line.split(":", 1)[1].strip()
                        break # Assuming all cores have the same model name
        except FileNotFoundError:
            # Fallback to lscpu if /proc/cpuinfo is not available
            lscpu_output = run_command(["lscpu"])
            if lscpu_output:
                for line in lscpu_output.split('\n'):
                    if "Model name:" in line:
                        info["CPU Version"] = line.split(":", 1)[1].strip()
                        break
    elif platform.system() == "Darwin": # macOS
        cpu_name = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if cpu_name:
            info["CPU Version"] = cpu_name

    return info

def get_gpu_info() -> dict:
    """
    Gathers GPU information.
    Note: This is highly dependent on installed drivers and tools.
    """
    info = {
        "GPU Version": "N/A",
        "GPU Driver": "N/A"
    }

    # NVIDIA GPUs
    # Querying for Name, Driver, Total Memory, and Used Memory
    nvidia_smi_output = run_command([
        "nvidia-smi", 
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", 
        "--format=csv,noheader,nounits"
    ])
    if nvidia_smi_output:
        gpus = []
        for line in nvidia_smi_output.split('\n'):
            if line.strip():
                name, driver, mem_total, mem_used, util = line.split(',')
                gpus.append(f"{name.strip()} | Driver: {driver.strip()} | VRAM: {mem_used.strip()}/{mem_total.strip()} MB | Load: {util.strip()}%")
        if gpus:
            info["GPU Version"] = ", ".join(gpus)
            try:
                # Extract driver from the formatted string: "Name | Driver: version | ..."
                info["GPU Driver"] = gpus[0].split('| Driver: ')[1].split('|')[0].strip()
            except (IndexError, AttributeError):
                info["GPU Driver"] = "Unknown"

    # AMD GPUs (requires rocm-smi on Linux)
    if info["GPU Version"] == "N/A" and platform.system() == "Linux":
        rocm_smi_output = run_command(["rocm-smi", "--showproductname", "--csv"])
        if rocm_smi_output:
            # Example output: "Card,gfx906,Radeon Instinct MI50"
            lines = rocm_smi_output.split('\n')
            if len(lines) > 1:
                gpu_names = [line.split(',')[-1].strip() for line in lines[1:] if line.strip()]
                if gpu_names:
                    info["GPU Version"] = ", ".join(gpu_names)

    # Intel GPUs (Linux via lspci)
    if info["GPU Version"] == "N/A" and platform.system() == "Linux":
        lspci_output = run_command(["lspci", "-vnn"])
        if lspci_output:
            vga_devices = [line for line in lspci_output.split('\n') if "VGA compatible controller" in line]
            if vga_devices:
                info["GPU Version"] = "; ".join([dev.split(':')[-1].strip() for dev in vga_devices])

    # macOS (requires system_profiler)
    if info["GPU Version"] == "N/A" and platform.system() == "Darwin":
        system_profiler_output = run_command(["system_profiler", "SPDisplaysDataType"])
        if system_profiler_output:
            gpu_lines = [line for line in system_profiler_output.split('\n') if "Chipset Model:" in line]
            if gpu_lines:
                info["GPU Version"] = "; ".join([line.split(':')[-1].strip() for line in gpu_lines])

    return info

def get_specialized_accelerator_info() -> dict:
    """
    Attempts basic detection of specialized accelerators (NPU, TPU, DPU).
    """
    info = {
        "NPU": "Not detected",
        "TPU": "Not detected",
        "DPU": "Not detected"
    }

    if platform.system() == "Windows":
        # Scan for common NPU/TPU strings in Device Manager entities
        pnp_entities = run_command(["wmic", "path", "win32_pnpentity", "get", "name"])
        if pnp_entities:
            pnp_entities_lower = pnp_entities.lower()
            if any(x in pnp_entities_lower for x in ["intel(r) ai boost", "neural processor", "npu"]):
                info["NPU"] = "Detected (Generic/Intel/Qualcomm NPU)"
            if "edge tpu" in pnp_entities_lower:
                info["TPU"] = "Detected (Google Edge TPU)"
            if "bluefield" in pnp_entities_lower:
                info["DPU"] = "Detected (NVIDIA BlueField DPU)"

    elif platform.system() == "Linux":
        # Check PCIe devices for common accelerator IDs
        lspci_output = run_command(["lspci", "-n"])
        if lspci_output:
            # Intel NPU (Meteor Lake/Core Ultra) IDs: 7b50, 643e
            if "8086:7b50" in lspci_output or "8086:643e" in lspci_output:
                info["NPU"] = "Detected (Intel NPU)"
            # Coral Edge TPU (PCIe) ID: 1ac1:089a
            if "1ac1:089a" in lspci_output:
                info["TPU"] = "Detected (Google Edge TPU PCIe)"
            # NVIDIA BlueField DPU IDs (partial)
            if "15b3:a2d" in lspci_output:
                info["DPU"] = "Detected (NVIDIA BlueField DPU)"
        
        # Check USB for Edge TPU
        lsusb_output = run_command(["lsusb"])
        if lsusb_output and "1a6e:089a" in lsusb_output:
            info["TPU"] = "Detected (Google Edge TPU USB)"

    return info

def is_prime(n):
    """Prime check function for benchmarking."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _run_prime_chunk(r):
    """Helper function for multi-threaded prime counting."""
    return sum(1 for x in range(r[0], r[1]) if is_prime(x))

def run_cpu_benchmark(iterations: int = 1000000) -> dict:
    """
    Runs single-threaded and multi-threaded CPU-bound benchmarks.
    """
    print(f"\nRunning CPU benchmarks ({iterations} iterations)...")
    
    # Single-threaded
    start_time = time.time()
    primes_found_st = sum(1 for i in range(2, iterations + 2) if is_prime(i))
    st_duration = time.time() - start_time
    print(f"  Single-threaded: {st_duration:.4f} seconds (Found {primes_found_st} primes)")

    # Multi-threaded (using all available logical cores)
    cores = os.cpu_count() or 1
    chunk_size = iterations // cores
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        # Split the work across cores
        ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(cores)]
        results = list(executor.map(_run_prime_chunk, ranges))
        
    mt_duration = time.time() - start_time
    print(f"  Multi-threaded ({cores} cores): {mt_duration:.4f} seconds")

    return {
        "Single-threaded": st_duration,
        "Multi-threaded": mt_duration,
        "Speedup Multi-core": st_duration / mt_duration
    }

def run_gpu_benchmark(matrix_size: int = 1024, num_runs: int = 5) -> dict:
    """
    Runs a simple GPU-bound benchmark (matrix multiplication) using PyTorch.
    Calculates approximate TFLOPS.
    Returns a dictionary with benchmark results.
    """
    results = {
        "Status": "Skipped (PyTorch not installed or CUDA not available)",
        "Time (s)": "N/A",
        "TFLOPS (approx)": "N/A"
    }

    if torch is None:
        results["Status"] = "Skipped (PyTorch not installed)"
        return results

    if not torch.cuda.is_available():
        results["Status"] = "Skipped (CUDA not available)"
        return results

    device = torch.device("cuda")
    results["Status"] = f"Running on {torch.cuda.get_device_name(device)}"
    print(f"\nRunning GPU benchmark (Matrix Multiplication {matrix_size}x{matrix_size}, {num_runs} runs)...")

    try:
        # Warm-up run
        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)
        _ = torch.matmul(A, B)
        torch.cuda.synchronize() # Ensure warm-up is complete

        timings = []
        for _ in range(num_runs):
            A = torch.randn(matrix_size, matrix_size, device=device)
            B = torch.randn(matrix_size, matrix_size, device=device)
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            C = torch.matmul(A, B)
            end_event.record()
            
            torch.cuda.synchronize() # Wait for the computation to complete
            timings.append(start_event.elapsed_time(end_event) / 1000.0) # Convert ms to seconds

        avg_time = sum(timings) / num_runs
        results["Time (s)"] = f"{avg_time:.4f}"

        # Calculate FLOPS for N x N matrix multiplication: 2 * N^3
        # (N^3 multiplications and N^3 additions, approximately)
        flops = 2 * (matrix_size ** 3)
        tflops = (flops / avg_time) / 1e12
        results["TFLOPS (approx)"] = f"{tflops:.2f}"

    except Exception as e:
        results["Status"] = f"Error during GPU benchmark: {e}"

    return results

def main():
    print("--- System Information ---")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key}: {value}")

    print("\n--- CPU Information ---")
    cpu_info = get_cpu_info()
    for key, value in cpu_info.items():
        print(f"{key}: {value}")

    print("\n--- GPU Information ---")
    gpu_info = get_gpu_info()
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    if gpu_info["GPU Version"] == "N/A" and torch is None: # Only show this note if PyTorch is not installed
        print("  (Note: GPU detection relies on external tools like nvidia-smi, rocm-smi, lspci, system_profiler. Install them for more accurate results.)")

    print("\n--- Specialized Accelerator Information (NPU/TPU/DPU) ---")
    accelerator_info = get_specialized_accelerator_info()
    for key, value in accelerator_info.items():
        print(f"{key}: {value}")
    print("  (Note: Performance for NPUs is typically measured in TOPS (Integer ops), while GPUs focus on TFLOPS (Floating-point).)")
    if all(v == "Not detected" for v in accelerator_info.values()) and torch is None:
        print("  (General Note: Detecting and benchmarking these often requires vendor-specific SDKs and is beyond a generic script.)")

    cpu_results = run_cpu_benchmark()
    print(f"CPU Single-core Time: {cpu_results['Single-threaded']:.4f}s")
    print(f"CPU Multi-core Time:  {cpu_results['Multi-threaded']:.4f}s")
    print(f"Multi-core Speedup:   {cpu_results['Speedup Multi-core']:.2f}x")

    print("\n--- GPU Benchmark ---")
    gpu_benchmark_results = run_gpu_benchmark()
    for key, value in gpu_benchmark_results.items():
        print(f"{key}: {value}")
    if gpu_benchmark_results["Status"].startswith("Skipped"):
        print("  (Note: GPU TFLOPS calculation requires PyTorch with CUDA support installed.)")

if __name__ == "__main__":
    main()
