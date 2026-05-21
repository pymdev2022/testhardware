# Hardware Check & Benchmark Tool

A comprehensive Python script to identify system hardware components and perform performance benchmarks across different processing units.

## Features

- **System Information**: Identifies OS version, Architecture, and Computer Model.
- **CPU Details**: Reports processor type, logical core count, and model versions.
- **GPU Analytics**: Detects NVIDIA, AMD, Intel, and Apple GPUs, including VRAM usage and driver versions.
- **Specialized Accelerators**: Scans for NPUs (Intel AI Boost), TPUs (Google Coral), and DPUs (NVIDIA BlueField).
- **CPU Benchmark**: 
  - Single-threaded prime number calculation.
  - Multi-threaded prime number calculation using all available cores.
  - Multi-core speedup ratio.
- **GPU Benchmark**: Calculates approximate **TFLOPS** (Tera Floating-point Operations Per Second) using matrix multiplication (requires PyTorch).

## Setup

### 1. Prerequisites
The script uses standard Python libraries, but for advanced GPU features, it relies on several system-level tools. Ensure these are installed for best results:
- **Windows**: `wmic` (included by default).
- **Linux**: `hostnamectl`, `lscpu`, `lspci`, `lsusb`.
- **NVIDIA GPUs**: `nvidia-smi` (installed with NVIDIA drivers).

### 2. Python Environment
It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment
python -m venv EnvTH
source EnvTH/bin/activate  # On Linux/macOS
# Or on Windows PowerShell:
# .\EnvTH\Scripts\Activate.ps1
```

### 3. Dependencies
Install **PyTorch** if you wish to run the GPU TFLOPS benchmark:

```bash
pip install torch
```

## Running the Tool

Simply execute the script with Python:

```bash
python CheckHardware.py
```

*Note: The script does not require root/administrator privileges for basic operation, as it avoids commands like `sudo dmidecode`.*
