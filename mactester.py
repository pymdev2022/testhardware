import os
import platform
import subprocess


def check_mac_hardware():
    print("=== Analyzing Hardware Profile ===")

    # 1. Check Primary Operating System
    system_type = platform.system()
    print(f"Operating System: {system_type}")

    if system_type != "Darwin":
        print("[Result]: This machine is not running macOS.")
        return

    # 2. Extract Chip Name via macOS Native System Profiler
    try:
        cmd = ["system_profiler", "SPHardwareDataType"]
        output = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        chip_line = [
            line.strip() for line in output.stdout.split("\n") if "Chip:" in line
        ]
        model_line = [
            line.strip()
            for line in output.stdout.split("\n")
            if "Model Name:" in line
        ]

        hardware_chip = (
            chip_line[0].split(":")[1].strip() if chip_line else "Unknown"
        )
        hardware_model = (
            model_line[0].split(":")[1].strip() if model_line else "Unknown"
        )

        print(f"Detected Model:  {hardware_model}")
        print(f"Detected Chip:   {hardware_chip}")

    except Exception as e:
        print(f"Could not read system_profiler: {e}")
        hardware_chip = "Unknown"
        hardware_model = "Unknown"

    # 3. Detect Rosetta 2 Emulation Layer
    # Even if Python is compiled for x86_64, sysctl can tell us if the process is being translated
    try:
        # sysctl.proc_translated returns 1 if running under Rosetta translation
        res = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        is_translated = res.stdout.strip() == "1"
    except FileNotFoundError:
        is_translated = False

    # 4. Read Python Environment Architecture
    python_arch = platform.machine()
    print(f"Python Runtime Architecture: {python_arch}")
    print(f"Running via Rosetta Emulation: {is_translated}")

    print("\n=== Assessment ===")
    # Final Verdict Logic
    is_m1 = "M1" in hardware_chip or "M1" in platform.processor()
    is_air = "Air" in hardware_model

    if is_m1 and is_air:
        print("✅ Match: This is a MacBook Air running an Apple M1 chip.")
    elif is_m1:
        print(
            f"ℹ️ Partial Match: Apple M1 chip found, but model is a '{hardware_model}' (Not an Air)."
        )
    else:
        print("❌ No Match: This machine does not appear to be an M1 MacBook Air.")


if __name__ == "__main__":
    check_mac_hardware()
    