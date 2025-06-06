import json
import os
import subprocess
from pathlib import Path

CMAKE_PRESETS_FILE = "CMakePresets.json"
SOURCE_DIR = Path(".").resolve()
BUILD_DIR_BASE = SOURCE_DIR / "build"

def load_presets():
    with open(CMAKE_PRESETS_FILE, "r") as f:
        return json.load(f)["configurePresets"]

def get_cache_value(cache, key):
    return int(cache[key]) if key in cache else None

def get_cmd_args(preset):
    cache = preset.get("cacheVariables", {})
    name = preset["name"]
    rows = get_cache_value(cache, "GRID_ROWS")
    cols = get_cache_value(cache, "GRID_COLS")
    bx = get_cache_value(cache, "BLOCK_SIZE_X")
    by = get_cache_value(cache, "BLOCK_SIZE_Y")
    is_2d = "ARRAY_2D" in cache

    if None in [rows, cols, bx, by]:
        raise ValueError(f"Missing grid or block dimensions in preset {name}")

    if is_2d:
        filename = SOURCE_DIR / "data" / f"{name}_{rows}x{cols}_{bx}x{by}_2d.csv"
    else:
        bprod = bx * by
        filename = SOURCE_DIR / "data" / f"{name}_{rows}x{cols}_{bprod}_1d.csv"

    return ["16", str(filename)]

def main():
    presets = load_presets()

    for preset in presets:
        name = preset["name"]
        if name in ["cpu", "opencl", "cuda"]: continue

        print(f"\n>>> Processing preset: {name}")

        build_dir = BUILD_DIR_BASE / name
        build_dir.mkdir(parents=True, exist_ok=True)

        # Configure
        subprocess.run(["cmake", "--preset", name], check=True)

        # Build
        subprocess.run(["cmake", "--build", "--preset", name, "--clean-first"], check=True)

        # Path to binary (assume same name as target, adjust if not)
        exe_name = ""
        cwd = f'src'
        if "cpu" in name:
            if "parallel" in name:
                exe_name = f'{cwd}/{"CPUParallelBenchmark.exe" if os.name == "nt" else "CPUParallelBenchmark"}'
            else:
                exe_name = f'{cwd}/{"CPUSerialBenchmark.exe" if os.name == "nt" else "CPUSerialBenchmark"}'
        elif "opencl" in name:
            exe_name = f'{cwd}/{"OpenCLBenchmark.exe" if os.name == "nt" else "OpenCLBenchmark"}'
        else:
            cwd = f'{cwd}/Debug'
            exe_name = f'{cwd}/{"CudaBenckmark.exe" if os.name == "nt" else "CudaBenckmark"}'
        binary_path = build_dir / exe_name
        cwd = build_dir / cwd

        if not binary_path.exists():
            raise FileNotFoundError(f"Executable not found: {binary_path}")

        # Run with arguments
        args = get_cmd_args(preset)
        print(f"Running: {binary_path} {' '.join(args)}")
        subprocess.run([str(binary_path)] + args, check=True, cwd= cwd)

if __name__ == "__main__":
    main()
