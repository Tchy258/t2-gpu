import json
from math import isqrt

# Define grid sizes (product from 8192 up to 2^31 - 1)
grid_sizes = [
    (128, 64),       # 8192
    (512, 512),      # 262144
    (1024, 1024),    # 1M
    (2048, 4096),    # 8M
    (8192, 8192),    # 67M
    (32768, 32768)   # ~1B
]

targets = {
    "cpu-serial": {"USE_CPU": True, "USE_PARALLEL": False},
    "cpu-parallel": {"USE_CPU": True, "USE_PARALLEL": True},
    "opencl": {"USE_OPENCL": True},
    "cuda": {"USE_CUDA": True}
}

def make_preset_name(tag, r, c, bx, by, is_2d, bad):
    parts = [tag, f"grid{r}x{c}", f"block{bx}x{by}"]
    if is_2d:
        parts.append("2d")
    else:
        parts.append("1d")
    if bad:
        parts.append("unaligned")
    return "-".join(parts)

# Editar Paths aquí si es necesario
presets = [
    {
            "name": "cpu",
            "displayName": "GCC Release",
            "description": "Using compilers: C = C:\\msys64\\ucrt64\\bin\\gcc.exe, CXX = C:\\msys64\\ucrt64\\bin\\g++.exe",
            "generator": "MinGW Makefiles",
            "binaryDir": "${sourceDir}/build/${presetName}",
            "cacheVariables": {
                "GRID_ROWS": "128",
                "GRID_COLS": "64",
                "BLOCK_SIZE_X": "32",
                "BLOCK_SIZE_Y": "32",
                "ARRAY_2D": False,
                "USE_CPU": True,
                "USE_PARALLEL": False,
                "CMAKE_INSTALL_PREFIX": "${sourceDir}/out/install/${presetName}",
                "CMAKE_C_COMPILER": "C:/msys64/ucrt64/bin/gcc.exe",
                "CMAKE_CXX_COMPILER": "C:/msys64/ucrt64/bin/g++.exe"
            }
        },
        {
            "name": "opencl",
            "displayName": "MinGW OpenCL",
            "description": "Build with USE_OPENCL=ON",
            "generator": "MinGW Makefiles",
            "binaryDir": "${sourceDir}/build/${presetName}",
            "cacheVariables": {
                "GRID_ROWS": "128",
                "GRID_COLS": "64",
                "BLOCK_SIZE_X": "32",
                "BLOCK_SIZE_Y": "32",
                "ARRAY_2D": False,
                "CMAKE_C_COMPILER": "C:/msys64/ucrt64/bin/gcc.exe",
                "CMAKE_CXX_COMPILER": "C:/msys64/ucrt64/bin/g++.exe",
                "USE_OPENCL": True
            }
        },
        {
            "name": "cuda",
            "displayName": "VS2022 + CUDA (default toolset)",
            "generator": "Visual Studio 17 2022",
            "binaryDir": "${sourceDir}/build/${presetName}",
            "toolset": "host=x64",
            "architecture": "x64",
            "cacheVariables": {
                "GRID_ROWS": "128",
                "GRID_COLS": "64",
                "BLOCK_SIZE_X": "32",
                "BLOCK_SIZE_Y": "32",
                "ARRAY_2D": False,
                "CMAKE_C_COMPILER": "cl.exe",
                "CMAKE_CXX_COMPILER": "cl.exe",
                "USE_CUDA": True
            }
        },
]
build_presets = []
for (rows, cols) in grid_sizes:
    for is_2d in [True, False]:
        for bad_block in [False, True]:
            # Good block sizes
            if is_2d:
                if not bad_block:
                    bx, by = 32, 32
                else:
                    bx, by = 30, 30
            else:
                if not bad_block:
                    bx, by = 8, 4  # product = 32
                else:
                    bx, by = 7, 5  # product = 35

            for target, target_flags in targets.items():
                preset = {
                    "name": make_preset_name(target, rows, cols, bx, by, is_2d, bad_block),
                    "displayName": f"{target} - {rows}x{cols} - Block {bx}x{by} - {'2D' if is_2d else '1D'}{' - Bad' if bad_block else ''}",
                    "inherits": "cpu" if "cpu" in target else target,
                    "cacheVariables": {
                        "GRID_ROWS": str(rows),
                        "GRID_COLS": str(cols),
                        "BLOCK_SIZE_X": str(bx),
                        "BLOCK_SIZE_Y": str(by),
                        **target_flags
                    }
                }
                build_preset = {
                    "name": make_preset_name(target, rows, cols, bx, by, is_2d, bad_block),
                    "displayName": f"{target} - {rows}x{cols} - Block {bx}x{by} - {'2D' if is_2d else '1D'}{' - Bad' if bad_block else ''}",
                    "description": "",
                    "configurePreset": make_preset_name(target, rows, cols, bx, by, is_2d, bad_block)
                }
                if is_2d:
                    preset["cacheVariables"]["ARRAY_2D"] = True
                presets.append(preset)
                build_presets.append(build_preset)
# Final JSON structure
cmake_presets = {
    "version": 8,
    "configurePresets": presets,
    "buildPresets": build_presets
}

# Save to file
with open("GeneratedCMakePresets.json", "w") as f:
    json.dump(cmake_presets, f, indent=4)

print(f"Generated {len(presets)} presets in GeneratedCMakePresets.json")
