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

presets = []

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
                    "inherits": "release" if target.startswith("cpu") else target,
                    "cacheVariables": {
                        "GRID_ROWS": str(rows),
                        "GRID_COLS": str(cols),
                        "BLOCK_SIZE_X": str(bx),
                        "BLOCK_SIZE_Y": str(by),
                        **target_flags
                    }
                }
                if is_2d:
                    preset["cacheVariables"]["ARRAY_2D"] = True
                presets.append(preset)

# Final JSON structure
cmake_presets = {
    "version": 8,
    "configurePresets": presets
}

# Save to file
with open("GeneratedCMakePresets.json", "w") as f:
    json.dump(cmake_presets, f, indent=4)

print(f"Generated {len(presets)} presets in GeneratedCMakePresets.json")
