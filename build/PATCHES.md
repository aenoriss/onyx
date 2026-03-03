# OpenMVS Source Patches

Applied at build time via `sed` in `Dockerfile.onyx-openmvs`.
Target: OpenMVS v2.3.0 — `libs/MVS/SceneDensify.cpp`

---

## Patch 1 — FilterDepthMaps: skip empty depth maps (crash fix)

**Commit:** `dc89bea`
**Status:** Active (defensive safety net)

### Problem
`DenseReconstructionFilter` (the FilterDepthMaps event handler) guards the call
to `FilterDepthMap` with `IsValid()` but not `IsEmpty()`. When geom-consistency
empties a depth map (sets all depths to 0) for cameras with no stereo coverage,
`FilterDepthMap` hits an internal `ASSERT(!IsEmpty())` → `exit(1)` in v2.1.0 or
SIGSEGV in v2.3.0.

### Affected code
```cpp
// BEFORE (SceneDensify.cpp, DenseReconstructionFilter)
if (!depthData.IsValid()) {
    data.SignalCompleteDepthmapFilter();
    break;
}
// FilterDepthMap called even when depthData.IsEmpty() → ASSERT → crash
```

### Fix
```cpp
// AFTER
if (!depthData.IsValid() || depthData.IsEmpty()) {
    data.SignalCompleteDepthmapFilter();
    break;
}
```

### sed command
```bash
sed -i \
    '/if (!depthData\.IsValid()) {/{N; /SignalCompleteDepthmapFilter/s/if (!depthData\.IsValid()) {/if (!depthData.IsValid() || depthData.IsEmpty()) {/}' \
    libs/MVS/SceneDensify.cpp
```

### Revert
```bash
sed -i \
    's/if (!depthData\.IsValid() || depthData\.IsEmpty()) {/if (!depthData.IsValid()) {/' \
    libs/MVS/SceneDensify.cpp
```

### Notes
- Currently a no-op in normal operation (Patch 2 prevents depth maps from being
  emptied). Kept as a safety net for edge cases.
- `MergeDepthMaps` and `FuseDepthMaps` already handle empty maps with
  `if (depthData.IsEmpty()) continue;` — this patch makes the filter caller
  consistent with that pattern.

---

## Patch 2 — Geom-consistency: preserve depth maps when geo.dmap missing (cascade fix)

**Commit:** `dbd1ba8`
**Status:** Active (root cause fix)

### Problem
After each geom-consistency iteration, OpenMVS unconditionally:
1. `File::deleteFile(rawName)` — deletes the valid photometric `.dmap`
2. `File::renameFile("geo.dmap", rawName)` — replaces it with the geo-consistent version

For cameras that **failed** geom-consistency (e.g. floor/ceiling cameras in 360°
rigs with 0 stereo pairs), `geo.dmap` is never created. So `deleteFile` removes
the only valid depth data and `renameFile` silently fails. The camera now has no
`.dmap` file.

`FuseDepthMaps` then calls `IncRef(fileName)` for each camera → returns 0
(file missing) → `connection.score = 0` → removed from connections list. In 360°
scenes where every camera's neighbourhood includes some floor/ceiling cameras,
this cascades until `connections.empty()` → FuseDepthMaps returns **0 fused
points**.

### Affected code
```cpp
// BEFORE (SceneDensify.cpp, geom-consistency iteration, ~line 2156)
// replace raw depth-maps with the geometric-consistent ones
for (IIndex idx: data.images) {
    const DepthData& depthData(data.depthMaps.arrDepthData[idx]);
    if (!depthData.IsValid())
        continue;
    const String rawName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
    File::deleteFile(rawName);           // ← always deletes valid dmap
    File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName);
    // ↑ silently fails if geo.dmap doesn't exist → camera loses its only dmap
}
```

### Fix
Only delete + rename when `geo.dmap` actually exists. Cameras that completed
geom-consistency get geometrically-refined depths; cameras that did not keep
their photometric depth maps and still contribute to `FuseDepthMaps`.

```cpp
// AFTER
for (IIndex idx: data.images) {
    const DepthData& depthData(data.depthMaps.arrDepthData[idx]);
    if (!depthData.IsValid())
        continue;
    const String rawName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
    if (File::isFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"))) {
        File::deleteFile(rawName);
        File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName);
    }
}
```

### sed command (as used in Dockerfile — produces single-line conditional)
```bash
sed -i \
    '/File::deleteFile(rawName);/{N; /File::renameFile.*geo\.dmap/s/.*\n.*/\t\tif (File::isFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"))) { File::deleteFile(rawName); File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName); }/}' \
    libs/MVS/SceneDensify.cpp
```

### Revert
```bash
sed -i \
    's/if (File::isFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"))) { File::deleteFile(rawName); File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName); }/File::deleteFile(rawName);\n\t\tFile::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName);/' \
    libs/MVS/SceneDensify.cpp
```

### Behaviour after patch
| Camera type | geo.dmap exists? | Result |
|---|---|---|
| Has stereo pairs, completed geom-consistency | Yes | Geometric-refined depth (best quality) |
| Has stereo pairs, geom-consistency failed (neighbour missing) | No | Photometric-only depth (kept, contributes to fusion) |
| 0 stereo pairs (floor/ceiling 360° cameras) | No | No depth map (never estimated) |

### Notes
- No regression for normal scenes: all cameras complete geom-consistency, all
  `geo.dmap` files exist → behaviour identical to unpatched code.
- Replaces the `--geometric-iters 0` workaround that was previously applied for
  360° scenes via `run_pipeline.py`. That workaround is now removed.
- The `--no-geom-consistency` flag still exists in `wrapper.py` as a user option
  but is no longer set automatically.
- Verified in Docker build: `grep -q "File::isFile.*geo.dmap" libs/MVS/SceneDensify.cpp`
