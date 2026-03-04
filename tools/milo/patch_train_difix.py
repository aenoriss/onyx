#!/usr/bin/env python3
"""
Apply Difix3D+ integration patch to MILo's train.py.

Run once at Docker build time:
    python3 /workspace/patch_train_difix.py

Makes three minimal, targeted edits to /workspace/MILo/milo/train.py:
  1. Adds --difix3d and --difix3d_iters arguments to the argparser.
  2. Initialises the Difix model and schedules fix iterations at
     the start of the training() function (before the main loop).
  3. Calls difix_fix_step() after each optimizer/checkpoint step when
     the current iteration matches a scheduled fix iteration.

All changes are conditional on getattr(args, 'difix3d', False), so the
default MILo flow is completely unchanged when --difix3d is not passed.
"""

import sys

TRAIN_PY = "/workspace/MILo/milo/train.py"

with open(TRAIN_PY, "r") as f:
    code = f.read()

# ── Patch 1: argparser — add --difix3d and --difix3d_iters ───────────────────
# Inserted right after the existing --wandb_entity argument.

_P1_OLD = (
    '    # ----- Logging -----\n'
    '    parser.add_argument("--log_interval", type=int, default=None)\n'
    '    parser.add_argument("--wandb_project", type=str, default=None)\n'
    '    parser.add_argument("--wandb_entity", type=str, default=None)'
)

_P1_NEW = (
    '    # ----- Logging -----\n'
    '    parser.add_argument("--log_interval", type=int, default=None)\n'
    '    parser.add_argument("--wandb_project", type=str, default=None)\n'
    '    parser.add_argument("--wandb_entity", type=str, default=None)\n'
    '\n'
    '    # ----- Difix3D+ Fix Cycles -----\n'
    '    parser.add_argument(\n'
    '        "--difix3d", action="store_true", default=False,\n'
    '        help="Enable Difix3D+ fix cycles within MILo training",\n'
    '    )\n'
    '    parser.add_argument(\n'
    '        "--difix3d_iters", nargs="+", type=int, default=[9000, 13000, 17000],\n'
    '        help="Iterations at which to run Difix3D+ fix cycles",\n'
    '    )\n'
    '    parser.add_argument(\n'
    '        "--difix3d_views", type=int, default=16,\n'
    '        help="Novel views to generate per Difix3D+ fix cycle (default: 16)",\n'
    '    )'
)

assert _P1_OLD in code, "Patch 1 anchor not found in train.py"
code = code.replace(_P1_OLD, _P1_NEW, 1)
print("[patch] 1/3  --difix3d argparse args added")

# ── Patch 2: training() — initialise Difix before the main loop ──────────────
# Inserted right after:  first_iter += 1

_P2_OLD = (
    '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")\n'
    '    first_iter += 1'
)

_P2_NEW = (
    '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")\n'
    '    first_iter += 1\n'
    '\n'
    '    # --- Difix3D+ initialisation (no-op unless --difix3d is set) ---\n'
    '    _difix_pipe = None\n'
    '    _difix_iters_set = set()\n'
    '    if getattr(args, "difix3d", False):\n'
    '        from difix_integration import init_difix, difix_fix_step as _difix_fix_step\n'
    '        _difix_pipe = init_difix("cuda")\n'
    '        _difix_iters_set = set(getattr(args, "difix3d_iters", [9000, 13000, 17000]))\n'
    '        print(f"[difix] Fix cycles scheduled at: {sorted(_difix_iters_set)}")'
)

assert _P2_OLD in code, "Patch 2 anchor not found in train.py"
code = code.replace(_P2_OLD, _P2_NEW, 1)
print("[patch] 2/3  Difix initialisation block added")

# ── Patch 3: training loop — call fix step at scheduled iterations ────────────
# Inserted before the memory-cleanup block at the bottom of each iteration.
# This position is OUTSIDE torch.no_grad() (needed for backward) and runs after
# the normal optimizer + checkpoint step.

_P3_OLD = (
    '        if iteration % 100 == 0:\n'
    '            torch.cuda.empty_cache()\n'
    '            gc.collect()'
)

_P3_NEW = (
    '        # --- Difix3D+ fix cycle (runs outside torch.no_grad()) ---\n'
    '        if _difix_pipe is not None and iteration in _difix_iters_set:\n'
    '            _difix_fix_step(\n'
    '                gaussians=gaussians, scene=scene,\n'
    '                difix_pipe=_difix_pipe, render_func=render, pipe=pipe,\n'
    '                gaussians_optimizer=gaussians.optimizer, opt=opt,\n'
    '                background=background, iteration=iteration,\n'
    '                n_views=getattr(args, "difix3d_views", 16),\n'
    '            )\n'
    '            viewpoint_stack = None  # force viewpoint refresh on next iter\n'
    '\n'
    '        if iteration % 100 == 0:\n'
    '            torch.cuda.empty_cache()\n'
    '            gc.collect()'
)

assert _P3_OLD in code, "Patch 3 anchor not found in train.py"
code = code.replace(_P3_OLD, _P3_NEW, 1)
print("[patch] 3/3  Difix fix-cycle call inserted into training loop")

with open(TRAIN_PY, "w") as f:
    f.write(code)

print(f"\n[patch] Successfully patched {TRAIN_PY}")
print("[patch] Default MILo flow is unchanged — Difix only active with --difix3d")
