#!/usr/bin/env python3
"""Onyx pipeline progress reporting — shared across all containers.

Installed in onyx-base at /workspace/pipeline_progress.py so all
downstream images inherit it. The orchestrator polls .progress.json
to render a live dashboard.

Usage in wrappers:
    from pipeline_progress import progress, run_with_progress

    progress("initializing", step=1, total_steps=2)
    run_with_progress(["some-tool", "--flag"], stage="training",
                      step=1, total_steps=2, patterns={...})
    progress("done", status="completed", pct=100, step=2, total_steps=2)
"""

import json
import os
import re
import subprocess
import sys
import time

PROGRESS_FILE = os.environ.get("ONYX_PROGRESS_FILE", "/data/.progress.json")


def progress(stage, status="running", pct=None, detail="",
             step=None, total_steps=None, error=None):
    """Write progress JSON atomically (tmp + rename).

    Args:
        stage: Current substep name (e.g. "training", "mesh_extraction")
        status: "running", "completed", or "failed"
        pct: Percentage 0-100 within this substep, or None if indeterminate
        detail: Human-readable context (e.g. "Step 9900/18000")
        step: Position in substep sequence (1-indexed)
        total_steps: Total substeps for this container
        error: Error message on failure (last lines of output)
    """
    try:
        data = {
            "stage": stage,
            "status": status,
            "percent": pct,
            "detail": detail,
            "step": step,
            "total_steps": total_steps,
            "error": error,
            "t": time.time(),
        }
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, PROGRESS_FILE)
    except Exception:
        pass  # Never crash the tool over progress reporting


def run_with_progress(cmd, stage, step=None, total_steps=None,
                      patterns=None, cwd=None):
    """Run subprocess, stream stdout, parse lines for progress updates.

    Args:
        cmd: Command list to execute
        stage: Substep name for progress reporting
        step: Position in substep sequence (1-indexed)
        total_steps: Total substeps for this container
        patterns: Dict of {regex_string: handler_fn(match) -> (pct, detail)}.
                  Each line of stdout is tested against all patterns.
        cwd: Working directory for the subprocess

    Returns:
        subprocess return code (also calls sys.exit on failure)
    """
    progress(stage, "running", 0, step=step, total_steps=total_steps)

    print(f"\n{'='*60}")
    print(f"[STAGE] {stage}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=cwd,
    )

    compiled_patterns = {}
    if patterns:
        compiled_patterns = {re.compile(p): h for p, h in patterns.items()}

    last_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        last_lines.append(line.rstrip())
        if len(last_lines) > 20:
            last_lines.pop(0)
        for regex, handler in compiled_patterns.items():
            m = regex.search(line)
            if m:
                pct, detail = handler(m)
                progress(stage, "running", pct, detail,
                         step=step, total_steps=total_steps)

    proc.wait()

    if proc.returncode != 0:
        error_context = "\n".join(last_lines[-5:])
        progress(stage, "failed",
                 error=f"Exit code {proc.returncode}: {error_context}",
                 step=step, total_steps=total_steps)
        print(f"\n[ERROR] {stage} failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    progress(stage, "completed", 100, step=step, total_steps=total_steps)
    print(f"\n[SUCCESS] {stage} completed")
    return proc.returncode
