#!/usr/bin/env python3
"""Onyx 3D Reconstruction Pipeline Orchestrator.

Single script that chains all pipeline steps with persistent state tracking,
resume capability, and live progress dashboard.

State model:
    - INGEST and SFM are "shared steps" — run once, reused across forks
    - RECONSTRUCTION and SEGFORMER are "per-run steps" — each fork creates
      a new run with its own mode/quality and independent step tracking

Usage:
    # Install (pull all container images — run once)
    python run_pipeline.py --install

    # New run (auto-detects 360 video)
    python run_pipeline.py \\
        --input video.mp4 --output ./workdir \\
        --mode gaussian --scene outdoor --quality production

    # Force 360 mode
    python run_pipeline.py \\
        --input video.mp4 --output ./workdir \\
        --mode gaussian --scene outdoor --quality production --video 360

    # Resume failed run (resumes latest run)
    python run_pipeline.py --resume ./workdir

    # Resume from specific step
    python run_pipeline.py --resume ./workdir --from-step RECONSTRUCTION

    # Fork: run different reconstruction on same data (preserves history)
    python run_pipeline.py --fork ./workdir --mode photogrammetry --quality proto

    # Dry-run (print commands without executing)
    python run_pipeline.py --dry-run \\
        --input video.mp4 --output ./workdir \\
        --mode gaussian --scene outdoor --quality production

    # Purge working directory
    python run_pipeline.py --purge ./workdir
    python run_pipeline.py --purge --keep-output ./workdir
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

_WINDOWS = sys.platform == "win32"


def _setup_windows_console():
    """Enable ANSI escape codes and UTF-8 output on Windows.

    Without this:
    - ANSI codes (\033[2K etc.) print as literal garbage in CMD/PowerShell
    - Unicode characters (━ █ ░) raise UnicodeEncodeError on CP1252 consoles
    """
    if not _WINDOWS:
        return
    # Reconfigure stdout/stderr to UTF-8 (replaces unencodable chars instead of crashing)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Enable VT100 virtual terminal processing via Windows API
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-10, -11, -12):  # stdin, stdout, stderr
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass  # Non-fatal: ANSI codes may not render but script will still run


# ─── Constants ────────────────────────────────────────────────

GHCR_REGISTRY = "ghcr.io/aenoriss"

SHARED_STEPS = ["INGEST", "SFM"]
RUN_STEPS = ["RECONSTRUCTION", "SEGFORMER"]
ALL_STEPS = SHARED_STEPS + RUN_STEPS

TARGET_IMAGES = {
    ("outdoor", "production"): 300,
    ("outdoor", "proto"):      150,
    ("outdoor", "yono"):       150,
    ("indoor",  "production"): 150,
    ("indoor",  "proto"):      100,
    ("indoor",  "yono"):       100,
}

RECONSTRUCTION_CONTAINER = {
    ("gaussian", "production"):       "onyx-milo",
    ("gaussian", "proto"):            "onyx-gsplat",
    ("gaussian", "yono"):             "onyx-yonosplat",
    ("photogrammetry", "production"): "onyx-openmvs",
    ("photogrammetry", "proto"):      "onyx-openmvs",
}

ALL_CONTAINERS = [
    "onyx-ingest",
    "onyx-instantsfm",
    "onyx-milo",
    "onyx-gsplat",
    "onyx-yonosplat",
    "onyx-openmvs",
    "onyx-segformer",
]


# ─── Helpers ──────────────────────────────────────────────────

def uid_gid_flags():
    """Return --user UID:GID flags for Docker (Linux/macOS only).

    On Windows, Docker Desktop runs containers as the current user automatically,
    so no --user flag is needed.
    """
    if sys.platform == "win32":
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def docker_path(host_path):
    """Convert a host path to a Docker-compatible mount path.

    On Windows, Docker Desktop (WSL2 backend) expects paths like /c/Users/...
    instead of C:\\Users\\...
    """
    p = Path(host_path).resolve()
    if sys.platform == "win32":
        # Convert C:\foo\bar  →  /c/foo/bar
        drive, rest = os.path.splitdrive(str(p))
        drive_letter = drive.rstrip(":").lower()
        rest_posix = rest.replace("\\", "/")
        return f"/{drive_letter}{rest_posix}"
    return str(p)


def image_name(local, container):
    """Return full Docker image name."""
    if local:
        return f"{container}:latest"
    return f"{GHCR_REGISTRY}/{container}:latest"


def format_elapsed(seconds):
    """Format seconds as Xm Ys."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m {s:02d}s"


def check_docker():
    """Verify Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        if result.returncode != 0:
            return False, "Docker daemon not running"
        return True, result.stdout.strip()
    except FileNotFoundError:
        return False, "Docker not installed"
    except Exception as e:
        return False, str(e)


def check_image_exists(image):
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, timeout=10,
    )
    return result.returncode == 0


def check_images_installed(local=False):
    """Check that all pipeline images are pulled. Returns (ok, missing)."""
    missing = []
    for container in ALL_CONTAINERS:
        img = image_name(local, container)
        if not check_image_exists(img):
            missing.append(container)
    return (len(missing) == 0), missing


def do_install(local=False):
    """Pull all pipeline container images from ghcr.io."""
    ok, docker_info = check_docker()
    if not ok:
        print(f"Error: {docker_info}")
        print("Docker is required to run the Onyx pipeline.")
        sys.exit(1)

    print(f"Docker version: {docker_info}")
    print(f"Installing Onyx pipeline containers from {GHCR_REGISTRY}...\n")

    total = len(ALL_CONTAINERS)
    failed = []

    for i, container in enumerate(ALL_CONTAINERS, 1):
        img = image_name(local, container)

        # Check if already present
        if check_image_exists(img):
            print(f"  [{i}/{total}] {container:<24s} already installed")
            continue

        print(f"  [{i}/{total}] {container:<24s} pulling...", end="", flush=True)
        result = subprocess.run(
            ["docker", "pull", img],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            print(f"\r  [{i}/{total}] {container:<24s} done")
        else:
            print(f"\r  [{i}/{total}] {container:<24s} FAILED")
            failed.append((container, result.stderr.strip().split("\n")[-1]))

    print()
    if failed:
        print(f"[ERROR] {len(failed)} image(s) failed to pull:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)

    print(f"[DONE] All {total} containers installed")
    print(f"\nRun a pipeline with:")
    print(f"  python run_pipeline.py \\")
    print(f"    --input video.mp4 --output ./workdir \\")
    print(f"    --mode gaussian --scene outdoor --quality production")


# ─── Pipeline State ──────────────────────────────────────────

class PipelineState:
    """Persistent pipeline state — survives crashes, enables resume.

    State schema:
        {
            "id": "onyx-20260212-143022",
            "base_config": { "scene", "video", "input" },
            "started_at": "...",
            "shared_steps": {
                "INGEST": { "status": "completed", ... },
                "SFM": { "status": "completed", ... }
            },
            "runs": [
                {
                    "mode": "gaussian",
                    "quality": "production",
                    "started_at": "...",
                    "steps": {
                        "RECONSTRUCTION": { ... },
                        "SEGFORMER": { ... }
                    }
                }
            ]
        }
    """

    def __init__(self, path, data):
        self.path = Path(path)
        self.data = data
        self._active_run_idx = len(data.get("runs", [])) - 1

    @classmethod
    def create(cls, output_dir, config):
        """Create new pipeline state for a fresh run."""
        run_id = f"onyx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        data = {
            "id": run_id,
            "base_config": {
                "scene": config["scene"],
                "video": config["video"],
                "input": config["input"],
                "segment": config.get("segment", False),
            },
            "started_at": datetime.now().isoformat(),
            "shared_steps": {
                step: (
                    {"status": "skipped", "reason": "pose-free quality level"}
                    if step == "SFM" and config.get("quality") == "yono"
                    else {"status": "pending"}
                )
                for step in SHARED_STEPS
            },
            "runs": [
                {
                    "mode": config["mode"],
                    "quality": config["quality"],
                    "started_at": datetime.now().isoformat(),
                    "steps": {
                        step: {"status": "pending"}
                        for step in RUN_STEPS
                        if step != "SEGFORMER" or config.get("segment", False)
                    },
                }
            ],
        }
        state = cls(Path(output_dir) / "pipeline_state.json", data)
        state.save()
        return state

    @classmethod
    def load(cls, output_dir):
        """Load existing pipeline state for resume."""
        path = Path(output_dir) / "pipeline_state.json"
        if not path.exists():
            print(f"Error: No pipeline state found at {path}")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(path, data)

    def _compute_state_hash(self):
        """Compute a fingerprint from meaningful state (statuses + run configs).

        Changes when: a step completes/fails, a fork is added, a step resets.
        Does NOT change for: timestamps, elapsed times, error messages.
        """
        fingerprint = {
            "shared": {
                s: self.data["shared_steps"][s].get("status", "pending")
                for s in SHARED_STEPS
            },
            "runs": [
                {
                    "mode": r["mode"],
                    "quality": r["quality"],
                    "steps": {
                        s: r["steps"][s].get("status", "pending")
                        for s in r["steps"]
                    },
                }
                for r in self.data.get("runs", [])
            ],
        }
        raw = json.dumps(fingerprint, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    def save(self):
        """Atomic write (tmp + rename). Recomputes state hash."""
        self.data["state_hash"] = self._compute_state_hash()
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, str(self.path))

    @property
    def state_hash(self):
        return self.data.get("state_hash", "unknown")

    @property
    def run_id(self):
        return self.data["id"]

    @property
    def active_run(self):
        """Return the active run dict."""
        return self.data["runs"][self._active_run_idx]

    @property
    def base_config(self):
        return self.data["base_config"]

    def full_config(self, **overrides):
        """Merge base_config + active run config + overrides into a full config."""
        cfg = dict(self.base_config)
        cfg["mode"] = self.active_run["mode"]
        cfg["quality"] = self.active_run["quality"]
        cfg.update(overrides)
        return cfg

    def num_runs(self):
        return len(self.data["runs"])

    def set_active_run(self, idx):
        self._active_run_idx = idx

    # ── Step access (routes to shared_steps or active run) ────

    def _step_dict(self, step):
        """Return the dict containing this step's status."""
        if step in SHARED_STEPS:
            return self.data["shared_steps"]
        return self.active_run["steps"]

    def step_status(self, step):
        return self._step_dict(step).get(step, {}).get("status", "pending")

    def step_info(self, step):
        return self._step_dict(step).get(step, {})

    def start_step(self, step, container):
        self._step_dict(step)[step] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "container": container,
        }
        self.save()

    def complete_step(self, step):
        info = self._step_dict(step)[step]
        started = datetime.fromisoformat(info["started_at"])
        elapsed = (datetime.now() - started).total_seconds()
        info["status"] = "completed"
        info["completed_at"] = datetime.now().isoformat()
        info["elapsed"] = round(elapsed, 1)
        self.save()

    def fail_step(self, step, error, substep_info=None):
        info = self._step_dict(step)[step]
        started = datetime.fromisoformat(info["started_at"])
        elapsed = (datetime.now() - started).total_seconds()
        info["status"] = "failed"
        info["elapsed"] = round(elapsed, 1)
        info["error"] = error
        if substep_info:
            info.update(substep_info)
        self.save()

    def skip_step(self, step, reason=""):
        self._step_dict(step)[step] = {
            "status": "skipped",
            "reason": reason,
        }
        self.save()

    @property
    def all_steps(self):
        """Return ordered steps for this run — only includes steps present in state."""
        result = list(SHARED_STEPS)
        for s in RUN_STEPS:
            if s in self.active_run["steps"]:
                result.append(s)
        return result

    def reset_from(self, step):
        """Reset this step and all subsequent steps to pending."""
        found = False
        for s in self.all_steps:
            if s == step:
                found = True
            if found:
                self._step_dict(s)[s] = {"status": "pending"}
        self.save()

    def add_run(self, mode, quality):
        """Append a new run and make it active."""
        segment = self.data["base_config"].get("segment", False)
        run = {
            "mode": mode,
            "quality": quality,
            "started_at": datetime.now().isoformat(),
            "steps": {
                step: {"status": "pending"}
                for step in RUN_STEPS
                if step != "SEGFORMER" or segment
            },
        }
        self.data["runs"].append(run)
        self._active_run_idx = len(self.data["runs"]) - 1
        self.save()
        return run


# ─── Progress Dashboard ──────────────────────────────────────

class ProgressDashboard:
    """Polls .progress.json and renders live dashboard to stderr."""

    def __init__(self, state, output_dir):
        self.state = state
        self.output_dir = Path(output_dir)
        self.progress_file = self.output_dir / ".progress.json"
        self._stop = threading.Event()
        self._thread = None
        self._current_step = None
        self._step_start = None
        self._is_resumed = False

    def start(self, step, resumed=False):
        self._current_step = step
        self._step_start = time.time()
        self._is_resumed = resumed
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _poll_loop(self):
        while not self._stop.is_set():
            self._render()
            self._stop.wait(2)

    def _read_progress(self):
        try:
            if self.progress_file.exists():
                with open(self.progress_file, encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _render(self):
        lines = []
        run = self.state.active_run
        run_label = f"{run['mode']}/{run['quality']}"
        run_num = self.state.num_runs()
        run_tag = f" run {self.state._active_run_idx + 1}/{run_num}" if run_num > 1 else ""
        resumed_tag = " (resumed)" if self._is_resumed else ""

        lines.append(
            f"\r\033[K━━━ ONYX [{self.state.run_id}] "
            f"{run_label}{run_tag}{resumed_tag} ━━━━━━━━━"
        )

        live = self._read_progress()

        for step in self.state.all_steps:
            status = self.state.step_status(step)
            info = self.state.step_info(step)

            if status == "completed":
                elapsed = info.get("elapsed", 0)
                prev = " (previous run)" if self._is_resumed and step != self._current_step else ""
                lines.append(f" [+] {step:<20s} {format_elapsed(elapsed)}{prev}")

            elif status == "skipped":
                reason = info.get("reason", "")
                lines.append(f" [-] {step:<20s} (skipped{' — ' + reason if reason else ''})")

            elif step == self._current_step and status == "running":
                elapsed = time.time() - (self._step_start or time.time())
                detail = ""
                if live:
                    stage = live.get("stage", "")
                    s = live.get("step")
                    ts = live.get("total_steps")
                    pct = live.get("percent")
                    det = live.get("detail", "")

                    step_info = f" [{s}/{ts}]" if s and ts else ""
                    if pct is not None:
                        bar_len = 10
                        filled = int(bar_len * pct / 100)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        detail = f"  {stage}{step_info} {bar} {pct}%"
                        if det:
                            detail += f"  {det}"
                    else:
                        # Indeterminate: show stage name only, no progress bar
                        detail = f"  {stage}{step_info}"
                        if det:
                            detail += f"  {det}"

                lines.append(f" [>] {step:<20s} {format_elapsed(elapsed)}{detail}")

            elif status == "failed":
                elapsed = info.get("elapsed", 0)
                error = info.get("error", "")
                lines.append(f" [X] {step:<20s} {format_elapsed(elapsed)}  FAILED")
                if error:
                    lines.append(f"     Error: {error[:80]}")

            else:
                lines.append(f" [ ] {step}")

        lines.append("━" * 66)

        sys.stderr.write("\033[2K")
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()


# ─── Docker Runner ────────────────────────────────────────────

def run_docker(cmd, step_name, state, output_dir, dry_run=False):
    """Run a Docker container, tee output to terminal + log file."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{step_name.lower()}.log"

    print(f"\n{'='*60}")
    print(f"[{step_name}] Starting...")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}\n")

    if dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        state.complete_step(step_name)
        return 0

    dashboard = ProgressDashboard(state, output_dir)
    dashboard.start(step_name, resumed=False)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    last_lines = []
    with open(log_file, "w", encoding="utf-8") as lf:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
            last_lines.append(line.rstrip())
            if len(last_lines) > 20:
                last_lines.pop(0)

    proc.wait()
    dashboard.stop()

    if proc.returncode != 0:
        substep_info = {}
        progress_file = Path(output_dir) / ".progress.json"
        try:
            if progress_file.exists():
                with open(progress_file, encoding="utf-8") as f:
                    pdata = json.load(f)
                substep_info = {
                    "substep": pdata.get("stage"),
                    "substep_num": pdata.get("step"),
                    "substep_total": pdata.get("total_steps"),
                }
        except Exception:
            pass

        error_msg = "\n".join(last_lines[-5:])
        state.fail_step(step_name, error_msg, substep_info)

        elapsed = state.step_info(step_name).get("elapsed", 0)
        print(f"\n{'='*60}")
        print(f"[FAILED] {step_name} failed after {format_elapsed(elapsed)}")
        print(f"  Last output:")
        for line in last_lines[-20:]:
            print(f"    {line}")
        print(f"  Full log: {log_file}")
        print(f"{'='*60}")
        sys.exit(proc.returncode)

    state.complete_step(step_name)
    elapsed = state.step_info(step_name).get("elapsed", 0)
    print(f"\n[SUCCESS] {step_name} completed in {format_elapsed(elapsed)}")
    return 0


# ─── Pipeline Steps ──────────────────────────────────────────

def step_ingest(config, state, output_dir, dry_run=False):
    """Step 1: Extract frames from video.

    The ingest wrapper handles:
    - 360 auto-detection (from metadata + aspect ratio)
    - Interval calculation (from target image count + video duration)
    - Writes .video_type file for orchestrator to read back
    """
    container = "onyx-ingest"
    state.start_step("INGEST", container)

    target = TARGET_IMAGES.get(
        (config["scene"], config["quality"]), 300
    )

    # Resolve the real video path. input_video.mp4 in output_dir is an absolute
    # symlink that breaks inside the container (target path not in mount ns).
    # Mount the actual file at a separate /video/ path to avoid colliding with
    # the existing symlink already present under /data.
    real_video = Path(config["input"]).resolve()

    cmd = [
        "docker", "run", "--rm",
        *uid_gid_flags(),
        "-v", f"{docker_path(output_dir)}:/data",
        "-v", f"{docker_path(real_video)}:/video/input.mp4:ro",
        image_name(config.get("local", False), container),
        "--input", "/video/input.mp4",
        "--output", "/data/images",
        "--target-images", str(target),
        "--video-type", config.get("video", "auto"),
    ]

    # Pass through interval override if user specified one
    if config.get("interval_override"):
        cmd.extend(["--interval", str(config["interval_override"])])

    run_docker(cmd, "INGEST", state, output_dir, dry_run)

    # Read back detected video type from container
    video_type_file = Path(output_dir) / ".video_type"
    if video_type_file.exists():
        detected = video_type_file.read_text(encoding="utf-8").strip()
        if detected in ("360", "normal"):
            state.data["base_config"]["video"] = detected
            state.save()
            print(f"[DETECTED] Video type: {detected}")
    elif config.get("video", "auto") == "auto":
        # Container didn't write type file, default to normal
        state.data["base_config"]["video"] = "normal"
        state.save()


def _ensure_images_flat(output_dir):
    """Flatten *_processed/ subdirs in images/ before SFM runs.

    The 360Extractor writes to {images}/{video}_processed/ instead of {images}/
    directly. InstantSfM expects images directly in images/. This handles cases
    where the ingest container already completed (and its flatten ran), but also
    guards against resumed runs where INGEST was skipped and old structure remains.
    """
    images_dir = Path(output_dir) / "images"
    if not images_dir.exists():
        return

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    direct = [f for f in images_dir.iterdir()
              if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if direct:
        return  # Already flat — nothing to do

    processed = [d for d in images_dir.iterdir()
                 if d.is_dir() and d.name.endswith("_processed")]
    if not processed:
        return

    print(f"[PREFLIGHT] Flattening _processed/ subdirectory in images/...")
    for subdir in processed:
        for f in subdir.iterdir():
            shutil.move(str(f), str(images_dir / f.name))
        try:
            subdir.rmdir()
        except OSError:
            pass
    moved = len([f for f in images_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
    print(f"[PREFLIGHT] {moved} images ready in images/")


def step_sfm(config, state, output_dir, dry_run=False):
    """Step 2: Structure from Motion."""
    container = "onyx-instantsfm"
    state.start_step("SFM", container)

    # Guard: flatten _processed/ subdir if ingest left old structure
    if not dry_run:
        _ensure_images_flat(output_dir)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        *uid_gid_flags(),
        "-e", "HOME=/tmp",
        "-v", f"{docker_path(output_dir)}:/data",
        image_name(config.get("local", False), container),
        "--data_path", "/data",
    ]

    run_docker(cmd, "SFM", state, output_dir, dry_run)


def step_reconstruction(config, state, output_dir, dry_run=False):
    """Step 3: 3D Reconstruction (branches on mode x quality)."""
    mode = config["mode"]
    quality = config["quality"]

    container = RECONSTRUCTION_CONTAINER[(mode, quality)]
    state.start_step("RECONSTRUCTION", container)

    if mode == "gaussian" and quality == "production":
        _run_milo(config, state, output_dir, dry_run)
    elif mode == "gaussian" and quality == "proto":
        _run_gsplat(config, state, output_dir, dry_run)
    elif mode == "gaussian" and quality == "yono":
        _run_yonosplat(config, state, output_dir, dry_run)
    elif mode == "photogrammetry":
        _run_openmvs(config, state, output_dir, dry_run)


def _run_milo(config, state, output_dir, dry_run):
    scene = config["scene"]
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        *uid_gid_flags(),
        "-e", "HOME=/tmp",
        "-v", f"{docker_path(output_dir)}:/data",
        image_name(config.get("local", False), "onyx-milo"),
        "--scene", "/data",
        "--metric", scene,
        "--rasterizer", "radegs",
        "--extract-mesh",
        # indoor: verylowres reduces SDF mesh complexity during training,
        # which avoids nvdiffrast CUDA error 700 on some GPUs/drivers
        *(["--mesh_config", "verylowres"] if scene == "indoor" else []),
    ]

    run_docker(cmd, "RECONSTRUCTION", state, output_dir, dry_run)


def _run_gsplat(config, state, output_dir, dry_run):
    scene = config["scene"]
    resolution = "4" if scene == "outdoor" else "2"
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        *uid_gid_flags(),
        "-e", "HOME=/tmp",
        "-v", f"{docker_path(output_dir)}:/data",
        image_name(config.get("local", False), "onyx-gsplat"),
        "--scene", "/data",
        "--iterations", "7000",
        "--resolution", resolution,
    ]

    run_docker(cmd, "RECONSTRUCTION", state, output_dir, dry_run)


def _run_openmvs(config, state, output_dir, dry_run):
    quality = config["quality"]
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{docker_path(output_dir)}:/data",
        image_name(config.get("local", False), "onyx-openmvs"),
        "--data_path", "/data",
    ]
    if quality == "proto":
        cmd.extend(["--decimate", "0.2"])

    run_docker(cmd, "RECONSTRUCTION", state, output_dir, dry_run)


def _run_yonosplat(config, state, output_dir, dry_run):
    """Run YonoSplat pose-free feed-forward Gaussian Splatting (no SFM input needed)."""
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        *uid_gid_flags(),
        "-e", "HOME=/tmp",
        "-e", "USER=onyx",  # getpass.getuser() needs USER when uid has no /etc/passwd entry
        "-v", f"{docker_path(output_dir)}:/data",
        image_name(config.get("local", False), "onyx-yonosplat"),
        "--scene",      "/data/images",
        "--output",     "/data/output/yonosplat",
        "--scene-type", config["scene"],
        "--quality",    config["quality"],
    ]
    run_docker(cmd, "RECONSTRUCTION", state, output_dir, dry_run)


def step_segformer(config, state, output_dir, dry_run=False):
    """Step 4: Segformer masking + optional Gaussian filtering."""
    container = "onyx-segformer"
    state.start_step("SEGFORMER", container)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        *uid_gid_flags(),
        "-e", "HOME=/tmp",
        "-v", f"{docker_path(output_dir)}:/data",
        image_name(config.get("local", False), container),
        "--input", "/data/images",
        "--output", "/data/masks",
        "--classes", "sky,water,sea,person,boat",
    ]

    mode = config["mode"]
    quality = config["quality"]
    if mode == "gaussian":
        if quality == "production":
            ply = "/data/output/milo/point_cloud/iteration_18000/point_cloud.ply"
        elif quality == "yono":
            ply = "/data/output/yonosplat/splat.ply"
        else:
            ply = "/data/output/splatfacto/ply/splat.ply"
        cmd.extend([
            "--ply", ply,
            "--cameras", "/data/sparse/0",
        ])

    run_docker(cmd, "SEGFORMER", state, output_dir, dry_run)


# ─── Purge ────────────────────────────────────────────────────

def _rmtree(path):
    """shutil.rmtree wrapper that handles read-only files on Windows."""
    if _WINDOWS:
        import stat
        def _remove_readonly(func, fpath, _):
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        shutil.rmtree(path, onerror=_remove_readonly)
    else:
        shutil.rmtree(path)


def do_purge(workdir, keep_output=False):
    """Delete working directory data."""
    workdir = Path(workdir).resolve()
    if not workdir.exists():
        print(f"Error: Directory not found: {workdir}")
        sys.exit(1)

    state_file = workdir / "pipeline_state.json"
    if not state_file.exists():
        print(f"Error: Not an Onyx working directory (no pipeline_state.json)")
        sys.exit(1)

    state = PipelineState.load(workdir)
    run_id = state.run_id

    if keep_output:
        print(f"Purging {workdir} (keeping output/)...")
        for item in workdir.iterdir():
            if item.name == "output":
                continue
            if item.is_dir():
                _rmtree(item)
            else:
                item.unlink()
        print(f"[DONE] Purged {run_id} — output/ preserved")
    else:
        print(f"Purging {workdir}...")
        _rmtree(workdir)
        print(f"[DONE] Purged {run_id} — all data deleted")


def do_purge_all(scan_dir, keep_output=False):
    """Purge ALL Onyx batches found in the given directory."""
    scan_dir = Path(scan_dir).resolve()
    if not scan_dir.exists():
        print(f"Error: Directory not found: {scan_dir}")
        sys.exit(1)

    # Find all batches (subdirs with pipeline_state.json)
    batches = []
    for child in sorted(scan_dir.iterdir()):
        if child.is_dir() and (child / "pipeline_state.json").exists():
            batches.append(child)

    if not batches:
        print(f"No batches found in {scan_dir}")
        return

    print(f"Found {len(batches)} batch(es) to purge:")
    for b in batches:
        try:
            with open(b / "pipeline_state.json", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  {data.get('id', '?'):<28s} {b.name}")
        except Exception:
            print(f"  {'?':<28s} {b.name}")

    # Confirmation prompt
    action = "purge (keep output)" if keep_output else "DELETE"
    resp = input(f"\n{action} all {len(batches)} batches? [y/N] ")
    if resp.lower() != "y":
        print("Aborted")
        return

    for b in batches:
        try:
            with open(b / "pipeline_state.json", encoding="utf-8") as f:
                run_id = json.load(f).get("id", "?")
        except Exception:
            run_id = "?"
        do_purge(b, keep_output)

    print(f"\n[DONE] Purged {len(batches)} batches")


# ─── Fork ─────────────────────────────────────────────────────

def do_fork(source_dir, mode, quality, dry_run=False):
    """Run different reconstruction on same data.

    Appends a new run to the state — preserves all previous run history.
    Same batch ID, same working directory.
    """
    source_dir = Path(source_dir).resolve()
    state = PipelineState.load(source_dir)

    # Verify shared steps completed
    for step in SHARED_STEPS:
        status = state.step_status(step)
        if status not in ("completed", "skipped"):
            print(f"Error: Shared step {step} is {status}. "
                  f"Run the pipeline first or use --resume.")
            sys.exit(1)

    # Verify images and sparse exist
    if not (source_dir / "images").exists():
        print(f"Error: No images/ found in {source_dir}")
        sys.exit(1)
    if not (source_dir / "sparse" / "0").exists():
        print(f"Error: No sparse/0/ found in {source_dir}")
        sys.exit(1)

    # Determine new mode/quality
    prev_run = state.active_run
    new_mode = mode or prev_run["mode"]
    new_quality = quality or prev_run["quality"]

    if new_mode == prev_run["mode"] and new_quality == prev_run["quality"]:
        print(f"Error: --fork requires different --mode and/or --quality")
        print(f"  Current: {prev_run['mode']}/{prev_run['quality']}")
        sys.exit(1)

    # Check for duplicate — don't re-run an existing combination
    for i, run in enumerate(state.data["runs"]):
        if run["mode"] == new_mode and run["quality"] == new_quality:
            recon_status = run["steps"]["RECONSTRUCTION"]["status"]
            if recon_status == "completed":
                print(f"Error: {new_mode}/{new_quality} already completed (run {i + 1})")
                sys.exit(1)
            elif recon_status in ("failed", "running"):
                print(f"Resuming existing {new_mode}/{new_quality} run (run {i + 1})")
                state.set_active_run(i)
                # Reset failed steps
                for s in RUN_STEPS:
                    if s in run["steps"] and run["steps"][s]["status"] in ("failed", "running"):
                        run["steps"][s] = {"status": "pending"}
                state.save()
                break
    else:
        # New combination — add a new run
        state.add_run(new_mode, new_quality)

    config = state.full_config(local=False)
    run = state.active_run

    print(f"Fork [{state.run_id}]: adding run {state.num_runs()} "
          f"({run['mode']}/{run['quality']})")
    print(f"  Previous runs:")
    for i, r in enumerate(state.data["runs"]):
        if i == state._active_run_idx:
            continue
        recon = r["steps"]["RECONSTRUCTION"].get("status", "pending")
        print(f"    Run {i + 1}: {r['mode']}/{r['quality']} — {recon}")

    # Run per-run steps
    for step in RUN_STEPS:
        if step not in state.active_run["steps"]:
            continue
        status = state.step_status(step)
        if status == "completed":
            elapsed = state.step_info(step).get("elapsed", 0)
            print(f"[SKIP] {step} — already completed ({format_elapsed(elapsed)})")
            continue
        if status == "skipped":
            reason = state.step_info(step).get("reason", "")
            print(f"[SKIP] {step} — {reason}")
            continue

        if step == "RECONSTRUCTION":
            step_reconstruction(config, state, source_dir, dry_run)
        elif step == "SEGFORMER":
            step_segformer(config, state, source_dir, dry_run)

    print_summary(state, source_dir)


# ─── Export / Import ──────────────────────────────────────────

EXPORT_DIRS = ["images", "sparse", "output", "masks", "logs"]
EXPORT_FILES = ["pipeline_state.json"]
SKIP_FILES = {".progress.json", ".progress.json.tmp",
              "pipeline_state.json.tmp"}


def do_list(scan_dir):
    """List all Onyx batches found in the given directory.

    Scans immediate subdirectories for pipeline_state.json files.
    Also checks if scan_dir itself is a batch.
    """
    scan_dir = Path(scan_dir).resolve()
    if not scan_dir.exists():
        print(f"Error: Directory not found: {scan_dir}")
        sys.exit(1)

    batches = []

    # Check if scan_dir itself is a batch
    state_file = scan_dir / "pipeline_state.json"
    if state_file.exists():
        batches.append((scan_dir, state_file))

    # Scan immediate subdirectories
    if scan_dir.is_dir():
        for child in sorted(scan_dir.iterdir()):
            if child.is_dir():
                sf = child / "pipeline_state.json"
                if sf.exists() and sf != state_file:
                    batches.append((child, sf))

    if not batches:
        print(f"No batches found in {scan_dir}")
        return

    print(f"{'ID':<28s} {'Hash':<10s} {'Scene':<9s} {'Runs':<6s} {'Status':<12s} Path")
    print("─" * 100)

    for workdir, sf in batches:
        try:
            with open(sf, encoding="utf-8") as f:
                data = json.load(f)
            run_id = data.get("id", "?")
            state_hash = data.get("state_hash", "?")
            base = data.get("base_config", {})
            scene = base.get("scene", "?")
            runs = data.get("runs", [])
            num_runs = len(runs)

            # Determine overall status from latest run
            all_done = True
            has_failed = False
            for run in runs:
                for step in RUN_STEPS:
                    status = run.get("steps", {}).get(step, {}).get("status", "pending")
                    if status == "failed":
                        has_failed = True
                    if status not in ("completed", "skipped"):
                        all_done = False
            # Also check shared steps
            for step in SHARED_STEPS:
                status = data.get("shared_steps", {}).get(step, {}).get("status", "pending")
                if status == "failed":
                    has_failed = True
                if status not in ("completed", "skipped"):
                    all_done = False

            if has_failed:
                overall = "FAILED"
            elif all_done:
                overall = "completed"
            else:
                overall = "in progress"

            # Run details
            run_labels = []
            for r in runs:
                m = r.get("mode", "?")[:4]
                q = r.get("quality", "?")[:4]
                run_labels.append(f"{m}/{q}")

            rel = workdir.relative_to(scan_dir) if workdir != scan_dir else Path(".")
            print(f"{run_id:<28s} {state_hash:<10s} {scene:<9s} "
                  f"{num_runs:<6d} {overall:<12s} {rel}")
            for i, r in enumerate(runs):
                m = r.get("mode", "?")
                q = r.get("quality", "?")
                recon = r.get("steps", {}).get("RECONSTRUCTION", {}).get("status", "pending")
                print(f"  run {i+1}: {m}/{q} — {recon}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"{'?':<28s} {'?':<10s} {'?':<9s} {'?':<6s} {'error':<12s} "
                  f"{workdir.relative_to(scan_dir)}")


def do_export(workdir, zip_path=None):
    """Export entire batch as a zip file.

    Includes: pipeline_state.json, input video, images/, sparse/, output/,
    masks/, logs/. Resolves symlinks (e.g., input_video.mp4 -> actual file).
    """
    workdir = Path(workdir).resolve()
    if not workdir.exists():
        print(f"Error: Directory not found: {workdir}")
        sys.exit(1)

    state_file = workdir / "pipeline_state.json"
    if not state_file.exists():
        print(f"Error: Not an Onyx working directory (no pipeline_state.json)")
        sys.exit(1)

    state = PipelineState.load(workdir)
    run_id = state.run_id

    if zip_path is None:
        zip_path = Path.cwd() / f"{run_id}.zip"
    else:
        zip_path = Path(zip_path).resolve()

    # Collect files to export
    files_to_add = []  # (absolute_path, archive_name)

    # Top-level files
    for name in EXPORT_FILES:
        p = workdir / name
        if p.exists():
            files_to_add.append((p, name))

    # Input video (resolve symlink — always included)
    video = workdir / "input_video.mp4"
    if video.exists():
        real = video.resolve()
        files_to_add.append((real, "input_video.mp4"))

    # Directories
    for dirname in EXPORT_DIRS:
        d = workdir / dirname
        if d.exists():
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.name not in SKIP_FILES:
                    rel = f.relative_to(workdir)
                    files_to_add.append((f, str(rel)))

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f, _ in files_to_add)
    total_mb = total_bytes / (1024 * 1024)
    print(f"Exporting [{run_id}] (hash: {state.state_hash}) — "
          f"{len(files_to_add)} files, {total_mb:.1f} MB")

    # Create zip (ZIP_STORED — content already compressed: JPGs, PLYs)
    written = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for abs_path, arc_name in files_to_add:
            # Use ZIP_DEFLATED for small text files (JSON, logs)
            compress = zipfile.ZIP_DEFLATED if abs_path.suffix in (
                ".json", ".log", ".txt", ".bin"
            ) else zipfile.ZIP_STORED

            zf.write(abs_path, arc_name, compress_type=compress)
            written += abs_path.stat().st_size
            pct = int(100 * written / total_bytes) if total_bytes else 100
            sys.stderr.write(f"\r  Packing... {pct}%")
            sys.stderr.flush()

    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    sys.stderr.write("\r" + " " * 40 + "\r")
    print(f"[DONE] Exported to {zip_path} ({zip_mb:.1f} MB)")
    print(f"  Contains {len(files_to_add)} files")


def do_import(zip_path, output_dir=None):
    """Import a batch from a zip file.

    Extracts to the specified output directory or uses the run ID from state.
    Validates that the zip contains a valid pipeline_state.json.
    """
    zip_path = Path(zip_path).resolve()
    if not zip_path.exists():
        print(f"Error: File not found: {zip_path}")
        sys.exit(1)

    if not zipfile.is_zipfile(zip_path):
        print(f"Error: Not a valid zip file: {zip_path}")
        sys.exit(1)

    # Read state from zip to get run ID
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if "pipeline_state.json" not in names:
            print(f"Error: Not an Onyx export (no pipeline_state.json in zip)")
            sys.exit(1)

        with zf.open("pipeline_state.json") as f:
            state_data = json.load(f)
        run_id = state_data.get("id", "unknown")
        import_hash = state_data.get("state_hash", "unknown")

        # Determine output directory
        if output_dir is None:
            output_dir = Path.cwd() / run_id
        else:
            output_dir = Path(output_dir).resolve()

        # Check for existing batch at target
        existing_state = output_dir / "pipeline_state.json"
        if existing_state.exists():
            with open(existing_state, encoding="utf-8") as ef:
                existing = json.load(ef)
            existing_id = existing.get("id", "unknown")
            existing_hash = existing.get("state_hash", "unknown")
            if existing_id == run_id:
                if existing_hash == import_hash:
                    print(f"Error: Batch [{run_id}] already exists at {output_dir}")
                    print(f"  State hash: {existing_hash} (identical)")
                else:
                    print(f"Error: Batch [{run_id}] already exists at {output_dir}")
                    print(f"  Local hash:  {existing_hash}")
                    print(f"  Import hash: {import_hash}")
                    print(f"  States differ — use --purge first if you want to replace it")
            else:
                print(f"Error: Target directory contains a different batch [{existing_id}]")
                print(f"  Use --output to specify a different path")
            sys.exit(1)

        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Error: Target directory not empty: {output_dir}")
            print(f"  Use --output to specify a different path")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        total = len(names)
        print(f"Importing [{run_id}] — {total} files to {output_dir}")

        for i, name in enumerate(names):
            zf.extract(name, output_dir)
            pct = int(100 * (i + 1) / total) if total else 100
            sys.stderr.write(f"\r  Extracting... {pct}%")
            sys.stderr.flush()

    sys.stderr.write("\r" + " " * 40 + "\r")

    # Update state with new paths
    state = PipelineState.load(output_dir)
    video_path = output_dir / "input_video.mp4"
    if video_path.exists():
        state.data["base_config"]["input"] = str(video_path)
    state.save()

    # Print summary
    print(f"[DONE] Imported [{run_id}] (hash: {state.state_hash}) to {output_dir}")

    runs = state.data.get("runs", [])
    for i, run in enumerate(runs):
        label = f"{run['mode']}/{run['quality']}"
        recon = run["steps"].get("RECONSTRUCTION", {}).get("status", "pending")
        print(f"  Run {i + 1}: {label} — reconstruction: {recon}")

    shared = state.data.get("shared_steps", {})
    ingest = shared.get("INGEST", {}).get("status", "pending")
    sfm = shared.get("SFM", {}).get("status", "pending")
    print(f"  Shared steps: INGEST={ingest}, SFM={sfm}")
    print(f"\n  To resume: python run_pipeline.py --resume {output_dir}")


# ─── Summary ─────────────────────────────────────────────────

def print_summary(state, output_dir):
    """Print final pipeline summary."""
    print(f"\n{'='*60}")
    print(f"ONYX PIPELINE [{state.run_id}] (hash: {state.state_hash}) — COMPLETE")
    print(f"{'='*60}")

    # Shared steps
    total_time = 0
    for step in SHARED_STEPS:
        info = state.step_info(step)
        status = info.get("status", "pending")
        elapsed = info.get("elapsed", 0)
        total_time += elapsed
        if status == "completed":
            print(f"  [+] {step:<20s} {format_elapsed(elapsed)}")
        elif status == "skipped":
            reason = info.get("reason", "")
            print(f"  [-] {step:<20s} skipped{' — ' + reason if reason else ''}")

    # All runs
    for i, run in enumerate(state.data["runs"]):
        run_label = f"{run['mode']}/{run['quality']}"
        print(f"\n  Run {i + 1}: {run_label}")
        for step in RUN_STEPS:
            info = run["steps"].get(step, {})
            status = info.get("status", "pending")
            elapsed = info.get("elapsed", 0)
            total_time += elapsed
            if status == "completed":
                print(f"    [+] {step:<18s} {format_elapsed(elapsed)}")
            elif status == "skipped":
                reason = info.get("reason", "")
                print(f"    [-] {step:<18s} skipped{' — ' + reason if reason else ''}")
            elif status == "failed":
                print(f"    [X] {step:<18s} FAILED")
            else:
                print(f"    [ ] {step}")

    print(f"\n  Total: {format_elapsed(total_time)}")
    print(f"  Output: {output_dir}")

    # List key output files
    output_path = Path(output_dir) / "output"
    if output_path.exists():
        for f in sorted(output_path.rglob("*.ply")):
            size_mb = f.stat().st_size / (1024 * 1024)
            rel = f.relative_to(output_dir)
            print(f"  {rel}: {size_mb:.1f} MB")

    print(f"{'='*60}")


# ─── Main ────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Onyx 3D Reconstruction Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # New run arguments
    parser.add_argument("--input", "-i", help="Input video file path")
    parser.add_argument("--output", "-o", help="Working directory (created if missing)")
    parser.add_argument("--mode", choices=["gaussian", "photogrammetry"],
                        help="Reconstruction mode")
    parser.add_argument("--scene", choices=["indoor", "outdoor"],
                        help="Scene type")
    parser.add_argument("--quality", choices=["proto", "production", "yono"],
                        help="Quality level (yono = pose-free feed-forward, no SFM)")
    parser.add_argument("--video", choices=["normal", "360", "auto"],
                        default="auto",
                        help="Video type (default: auto-detect)")

    # Resume
    parser.add_argument("--resume", metavar="WORKDIR",
                        help="Resume from existing working directory")
    parser.add_argument("--from-step", choices=ALL_STEPS,
                        help="Force re-run from this step (use with --resume)")

    # Fork
    parser.add_argument("--fork", metavar="WORKDIR",
                        help="Run different reconstruction on same data")

    # Purge
    parser.add_argument("--purge", metavar="WORKDIR", nargs="?", const=".",
                        help="Delete working directory data")
    parser.add_argument("--purge-all", metavar="DIR", nargs="?", const=".",
                        help="Purge ALL batches in directory (default: current dir)")
    parser.add_argument("--keep-output", action="store_true",
                        help="With --purge/--purge-all: keep output/ directories")

    # Export / Import / List
    parser.add_argument("--export", metavar="WORKDIR",
                        help="Export batch as zip file")
    parser.add_argument("--import-batch", metavar="ZIPFILE", dest="import_batch",
                        help="Import batch from zip file")
    parser.add_argument("--list", metavar="DIR", nargs="?", const=".",
                        help="List batches in directory (default: current dir)")

    # Install
    parser.add_argument("--install", action="store_true",
                        help="Pull all pipeline container images from ghcr.io")

    # Optional overrides
    parser.add_argument("--interval", type=float,
                        help="Override frame extraction interval (seconds)")
    parser.add_argument("--local", action="store_true",
                        help="Use local Docker images instead of ghcr.io")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    # Skip shortcuts
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingest (use existing images/)")
    parser.add_argument("--skip-sfm", action="store_true",
                        help="Skip SfM (use existing sparse/)")

    # Opt-in steps
    parser.add_argument("--segment", action="store_true",
                        help="Run Segformer masking after reconstruction")

    return parser.parse_args()


def main():
    _setup_windows_console()
    args = parse_args()

    # Handle SIGINT gracefully
    def handle_sigint(sig, frame):
        print("\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(130)
    signal.signal(signal.SIGINT, handle_sigint)

    # ── Install mode ────────────────────────────────────────────
    if args.install:
        do_install(local=args.local)
        return

    # ── List mode ─────────────────────────────────────────────
    if args.list is not None:
        do_list(args.list)
        return

    # ── Purge mode ────────────────────────────────────────────
    if args.purge is not None:
        do_purge(args.purge, args.keep_output)
        return

    if args.purge_all is not None:
        do_purge_all(args.purge_all, args.keep_output)
        return

    # ── Export mode ───────────────────────────────────────────
    if args.export:
        zip_out = None
        if args.output:
            zip_out = args.output
        do_export(args.export, zip_path=zip_out)
        return

    # ── Import mode ───────────────────────────────────────────
    if args.import_batch:
        target = args.output if args.output else None
        do_import(args.import_batch, output_dir=target)
        return

    # ── Fork mode ─────────────────────────────────────────────
    if args.fork:
        do_fork(args.fork, args.mode, args.quality, args.dry_run)
        return

    # ── Resume mode ───────────────────────────────────────────
    if args.resume:
        output_dir = Path(args.resume).resolve()
        state = PipelineState.load(output_dir)
        config = state.full_config(local=args.local)
        if args.interval:
            config["interval_override"] = args.interval

        if args.from_step:
            state.reset_from(args.from_step)

        print(f"Resuming pipeline [{state.run_id}] from {output_dir}")
        video_label = config.get('video', 'auto')
        print(f"Config: mode={config['mode']} scene={config['scene']} "
              f"quality={config['quality']} video={video_label}")

    # ── New run mode ──────────────────────────────────────────
    else:
        missing = []
        for arg in ("input", "output", "mode", "scene", "quality"):
            if getattr(args, arg) is None:
                missing.append(f"--{arg}")
        if missing:
            print(f"Error: Required arguments missing: {', '.join(missing)}")
            print("For a new run, all 5 parameters are required:")
            print("  --input, --output, --mode, --scene, --quality")
            print("  (--video defaults to auto-detect; override with --video 360 or --video normal)")
            print("To resume: python run_pipeline.py --resume ./workdir")
            sys.exit(1)

        input_path = Path(args.input).resolve()
        if not input_path.exists():
            print(f"Error: Input video not found: {input_path}")
            sys.exit(1)

        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "mode": args.mode,
            "scene": args.scene,
            "quality": args.quality,
            "video": args.video,
            "input": str(input_path),
            "local": args.local,
            "segment": args.segment,
        }
        if args.interval:
            config["interval_override"] = args.interval

        state = PipelineState.create(output_dir, config)

        video_dst = output_dir / "input_video.mp4"
        if not video_dst.exists():
            if _WINDOWS:
                # Symlinks require admin/Developer Mode on Windows — copy instead
                shutil.copy2(str(input_path), str(video_dst))
            else:
                try:
                    os.symlink(input_path, video_dst)
                except OSError:
                    shutil.copy2(str(input_path), str(video_dst))

        print(f"Starting pipeline [{state.run_id}]")
        video_label = config.get('video', 'auto')
        print(f"Config: mode={config['mode']} scene={config['scene']} "
              f"quality={config['quality']} video={video_label}")
        print(f"Output: {output_dir}")

    # ── Apply skip shortcuts ──────────────────────────────────
    if args.skip_ingest:
        if state.step_status("INGEST") != "completed":
            state.skip_step("INGEST", "user requested --skip-ingest")
    if args.skip_sfm:
        if state.step_status("SFM") != "completed":
            state.skip_step("SFM", "user requested --skip-sfm")

    # ── Pre-run: verify Docker images installed ────────────────
    if not args.dry_run:
        ok, missing = check_images_installed(local=args.local or config.get("local", False))
        if not ok:
            print(f"\nError: {len(missing)} container image(s) not found:")
            for name in missing:
                img = image_name(args.local or config.get("local", False), name)
                print(f"  - {img}")
            print(f"\nRun 'python run_pipeline.py --install' first to pull all images.")
            sys.exit(1)

    # ── Run all steps ─────────────────────────────────────────
    step_funcs = {
        "INGEST": step_ingest,
        "SFM": step_sfm,
        "RECONSTRUCTION": step_reconstruction,
        "SEGFORMER": step_segformer,
    }

    for step in state.all_steps:
        status = state.step_status(step)

        if status == "completed":
            elapsed = state.step_info(step).get("elapsed", 0)
            print(f"[SKIP] {step} — already completed ({format_elapsed(elapsed)})")
            continue

        if status == "skipped":
            reason = state.step_info(step).get("reason", "")
            print(f"[SKIP] {step} — {reason}")
            continue

        step_funcs[step](config, state, output_dir, args.dry_run)

    # ── Summary ───────────────────────────────────────────────
    print_summary(state, output_dir)


if __name__ == "__main__":
    main()
