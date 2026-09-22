"""Advisory process lock shared by new MVCT launchers; no frozen code changes."""
import json
import os
from pathlib import Path


SCRIPT_NAMES = {"run_repeated_validation.py", "run_checkpoint_selection.py",
                "run_missing_evidence.py", "main_cat.py", "run_ablation_cat.py",
                "run_weight_sensitivity.py", "run_repro_check.py"}


class RuntimeLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        # msvcrt locks a byte; keep the lock file, never unlink it on release.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"Another guarded job is running; lock: {self.path}") from exc
        self.handle = handle
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "cwd": os.getcwd()}).encode())
        handle.flush()
        return self

    def subprocess_options(self):
        # On Linux, children keep the flock even if the launcher is killed.
        return {"pass_fds": (self.handle.fileno(),)} if os.name != "nt" else {}

    def __exit__(self, *args):
        # Close rather than explicitly unlock: an inherited child FD may live on.
        self.handle.close()
        self.handle = None


def refuse_unmanaged_jobs():
    """Best-effort Linux scan: old direct commands do not honor our new lock."""
    if not Path("/proc").is_dir():
        return
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            names = {Path(arg.decode(errors="replace")).name for arg in args if arg}
            if names & SCRIPT_NAMES:
                found.append({"pid": int(entry.name), "scripts": sorted(names & SCRIPT_NAMES)})
        except (OSError, ValueError):
            continue
    if found:
        raise RuntimeError(f"Existing MVCT process detected; do not start another: {found}")


def default_lock_path():
    disk = Path("/root/autodl-tmp/MVCT_runtime")
    return disk / ".mvct_runtime.lock" if Path("/root/autodl-tmp").is_dir() else Path(__file__).resolve().parent / ".mvct_runtime.lock"
