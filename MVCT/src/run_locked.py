"""Use this launcher instead of directly starting frozen MVCT training scripts."""
import argparse
from pathlib import Path
import subprocess
import sys

from runtime_guard import RuntimeLock, default_lock_path, refuse_unmanaged_jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", choices=("run_checkpoint_selection.py", "run_repeated_validation.py"))
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    script = Path(__file__).resolve().parent / args.script
    if not script.is_file():
        raise FileNotFoundError(script)
    with RuntimeLock(default_lock_path()) as lock:
        refuse_unmanaged_jobs()
        print("Exclusive lock acquired; launching:", script, flush=True)
        child = subprocess.Popen([sys.executable, "-u", str(script), *args.script_args],
                                 cwd=str(script.parent), **lock.subprocess_options())
        try:
            return child.wait()
        except KeyboardInterrupt:
            # Do not kill an unknown PID or release protection around a live child.
            print("Interrupted launcher; waiting for its child to exit.", flush=True)
            return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
