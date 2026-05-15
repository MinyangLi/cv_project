#!/usr/bin/env python3
"""
Rename ROSE outputs that were saved as example-1.mp4, example-2.mp4, … (processing order)
to the real Unedited basenames (example-10.mp4, …).

Old ROSE used a 1-based counter; bash glob order over example-*.mp4 follows directory
entry order (often NOT numeric). This script recovers the same order as:

  bash -c 'shopt -s nullglob; for f in DIR/example-*.mp4; do echo "$f"; done'

Then: predicted/rose/example-k.mp4  ->  basename of the k-th path in that list.

Usage:
  python re_index.py --unedited /path/to/Unedited --rose /path/to/predicted/rose
  python re_index.py ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def bash_glob_example_mp4_order(unedited_dir: str) -> list[str]:
    """Same iteration order as `for f in dir/example-*.mp4` in bash (nullglob)."""
    unedited_dir = os.path.abspath(unedited_dir)
    if not os.path.isdir(unedited_dir):
        raise FileNotFoundError(f"Not a directory: {unedited_dir}")

    script = r"""
set -e
shopt -s nullglob
for f in "$1"/example-*.mp4; do
  printf '%s\n' "$f"
done
"""
    r = subprocess.run(
        ["bash", "-c", script, "_", unedited_dir],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return lines


def validate_example_basename(name: str) -> bool:
    return bool(re.fullmatch(r"example-\d+\.mp4", name, re.IGNORECASE))


def main() -> int:
    p = argparse.ArgumentParser(description="Fix ROSE predicted filenames vs Unedited order.")
    p.add_argument("--unedited", required=True, help="Benchmark/.../Unedited (same as inference input)")
    p.add_argument("--rose", required=True, help="Benchmark/.../predicted/rose")
    p.add_argument("--dry-run", action="store_true", help="Print planned renames only")
    args = p.parse_args()

    rose_dir = os.path.abspath(args.rose)
    if not os.path.isdir(rose_dir):
        print(f"[ERROR] Not a directory: {rose_dir}", file=sys.stderr)
        return 1

    paths = bash_glob_example_mp4_order(args.unedited)
    basenames = [os.path.basename(x) for x in paths]
    for b in basenames:
        if not validate_example_basename(b):
            print(f"[ERROR] Unexpected Unedited name (need example-<digits>.mp4): {b}", file=sys.stderr)
            return 1

    n = len(basenames)
    if n == 0:
        print("[ERROR] No example-*.mp4 under Unedited (bash glob).", file=sys.stderr)
        return 1

    if len(set(basenames)) != n:
        print("[ERROR] Duplicate basenames in Unedited glob order (unexpected).", file=sys.stderr)
        return 1

    pid = os.getpid()
    for k in range(1, n + 1):
        wrong_path = os.path.join(rose_dir, f"example-{k}.mp4")
        if not os.path.isfile(wrong_path):
            print(
                f"[ERROR] Missing {wrong_path} (need example-1.mp4 … example-{n}.mp4 from old ROSE).",
                file=sys.stderr,
            )
            return 1

    print(f"[INFO] Unedited bash-glob order ({n} files) -> rose example-{{1..{n}}}.mp4")
    for k in range(1, n + 1):
        print(f"  example-{k}.mp4  ==>  {basenames[k - 1]}")

    if all(f"example-{k}.mp4" == basenames[k - 1] for k in range(1, n + 1)):
        print("[OK] Nothing to do (example-k.mp4 already matches targets).")
        return 0

    if args.dry_run:
        print("[DRY-RUN] No files renamed.")
        return 0

    tmps: list[str] = []
    for k in range(1, n + 1):
        tmp = os.path.join(rose_dir, f".reindex_tmp_{k}_{pid}.mp4")
        tmps.append(tmp)

    for k in range(1, n + 1):
        os.rename(os.path.join(rose_dir, f"example-{k}.mp4"), tmps[k - 1])

    for k in range(1, n + 1):
        os.rename(tmps[k - 1], os.path.join(rose_dir, basenames[k - 1]))

    print(f"[OK] Reassigned {n} file(s) under {rose_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
