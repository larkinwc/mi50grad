#!/usr/bin/env python3
"""
Build GCN assembly kernels (.s) into HSACO code objects.

Usage:
    python build_kernels.py [--src-dir DIR] [--out-dir DIR] [--mcpu gfx906]

Pipeline: .s -> llvm-mc -> .o -> ld.lld -> .hsaco
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


LLVM_MC = "/opt/rocm/llvm/bin/llvm-mc"
LD_LLD = "/opt/rocm/llvm/bin/ld.lld"
DEFAULT_MCPU = "gfx906"


def assemble(src: Path, obj: Path, mcpu: str) -> bool:
    """Assemble .s -> .o using llvm-mc."""
    cmd = [
        LLVM_MC,
        f"--triple=amdgcn-amd-amdhsa",
        f"--mcpu={mcpu}",
        "--filetype=obj",
        str(src),
        "-o", str(obj),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL assemble {src.name}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def link(obj: Path, hsaco: Path) -> bool:
    """Link .o -> .hsaco using ld.lld."""
    cmd = [
        LD_LLD,
        "--shared",
        str(obj),
        "-o", str(hsaco),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL link {obj.name}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def build_kernel(src: Path, out_dir: Path, mcpu: str) -> bool:
    """Full build pipeline for a single .s file."""
    stem = src.stem
    obj = out_dir / f"{stem}.o"
    hsaco = out_dir / f"{stem}.hsaco"

    print(f"  {src.name} -> {hsaco.name} ... ", end="", flush=True)

    if not assemble(src, obj, mcpu):
        print("FAIL (assemble)")
        return False

    if not link(obj, hsaco):
        print("FAIL (link)")
        return False

    # Clean up .o
    obj.unlink(missing_ok=True)

    print("OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build GCN assembly kernels")
    parser.add_argument("--src-dir", type=Path, default=None,
                        help="Directory containing .s files")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for .hsaco files (default: same as src)")
    parser.add_argument("--mcpu", default=DEFAULT_MCPU,
                        help=f"Target GPU (default: {DEFAULT_MCPU})")
    parser.add_argument("files", nargs="*", type=Path,
                        help="Specific .s files to build")
    args = parser.parse_args()

    # Collect source files
    sources = []
    if args.files:
        sources = args.files
    elif args.src_dir:
        sources = sorted(args.src_dir.glob("*.s"))
    else:
        # Default: build all .s files in current directory
        sources = sorted(Path(".").glob("*.s"))

    if not sources:
        print("No .s files found to build.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or (args.src_dir if args.src_dir else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {len(sources)} kernel(s) for {args.mcpu}")
    print(f"Output: {out_dir}")

    ok = 0
    fail = 0
    for src in sources:
        if build_kernel(src, out_dir, args.mcpu):
            ok += 1
        else:
            fail += 1

    print(f"\nResults: {ok} OK, {fail} FAILED")
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
