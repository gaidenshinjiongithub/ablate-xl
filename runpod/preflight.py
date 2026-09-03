#!/usr/bin/env python3
"""Fail-fast hardware and storage checks for paid RunPod runs."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


GIB = 1024 ** 3


def _gib(value: int) -> float:
    return value / GIB


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-total-vram-gib", type=float, default=0)
    parser.add_argument("--min-disk-gib", type=float, default=0)
    parser.add_argument("--workspace", default="/workspace")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    failures: list[str] = []

    # Ablate XL currently uses one Python process plus Transformers device_map.
    # Starting it on every Instant Cluster node would load a full copy per node.
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    if num_nodes != 1:
        failures.append(
            "NUM_NODES is greater than 1. The current runtime is single-host; "
            "use one multi-GPU Pod, not an Instant Cluster."
        )

    gpu_count = torch.cuda.device_count()
    total_vram = 0
    gpu_rows = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        total_vram += props.total_memory
        gpu_rows.append(f"cuda:{index} {props.name} {_gib(props.total_memory):.1f} GiB")

    if not gpu_count:
        failures.append("No CUDA GPUs are visible inside the Pod.")
    if _gib(total_vram) < args.min_total_vram_gib:
        failures.append(
            f"Visible VRAM is {_gib(total_vram):.1f} GiB; "
            f"profile requires at least {args.min_total_vram_gib:.1f} GiB."
        )

    workspace = Path(args.workspace)
    if not workspace.exists():
        failures.append(f"Workspace path does not exist: {workspace}")
        disk_free = 0
    else:
        disk_free = shutil.disk_usage(workspace).free
        if not os.access(workspace, os.W_OK):
            failures.append(f"Workspace path is not writable: {workspace}")
        if _gib(disk_free) < args.min_disk_gib:
            failures.append(
                f"Free space under {workspace} is {_gib(disk_free):.1f} GiB; "
                f"profile requires at least {args.min_disk_gib:.1f} GiB."
            )

    print("Ablate XL RunPod preflight")
    print(f"  GPUs: {gpu_count}")
    for row in gpu_rows:
        print(f"  - {row}")
    print(f"  Aggregate VRAM: {_gib(total_vram):.1f} GiB")
    print(f"  Free workspace storage: {_gib(disk_free):.1f} GiB")
    print(f"  HF cache: {os.environ.get('HF_HOME', '(not set)')}")

    if failures:
        print("\nPreflight failed before model download:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
