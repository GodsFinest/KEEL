#!/usr/bin/env python3
"""Run one frozen KEEL runtime evaluation tier and preserve provenance."""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ALL_FAMILIES = [
    "chain", "diamond", "fanout_fanin", "shared_ancestor", "residual",
    "wide_independent",
    "matmul_heavy", "scratch_heavy", "transformer_like",
]
POLICIES = [
    "conservative", "greedy", "guarded_greedy", "prep_greedy",
    "limited_lookahead", "keel", "hybrid",
]
GRANT_PROFILES = {
    # Calibration measured <=35.2 KiB peak live residency (p90 25.6 KiB).
    # The selected profiles straddle that observed envelope instead of using
    # the historical, non-binding 512 KiB grants.
    "tight_b": {"old_grant_kb": 32, "b_grant_kb": 16},
    "roomy_b": {"old_grant_kb": 32, "b_grant_kb": 32},
    "larger_a": {"old_grant_kb": 64, "b_grant_kb": 16},
}


def tier_config(name):
    if name == "smoke":
        return dict(seed_lo=1, seed_hi=1, repeats=1, families=ALL_FAMILIES,
                    policies=POLICIES, pressures=[0.75],
                    budgets=[0.0, 0.10, 0.25, 0.50, 1.0],
                    frontiers=[0.5], grant_profiles=["tight_b", "roomy_b"])
    if name == "pilot":
        return dict(seed_lo=1, seed_hi=5, repeats=2, families=ALL_FAMILIES,
                    policies=POLICIES, pressures=[0.60, 0.75, 0.90],
                    budgets=[0.0, 0.10, 0.25, 0.50, 1.0], frontiers=[0.4, 0.6],
                    grant_profiles=list(GRANT_PROFILES))
    if name == "calibration":
        return dict(seed_lo=1, seed_hi=5, repeats=1, families=ALL_FAMILIES,
                    policies=POLICIES, pressures=[0.50, 0.75, 0.90],
                    budgets=[0.0, 0.10, 0.25, 0.50, 1.0],
                    frontiers=[0.25, 0.50, 0.75],
                    grant_profiles=list(GRANT_PROFILES))
    if name == "heldout":
        return dict(seed_lo=21, seed_hi=40, repeats=1, families=ALL_FAMILIES,
                    policies=POLICIES, pressures=[0.60, 0.75, 0.90],
                    budgets=[0.0, 0.10, 0.25, 0.50, 1.0], frontiers=[0.5],
                    grant_profiles=list(GRANT_PROFILES))
    if name == "heldout_oracle":
        return dict(seed_lo=21, seed_hi=30, repeats=1,
                    families=["chain", "diamond", "fanout_fanin"],
                    policies=["oracle"], pressures=[0.75],
                    budgets=[0.0, 0.25, 0.50], frontiers=[0.5],
                    grant_profiles=["tight_b", "roomy_b"])
    raise ValueError(f"unknown tier {name}")


def run_capture(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def git_value(root, args):
    try:
        return run_capture(["git", *args], root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build/research-debug")
    parser.add_argument("--tier", required=True,
                        choices=["smoke", "pilot", "calibration", "heldout",
                                 "heldout_oracle"])
    parser.add_argument("--result-root", default="experiments/results/evaluation_2026-11-01")
    parser.add_argument("--force", action="store_true",
                        help="allow replacing an existing tier output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build_dir = (root / args.build_dir).resolve()
    executable = build_dir / "keel_eval"
    if not executable.is_file():
        raise SystemExit(f"missing evaluator: {executable}; build it first")

    config = tier_config(args.tier)
    result_root = (root / args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / f"{args.tier}.jsonl"
    config_path = result_root / f"config_{args.tier}.json"
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to replace existing result: {output}")

    config_record = {
        "tier": args.tier,
        "build_dir": str(build_dir),
        "executable": str(executable),
        **config,
        "grant_profiles_config": GRANT_PROFILES,
        "block_size": 256,
        "protocol": "docs/EVALUATION_PROTOCOL.md",
        "bounds": {
            "abstract_graph_max_ops": 64,
            "evaluation_graph_max_ops": 11,
            "continuation_expansion_cap": 20000,
            "keel_candidate_cap": 2000,
            "keel_expansion_cap": 20000,
            "oracle_expansion_cap": 300000,
            "wait_control_deadline": "A completion; unfinished B is censored",
        },
    }
    config_path.write_text(json.dumps(config_record, indent=2) + "\n")
    common_command = [
        str(executable), "--tier", args.tier,
        "--seed-lo", str(config["seed_lo"]), "--seed-hi", str(config["seed_hi"]),
        "--repeats", str(config["repeats"]),
        "--families", ",".join(config["families"]),
        "--policies", ",".join(config["policies"]),
        "--pressures", ",".join(map(str, config["pressures"])),
        "--budgets", ",".join(map(str, config["budgets"])),
        "--frontiers", ",".join(map(str, config["frontiers"])),
    ]
    command_path = result_root / f"command_{args.tier}.txt"
    commands = []
    command_output = []

    started = dt.datetime.now(dt.timezone.utc)
    rows = 0
    with tempfile.TemporaryDirectory(prefix=f"keel-{args.tier}-") as temporary:
        with output.open("w") as combined:
            for profile in config["grant_profiles"]:
                grants = GRANT_PROFILES[profile]
                temporary_output = Path(temporary) / f"{profile}.jsonl"
                command = common_command + [
                    "--out", str(temporary_output),
                    "--grant-profile", profile,
                    "--old-grant-kb", str(grants["old_grant_kb"]),
                    "--b-grant-kb", str(grants["b_grant_kb"]),
                ]
                commands.append(command)
                command_output.append(f"$ {shlex.join(command)}\n")
                completed = run_capture(command, root)
                command_output.append(completed.stdout)
                with temporary_output.open() as source:
                    for line in source:
                        if line.strip():
                            json.loads(line)
                            combined.write(line)
                            rows += 1
    command_path.write_text("\n".join(command_output) + "\n")
    (result_root / f"stdout_{args.tier}.log").write_text("".join(command_output))

    metadata_path = result_root / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("tiers", {})[args.tier] = {
        "started_utc": started.isoformat(),
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": rows,
        "output": str(output),
        "output_sha256": file_sha256(output),
        "commands": commands,
    }
    metadata.update({
        "repository": str(root),
        "git_head": git_value(root, ["rev-parse", "HEAD"]),
        "git_status_porcelain": git_value(root, ["status", "--porcelain"]),
        "compiler": run_capture(["c++", "--version"], root).stdout.splitlines()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable_sha256": file_sha256(executable),
        "cmake_cache": str(build_dir / "CMakeCache.txt"),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{args.tier}: wrote {rows} rows to {output}")


if __name__ == "__main__":
    main()
