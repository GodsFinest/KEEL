#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a result directory."""

import argparse
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", default="MANIFEST.sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    output.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"manifest: {len(lines)} files -> {output}")


if __name__ == "__main__":
    main()
