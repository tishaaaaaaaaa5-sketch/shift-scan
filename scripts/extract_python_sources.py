#!/usr/bin/env python3
"""Safely extract Python source files from a Shift-Scan experiment ZIP.

Example:
  python scripts/extract_python_sources.py \
    archives/experiment_bundles/shift_scan_advanced_project.zip \
    --output src/shift_scan_legacy

The script extracts only .py files, rejects path traversal, and preserves
archive-relative paths. Review the extracted files before committing them.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def safe_destination(root: Path, member_name: str) -> Path | None:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        return None
    if member.suffix.lower() != ".py":
        return None

    # Remove the common top-level project folder if present.
    parts = member.parts[1:] if member.parts and member.parts[0] in {
        "shift_scan", "shift-scan"
    } else member.parts
    if not parts:
        return None

    destination = root.joinpath(*parts).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        return None
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("src/shift_scan_legacy"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    skipped = 0

    with ZipFile(args.archive) as archive:
        for info in archive.infolist():
            destination = safe_destination(args.output, info.filename)
            if destination is None:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
            extracted.append(destination)

    print(f"Extracted {len(extracted)} Python files to {args.output}")
    print(f"Skipped {skipped} non-Python or unsafe archive entries")
    for path in extracted:
        print(path)


if __name__ == "__main__":
    main()
