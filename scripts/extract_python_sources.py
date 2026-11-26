#!/usr/bin/env python3
"""Safely extract Python modules from the archived Shift-Scan project ZIP.

Example:
  python scripts/extract_python_sources.py archives/experiment_bundles/shift_scan_advanced_project.zip --output src/shift_scan_legacy

Review extracted code and run tests before committing it.
"""
from __future__ import annotations
import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def destination_for(root: Path, member_name: str) -> Path | None:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts or member.suffix.lower() != ".py":
        return None
    parts = member.parts[1:] if member.parts and member.parts[0] in {"shift_scan", "shift-scan"} else member.parts
    if not parts:
        return None
    dest = root.joinpath(*parts).resolve()
    base = root.resolve()
    return dest if dest == base or base in dest.parents else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("src/shift_scan_legacy"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extracted, skipped = [], 0
    with ZipFile(args.archive) as archive:
        for item in archive.infolist():
            dest = destination_for(args.output, item.filename)
            if dest is None:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(item))
            extracted.append(dest)
    print(f"Extracted {len(extracted)} Python files into {args.output}")
    print(f"Skipped {skipped} non-Python or unsafe entries")
    for path in extracted:
        print(path)


if __name__ == "__main__":
    main()
