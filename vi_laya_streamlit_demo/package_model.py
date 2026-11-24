"""Create and verify a ZIP of the complete original Vi Laya checkpoint."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = WORKSPACE_ROOT / "vi_laya_checkpoint" / "vi-laya-s0"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "vi_laya_model_bundle.zip"
REQUIRED_RELATIVE_FILES = (
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "questions.json",
    "rl_agent_config.json",
    "train_report.json",
    "checkpoint_latest/model.safetensors",
    "checkpoint_latest/questions.json",
    "checkpoint_latest/rl_agent_config.json",
)


def package_model(source: Path, output: Path) -> list[str]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {source}")

    missing = [relative for relative in REQUIRED_RELATIVE_FILES if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError("Checkpoint is incomplete; missing: " + ", ".join(missing))

    files = sorted(path for path in source.rglob("*") if path.is_file())
    archive_root = Path("vi_laya_checkpoint") / source.name
    expected = {str(archive_root / path.relative_to(source)) for path in files}
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, archive_root / path.relative_to(source))

    with zipfile.ZipFile(output, "r") as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise zipfile.BadZipFile(f"CRC verification failed for {bad_member}")
        actual = set(archive.namelist())
        missing_in_zip = expected - actual
        if missing_in_zip:
            raise zipfile.BadZipFile("ZIP is missing files: " + ", ".join(sorted(missing_in_zip)))
        archive_required = {
            str(archive_root / relative) for relative in REQUIRED_RELATIVE_FILES
        }
        if not archive_required.issubset(actual):
            raise zipfile.BadZipFile("ZIP verification failed for required checkpoint files")

    return sorted(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inventory = package_model(args.source, args.output)
    print(f"Created and verified: {args.output.resolve()}")
    print(f"ZIP size: {args.output.resolve().stat().st_size:,} bytes")
    print(f"Files ({len(inventory)}):")
    for name in inventory:
        print(f"  {name}")


if __name__ == "__main__":
    main()
