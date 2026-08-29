#!/usr/bin/env python3
"""Materialize the private GLM-OCR benchmark from the existing local corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    "/opt/src/ai/linkly-ai.bak/projects/linkly-ai-desktop-benchmark/ocr-benchmark"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pdf_page_text(pdf_path: Path, page: int) -> str:
    completed = subprocess.run(
        [
            "pdftotext",
            "-f",
            str(page),
            "-l",
            str(page),
            "-layout",
            str(pdf_path),
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip("\n\f ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).with_name("dataset-spec.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("dataset"),
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    image_groundtruth_path = source_root / "datasets/image_groundtruth.json"
    image_groundtruth = {
        row["file"]: row["gt_text"]
        for row in json.loads(image_groundtruth_path.read_text(encoding="utf-8"))
    }

    assets_dir = args.output / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    materialized = []

    for sample in spec["samples"]:
        source_path = source_root / sample["path"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        suffix = source_path.suffix.lower()
        output_name = f"{sample['id']}{suffix}"
        output_path = assets_dir / output_name
        shutil.copy2(source_path, output_path)

        if "groundtruth_file" in sample:
            expected_text = image_groundtruth[sample["groundtruth_file"]]
        else:
            expected_text = extract_pdf_page_text(
                source_root / sample["groundtruth_pdf"], sample["page"]
            )

        materialized.append(
            {
                "id": sample["id"],
                "kind": sample["kind"],
                "asset": f"assets/{output_name}",
                "prompt": sample.get("prompt", "Text Recognition:"),
                "expected_text": expected_text,
                "sha256": sha256(output_path),
                "bytes": output_path.stat().st_size,
            }
        )

    manifest = {
        "version": spec["version"],
        "description": spec["description"],
        "source_root": str(source_root),
        "samples": materialized,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(materialized)} samples at {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
