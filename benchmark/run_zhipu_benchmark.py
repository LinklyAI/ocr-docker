#!/usr/bin/env python3
"""Benchmark Zhipu GLM-OCR using the same prepared bytes as Runpod."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_benchmark import file_sha256, percentile, prepare_image_payload, similarity


def request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    attempts: int = 5,
) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 429} or exc.code >= 500
            if not retryable or attempt == attempts:
                raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def prepare_file(
    asset_path: Path,
    max_image_side: int,
    base64_format: str,
) -> tuple[str, str]:
    mime, image_bytes = prepare_image_payload(asset_path, max_image_side)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    if base64_format == "data-url":
        encoded = f"data:{mime};base64,{encoded}"
    return encoded, hashlib.sha256(image_bytes).hexdigest()


def run_request(
    api_url: str,
    api_key: str,
    sample: dict[str, Any],
    prepared_file: str,
) -> dict[str, Any]:
    started_at = time.time()
    try:
        response = request_json(
            api_url,
            api_key,
            {
                "model": "glm-ocr",
                "file": prepared_file,
                "return_crop_images": False,
                "need_layout_visualization": False,
            },
        )
        finished_at = time.time()
        content = response.get("md_results")
        if not isinstance(content, str):
            content = ""
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return {
            "sample_id": sample["id"],
            "kind": sample["kind"],
            "status": "COMPLETED",
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_time_ms": round((finished_at - started_at) * 1000, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "content": content,
            "content_chars": len(content),
            "quality_similarity": round(similarity(sample["expected_text"], content), 6),
            "request_id": response.get("request_id"),
            "task_id": response.get("id"),
            "error": None,
        }
    except Exception as exc:
        finished_at = time.time()
        return {
            "sample_id": sample["id"],
            "kind": sample["kind"],
            "status": "FAILED",
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_time_ms": round((finished_at - started_at) * 1000, 3),
            "error": str(exc),
        }


def aggregate(rows: list[dict[str, Any]], price_per_million_cny: float) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    latencies = [float(row["wall_time_ms"]) for row in completed]
    prompt_tokens = [int(row["prompt_tokens"]) for row in completed if row.get("prompt_tokens") is not None]
    completion_tokens = [
        int(row["completion_tokens"]) for row in completed if row.get("completion_tokens") is not None
    ]
    total_tokens = [int(row["total_tokens"]) for row in completed if row.get("total_tokens") is not None]
    quality = [float(row["quality_similarity"]) for row in completed]
    total_cost_cny = sum(total_tokens) * price_per_million_cny / 1_000_000
    return {
        "requests": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "wall_time_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "prompt_tokens": {
            "mean": statistics.fmean(prompt_tokens) if prompt_tokens else None,
            "sum": sum(prompt_tokens),
        },
        "completion_tokens": {
            "mean": statistics.fmean(completion_tokens) if completion_tokens else None,
            "sum": sum(completion_tokens),
        },
        "total_tokens": {
            "mean": statistics.fmean(total_tokens) if total_tokens else None,
            "p50": percentile([float(value) for value in total_tokens], 0.50),
            "p95": percentile([float(value) for value in total_tokens], 0.95),
            "sum": sum(total_tokens),
        },
        "quality_similarity": {
            "mean": statistics.fmean(quality) if quality else None,
            "min": min(quality) if quality else None,
        },
        "billed_cost_cny": total_cost_cny,
        "billed_cost_per_page_cny": total_cost_cny / len(completed) if completed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).with_name("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-image-side", type=int, default=1900)
    parser.add_argument("--price-per-million-cny", type=float, default=0.2)
    parser.add_argument("--base64-format", choices=("data-url", "raw"), default="data-url")
    parser.add_argument(
        "--api-url",
        default="https://open.bigmodel.cn/api/paas/v4/layout_parsing",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise SystemExit("ZHIPU_API_KEY is required")

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    prepared = {
        sample["id"]: prepare_file(
            args.dataset_dir / sample["asset"],
            args.max_image_side,
            args.base64_format,
        )
        for sample in samples
    }
    prepared_files = {sample_id: value[0] for sample_id, value in prepared.items()}
    request_sha256 = {sample_id: value[1] for sample_id, value in prepared.items()}

    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    partial_path.write_text("", encoding="utf-8")

    for round_index in range(args.rounds):
        offset = round_index % len(samples)
        ordered = samples[offset:] + samples[:offset]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    run_request,
                    args.api_url,
                    api_key,
                    sample,
                    prepared_files[sample["id"]],
                )
                for sample in ordered
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                row["round"] = round_index + 1
                rows.append(row)
                with partial_path.open("a", encoding="utf-8") as partial:
                    partial.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"round={round_index + 1} sample={row['sample_id']} "
                    f"status={row['status']} latency_ms={row['wall_time_ms']} "
                    f"tokens={row.get('total_tokens')} quality={row.get('quality_similarity')}"
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)
    result = {
        "schema_version": 1,
        "provider": "Zhipu BigModel",
        "model": "glm-ocr",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rounds": args.rounds,
            "client_concurrency": args.concurrency,
            "max_image_side": args.max_image_side,
            "base64_format": args.base64_format,
            "price_per_million_tokens_cny": args.price_per_million_cny,
            "billing_formula": "total_tokens * price_per_million_tokens_cny / 1000000",
            "api_url": args.api_url,
        },
        "dataset": {
            "version": manifest["version"],
            "sample_count": len(samples),
            "sample_sha256": {
                sample["id"]: file_sha256(args.dataset_dir / sample["asset"])
                for sample in samples
            },
            "request_image_sha256": request_sha256,
        },
        "summary": {
            "overall": aggregate(rows, args.price_per_million_cny),
            **{
                kind: aggregate(kind_rows, args.price_per_million_cny)
                for kind, kind_rows in grouped.items()
            },
        },
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
