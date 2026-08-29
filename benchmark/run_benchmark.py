#!/usr/bin/env python3
"""Run a repeatable GLM-OCR benchmark against a Runpod queue endpoint."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import difflib
import hashlib
import json
import mimetypes
import os
import statistics
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def open_json(request: urllib.request.Request, attempts: int = 5) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
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


def request_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return open_json(request)


def get_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return open_json(request)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return "".join(char for char in normalized if char.isalnum())


def similarity(expected: str, actual: str) -> float:
    left = normalize_text(expected)
    right = normalize_text(actual)
    if not left:
        return 1.0 if not right else 0.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_content(output: Any) -> str:
    if isinstance(output, dict):
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        for key in ("markdown", "text", "content"):
            if isinstance(output.get(key), str):
                return output[key]
    return ""


def prepare_image_payload(asset_path: Path, max_image_side: int) -> tuple[str, bytes]:
    raw = asset_path.read_bytes()
    mime = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    if max_image_side <= 0:
        return mime, raw

    with Image.open(BytesIO(raw)) as image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_image_side:
            return mime, raw

        ratio = max_image_side / float(longest)
        resized = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            Image.Resampling.LANCZOS,
        )
        output = BytesIO()
        if "A" in resized.getbands():
            resized.save(output, format="PNG", optimize=True)
            mime = "image/png"
        else:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            resized.save(output, format="JPEG", quality=90, optimize=True)
            mime = "image/jpeg"
        return mime, output.getvalue()


def make_input(
    dataset_dir: Path,
    sample: dict[str, Any],
    max_tokens: int,
    max_image_side: int = 0,
) -> tuple[dict[str, Any], str]:
    asset_path = dataset_dir / sample["asset"]
    mime, image_bytes = prepare_image_payload(asset_path, max_image_side)
    data = base64.b64encode(image_bytes).decode("ascii")
    request = {
        "model": "zai-org/GLM-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                    {"type": "text", "text": sample["prompt"]},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    return request, hashlib.sha256(image_bytes).hexdigest()


def make_url_input(url: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": "zai-org/GLM-OCR",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def run_job(
    api_base: str,
    endpoint_id: str,
    api_key: str,
    sample: dict[str, Any],
    job_input: dict[str, Any],
    poll_interval: float,
) -> dict[str, Any]:
    submitted_at = time.time()
    submission = request_json(
        f"{api_base}/{endpoint_id}/run",
        api_key,
        {"input": job_input, "policy": {"executionTimeout": 600000}},
    )
    job_id = submission["id"]

    while True:
        status = get_json(f"{api_base}/{endpoint_id}/status/{job_id}", api_key)
        state = status.get("status")
        if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            break
        time.sleep(poll_interval)

    finished_at = time.time()
    content = extract_content(status.get("output"))
    return {
        "sample_id": sample["id"],
        "kind": sample["kind"],
        "job_id": job_id,
        "status": state,
        "submitted_at": submitted_at,
        "finished_at": finished_at,
        "wall_time_ms": round((finished_at - submitted_at) * 1000, 3),
        "delay_time_ms": status.get("delayTime"),
        "execution_time_ms": status.get("executionTime"),
        "content": content,
        "content_chars": len(content),
        "quality_similarity": round(similarity(sample["expected_text"], content), 6),
        "error": status.get("error"),
    }


def aggregate(rows: list[dict[str, Any]], price_per_hour: float) -> dict[str, Any]:
    execution = [float(row["execution_time_ms"]) for row in rows if row.get("execution_time_ms") is not None]
    delays = [float(row["delay_time_ms"]) for row in rows if row.get("delay_time_ms") is not None]
    quality = [float(row["quality_similarity"]) for row in rows if row["status"] == "COMPLETED"]
    intervals = sorted(
        (
            row["submitted_at"] + float(row["delay_time_ms"] or 0) / 1000,
            row["submitted_at"]
            + float(row["delay_time_ms"] or 0) / 1000
            + float(row["execution_time_ms"]) / 1000,
        )
        for row in rows
        if row["status"] == "COMPLETED"
        and row.get("execution_time_ms") is not None
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    execution_window_s = sum(end - start for start, end in merged)
    completed = sum(row["status"] == "COMPLETED" for row in rows)
    return {
        "requests": len(rows),
        "completed": completed,
        "failed": len(rows) - completed,
        "execution_time_ms": {
            "mean": statistics.fmean(execution) if execution else None,
            "p50": percentile(execution, 0.50),
            "p95": percentile(execution, 0.95),
            "max": max(execution) if execution else None,
        },
        "delay_time_ms": {
            "p50": percentile(delays, 0.50),
            "p95": percentile(delays, 0.95),
        },
        "quality_similarity": {
            "mean": statistics.fmean(quality) if quality else None,
            "min": min(quality) if quality else None,
        },
        "execution_window_union_s": execution_window_s,
        "execution_window_source": "runpod_delay_time_plus_execution_time",
        "throughput_pages_per_s": completed / execution_window_s if execution_window_s else None,
        "execution_window_cost_per_page_usd": (
            execution_window_s * price_per_hour / 3600 / completed if completed else None
        ),
    }


def select_samples(samples: list[dict[str, Any]], kind: str | None) -> list[dict[str, Any]]:
    selected = samples if kind is None else [sample for sample in samples if sample["kind"] == kind]
    if not selected:
        raise ValueError(f"No benchmark samples found for kind={kind!r}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).with_name("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-image-side", type=int, default=0)
    parser.add_argument("--kind", choices=("image", "document"))
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--price-per-hour", type=float, default=1.10)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument(
        "--warmup-url",
        default="https://raw.githubusercontent.com/zai-org/GLM-OCR/main/resources/speed.png",
    )
    parser.add_argument("--api-base", default="https://api.runpod.ai/v2")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["samples"] = select_samples(manifest["samples"], args.kind)
    prepared = {
        sample["id"]: make_input(
            args.dataset_dir,
            sample,
            args.max_tokens,
            args.max_image_side,
        )
        for sample in manifest["samples"]
    }
    prepared_inputs = {sample_id: value[0] for sample_id, value in prepared.items()}
    request_sha256 = {sample_id: value[1] for sample_id, value in prepared.items()}

    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    partial_path.write_text("", encoding="utf-8")
    warmup_sample = {
        "id": "warmup-public-speed",
        "kind": "warmup",
        "expected_text": "",
    }
    warmup_input = make_url_input(args.warmup_url, args.max_tokens)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        warmup_futures = [
            pool.submit(
                run_job,
                args.api_base,
                args.endpoint_id,
                api_key,
                warmup_sample,
                warmup_input,
                args.poll_interval,
            )
            for _ in range(args.warmup_requests)
        ]
        warmup_rows = [future.result() for future in concurrent.futures.as_completed(warmup_futures)]
    if any(row["status"] != "COMPLETED" for row in warmup_rows):
        raise RuntimeError(f"Warmup failed: {warmup_rows}")
    time.sleep(1)

    for round_index in range(args.rounds):
        samples = manifest["samples"]
        offset = round_index % len(samples)
        ordered = samples[offset:] + samples[:offset]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    run_job,
                    args.api_base,
                    args.endpoint_id,
                    api_key,
                    sample,
                    prepared_inputs[sample["id"]],
                    args.poll_interval,
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
                    f"status={row['status']} exec_ms={row['execution_time_ms']} "
                    f"quality={row['quality_similarity']}"
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)

    result = {
        "schema_version": 1,
        "variant": args.variant,
        "endpoint_id": args.endpoint_id,
        "gpu": "NVIDIA GeForce RTX 4090",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rounds": args.rounds,
            "client_concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "max_image_side": args.max_image_side,
            "kind": args.kind,
            "price_per_hour_usd": args.price_per_hour,
            "warmup_requests": args.warmup_requests,
            "warmup_url": args.warmup_url,
        },
        "dataset": {
            "version": manifest["version"],
            "sample_count": len(manifest["samples"]),
            "sample_sha256": {
                sample["id"]: file_sha256(args.dataset_dir / sample["asset"])
                for sample in manifest["samples"]
            },
            "request_image_sha256": request_sha256,
        },
        "summary": {
            "overall": aggregate(rows, args.price_per_hour),
            **{kind: aggregate(kind_rows, args.price_per_hour) for kind, kind_rows in grouped.items()},
        },
        "warmup_rows": warmup_rows,
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
