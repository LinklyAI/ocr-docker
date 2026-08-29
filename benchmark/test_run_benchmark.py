import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from run_benchmark import (
    aggregate,
    extract_region_count,
    extract_usage,
    prepare_image_payload,
    select_samples,
)


class AggregateTest(unittest.TestCase):
    def test_execution_window_merges_overlapping_requests(self):
        rows = [
            {
                "status": "COMPLETED",
                "execution_time_ms": 2000,
                "delay_time_ms": 1000,
                "submitted_at": 10.0,
                "finished_at": 99.0,
                "quality_similarity": 1.0,
            },
            {
                "status": "COMPLETED",
                "execution_time_ms": 2000,
                "delay_time_ms": 1000,
                "submitted_at": 11.0,
                "finished_at": 99.0,
                "quality_similarity": 1.0,
            },
        ]

        result = aggregate(rows, price_per_hour=1.0)

        self.assertEqual(result["execution_window_union_s"], 3.0)
        self.assertEqual(
            result["execution_window_source"],
            "runpod_delay_time_plus_execution_time",
        )
        self.assertEqual(result["throughput_pages_per_s"], 2 / 3)
        self.assertAlmostEqual(
            result["execution_window_cost_per_page_usd"], 3 / 3600 / 2
        )

    def test_summarizes_tokens_and_layout_regions(self):
        rows = [
            {
                "status": "COMPLETED",
                "execution_time_ms": 1000,
                "delay_time_ms": 0,
                "submitted_at": 10.0,
                "finished_at": 11.0,
                "quality_similarity": 1.0,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "region_count": 3,
            },
            {
                "status": "COMPLETED",
                "execution_time_ms": 1000,
                "delay_time_ms": 0,
                "submitted_at": 11.0,
                "finished_at": 12.0,
                "quality_similarity": 1.0,
                "prompt_tokens": 200,
                "completion_tokens": 40,
                "total_tokens": 240,
                "region_count": 5,
            },
        ]

        result = aggregate(rows, price_per_hour=1.0)

        self.assertEqual(result["total_tokens"], {"mean": 180.0, "sum": 360})
        self.assertEqual(result["region_count"], {"mean": 4.0, "sum": 8})


class OutputMetricsTest(unittest.TestCase):
    def test_extracts_openai_usage(self):
        output = {
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 23,
                "total_tokens": 124,
            }
        }

        self.assertEqual(
            extract_usage(output),
            {"prompt_tokens": 101, "completion_tokens": 23, "total_tokens": 124},
        )

    def test_counts_sdk_regions_across_pages(self):
        output = {
            "layout_json": [
                [{"bbox_2d": [0, 0, 1, 1]}, {"bbox_2d": [1, 1, 2, 2]}],
                [{"bbox_2d": [2, 2, 3, 3]}],
            ]
        }

        self.assertEqual(extract_region_count(output), 3)

    def test_full_page_response_has_no_region_metric(self):
        self.assertIsNone(extract_region_count({"choices": []}))


class PrepareImagePayloadTest(unittest.TestCase):
    def test_resizes_long_side_and_reports_matching_mime(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (400, 200), "white").save(source)

            mime, payload = prepare_image_payload(source, max_image_side=100)

            self.assertEqual(mime, "image/jpeg")
            with Image.open(BytesIO(payload)) as resized:
                self.assertEqual(resized.size, (100, 50))


class SelectSamplesTest(unittest.TestCase):
    def test_filters_samples_by_kind(self):
        samples = [
            {"id": "image-1", "kind": "image"},
            {"id": "document-1", "kind": "document"},
        ]

        self.assertEqual(select_samples(samples, "image"), [samples[0]])
        self.assertEqual(select_samples(samples, None), samples)

    def test_rejects_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "kind='document'"):
            select_samples([{"id": "image-1", "kind": "image"}], "document")


if __name__ == "__main__":
    unittest.main()
