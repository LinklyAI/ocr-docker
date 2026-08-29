import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from run_benchmark import aggregate, prepare_image_payload


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


class PrepareImagePayloadTest(unittest.TestCase):
    def test_resizes_long_side_and_reports_matching_mime(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (400, 200), "white").save(source)

            mime, payload = prepare_image_payload(source, max_image_side=100)

            self.assertEqual(mime, "image/jpeg")
            with Image.open(BytesIO(payload)) as resized:
                self.assertEqual(resized.size, (100, 50))


if __name__ == "__main__":
    unittest.main()
