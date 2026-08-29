import unittest

from run_benchmark import aggregate


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


if __name__ == "__main__":
    unittest.main()
