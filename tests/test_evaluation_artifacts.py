import csv
from datetime import datetime, timezone
import json
import tempfile
from pathlib import Path
import unittest

from terrain_adaptation_rls.evaluation.artifacts import (
    create_run_dir,
    summarize_errors,
    write_json,
    write_streaming_errors_csv,
)
from terrain_adaptation_rls.evaluation.streaming import StreamingRecord


class EvaluationArtifactTests(unittest.TestCase):
    def test_create_run_dir_uses_timestamp_and_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_run_dir(
                tmpdir,
                run_name="debug",
                timestamp=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(path.name, "20260706T120000Z_debug")
            self.assertTrue(path.exists())

    def test_write_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            write_json(path, {"b": 2, "a": {"path": Path("x")}})
            data = json.loads(path.read_text())

            self.assertEqual(data, {"a": {"path": "x"}, "b": 2})

    def test_write_streaming_errors_csv(self):
        records = [
            StreamingRecord(0, 0.0, 1.0, 2.0, 1.0, None, None),
            StreamingRecord(1, None, 2.0, 5.0, 3.0, None, None),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "errors.csv"
            write_streaming_errors_csv(path, records)
            with path.open() as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0], {"index": "0", "time": "0.0", "error": "1.0"})
        self.assertEqual(rows[1], {"index": "1", "time": "", "error": "3.0"})

    def test_summarize_errors(self):
        records = [
            StreamingRecord(0, 0.0, None, None, 1.0, None, None),
            StreamingRecord(1, 0.1, None, None, 3.0, None, None),
        ]

        summary = summarize_errors(records)

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["mean_error"], 2.0)
        self.assertEqual(summary["final_accumulated_error"], 4.0)


if __name__ == "__main__":
    unittest.main()
