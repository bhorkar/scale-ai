import csv
import tempfile
import unittest
from pathlib import Path

from app import run_matching, write_matches

HERE = Path(__file__).parent


class TestApp(unittest.TestCase):
    def test_write_skips_error_rows(self):
        class FakeLlm:
            def build_prompt(self, task, people):
                return "Return a JSON object"

            def build_payload(self, prompt):
                return {"messages": [{"role": "user", "content": prompt}]}

            def call(self, payload, task=None, people=None):
                return {"task": task.id, "people": people[0].id, "reason": "ok"}

        out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        out.close()
        run_matching(HERE / "task.csv", HERE / "people.csv", out.name, FakeLlm())
        with open(out.name) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["people_id"], "p1")

        write_matches(
            out.name,
            [
                {"task": "t1", "people": "p1", "reason": "ok"},
                {"error": "bad", "task_id": "t2"},
            ],
        )
        with open(out.name) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["task_id"] for row in rows], ["t1"])


if __name__ == "__main__":
    unittest.main()
