import csv
import json
import os
import tempfile
import unittest

from app import MatchingApp
from matcher import validate_response


FOLDER = os.path.dirname(__file__)
TASKS_PATH = os.path.join(FOLDER, "task.csv")
PEOPLE_PATH = os.path.join(FOLDER, "people.csv")

PEOPLE_BY_TASK = {
    "t1": "p1",
    "t2": "p2",
    "t3": "p3",
    "t4": "p4",
}


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def call(self, payload=None):
        self.calls += 1
        prompt = payload["messages"][0]["content"]
        task_part = prompt.split("People:")[0]
        for task_id, person_id in PEOPLE_BY_TASK.items():
            if f'"id": "{task_id}"' in task_part:
                return json.dumps(
                    {
                        "task_id": task_id,
                        "person_id": person_id,
                        "reason": {"explanation": "good match"},
                    }
                )
        raise AssertionError("task id not found in prompt")


class FailingLLM:
    def call(self, payload=None):
        raise RuntimeError("match failed")


class TestApp(unittest.TestCase):
    def temporary_path(self):
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def test_run_returns_valid_matches(self):
        output_path = self.temporary_path()
        llm = FakeLLM()

        _, _, responses = MatchingApp(llm).run(
            TASKS_PATH,
            PEOPLE_PATH,
            output_path,
        )

        self.assertEqual(len(responses), 4)
        self.assertEqual(llm.calls, 4)
        for response in responses:
            validate_response(response)

    def test_run_writes_csv(self):
        output_path = self.temporary_path()

        MatchingApp(FakeLLM()).run(TASKS_PATH, PEOPLE_PATH, output_path)

        with open(output_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)

    def test_parse_failure_does_not_call_llm(self):
        llm = FakeLLM()

        def fail_parse(path):
            raise csv.Error("invalid csv")

        result = MatchingApp(llm, task_parser=fail_parse).run(
            TASKS_PATH,
            PEOPLE_PATH,
            self.temporary_path(),
        )

        self.assertEqual(result, ([], [], []))
        self.assertEqual(llm.calls, 0)

    def test_match_failure_is_raised(self):
        self.assertRaises(
            RuntimeError,
            MatchingApp(FailingLLM()).run,
            TASKS_PATH,
            PEOPLE_PATH,
            self.temporary_path(),
        )


if __name__ == "__main__":
    unittest.main()
