import json
import os
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from matching import Match, match_all, match_person_to_task, openai_call, shortlist, validate_match
from parsing import Person, Task


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.people = [
            Person("p1", "Alice", ["music", "piano"], 5),
            Person("p2", "Bob", ["python"], 7),
        ]
        self.t1 = Task("t1", "Music Agent Evaluation", "Evaluate piano music")
        self.t2 = Task("t2", "ML Response Evaluation", "Evaluate ML answers")

    def test_happy_path_match_person_to_task(self):
        def llm_call(_prompt):
            return json.dumps({"person_id": "p1", "reason": "piano"})

        match = match_person_to_task(self.t1, self.people, llm_call)
        self.assertEqual(match, Match("t1", "p1", "piano"))

    def test_malformed_json_raises(self):
        def llm_call(_prompt):
            return "not-json"

        with self.assertRaises(ValueError):
            match_person_to_task(self.t1, self.people, llm_call)

    def test_missing_keys_raises(self):
        def llm_call(_prompt):
            return json.dumps({"person_id": "p1"})

        with self.assertRaises(ValueError):
            match_person_to_task(self.t1, self.people, llm_call)

    def test_unknown_id_raises(self):
        def llm_call(_prompt):
            return json.dumps({"person_id": "p99", "reason": "invented"})

        with self.assertRaises(ValueError):
            match_person_to_task(self.t1, self.people, llm_call)

    def test_validate_match_rejects_unknown_id(self):
        with self.assertRaises(ValueError):
            validate_match(Match("t1", "p99", "nope"), self.people)

    def test_match_all_records_error_and_continues(self):
        def llm_call(prompt):
            if '"id": "t1"' in prompt:
                return "not-json"
            return json.dumps({"person_id": "p2", "reason": "python"})

        results = match_all([self.t1, self.t2], self.people, llm_call)
        self.assertIsNotNone(results[0]["error"])
        self.assertEqual(results[0]["task_id"], "t1")
        self.assertIsNone(results[0]["person_id"])
        self.assertIsNone(results[1]["error"])
        self.assertEqual(results[1]["person_id"], "p2")
        self.assertEqual(results[1]["task_id"], "t2")

    def test_openai_call_returns_message_content(self):
        payload = {
            "choices": [{"message": {"content": '{"person_id": "p1", "reason": "ok"}'}}]
        }

        class FakeResp:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("urllib.request.urlopen", return_value=FakeResp()):
                text = openai_call("hello")
        self.assertEqual(text, '{"person_id": "p1", "reason": "ok"}')

    def test_fenced_json_still_parses(self):
        def llm_call(_prompt):
            return '```json\n{"person_id": "p1", "reason": "fenced"}\n```'

        match = match_person_to_task(self.t1, self.people, llm_call)
        self.assertEqual(match.person_id, "p1")
        self.assertEqual(match.reason, "fenced")

    def test_shortlist_ranks_overlapping_skills_first(self):
        chosen = shortlist(self.t1, self.people, k=2)
        self.assertEqual(chosen[0].id, "p1")

    def test_skips_checkpointed_task_ids(self):
        calls = []

        def llm_call(prompt):
            calls.append(prompt)
            return json.dumps({"person_id": "p2", "reason": "ok"})

        results = match_all(
            [self.t1, self.t2],
            self.people,
            llm_call,
            k=2,
            workers=1,
            batch_size=2,
            done_ids={"t1"},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual([row["task_id"] for row in results], ["t2"])

    def test_openai_retries_on_429_then_succeeds(self):
        payload = {
            "choices": [{"message": {"content": '{"person_id": "p1", "reason": "ok"}'}}]
        }

        class FakeResp:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        rate_limited = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=BytesIO(b"{}"),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("urllib.request.urlopen", side_effect=[rate_limited, FakeResp()]):
                text = openai_call("hello")
        self.assertEqual(text, '{"person_id": "p1", "reason": "ok"}')


if __name__ == "__main__":
    unittest.main()
