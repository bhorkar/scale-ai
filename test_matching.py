import json
import os
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from matching import Match, cosine, match_all, match_person_to_task, openai_call, shortlist, validate_match
from parsing import Person, Task


def make_llm_call(person_id, reason="ok"):
    return lambda _prompt: json.dumps({"person_id": person_id, "reason": reason})


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.people = [
            Person("p1", "Alice", ["music", "piano"], 5),
            Person("p2", "Bob", ["python"], 7),
        ]
        self.t1 = Task("t1", "Music Agent Evaluation", "Evaluate piano music")
        self.t2 = Task("t2", "ML Response Evaluation", "Evaluate ML answers")

    def test_happy_path_match_person_to_task(self):
        match = match_person_to_task(self.t1, self.people, make_llm_call("p1", "piano"))
        self.assertEqual(match, Match("t1", "p1", "piano"))

    def test_malformed_json_raises(self):
        with self.assertRaises(ValueError):
            match_person_to_task(self.t1, self.people, lambda _p: "not-json")

    def test_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            match_person_to_task(self.t1, self.people, lambda _p: json.dumps({"person_id": "p1"}))

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            match_person_to_task(self.t1, self.people, make_llm_call("p99", "invented"))

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
            with patch("urllib.request.urlopen", return_value=FakeResp()) as mocked:
                text = openai_call("hello")
        self.assertEqual(text, '{"person_id": "p1", "reason": "ok"}')
        body = json.loads(mocked.call_args[0][0].data)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_fenced_json_still_parses(self):
        match = match_person_to_task(
            self.t1,
            self.people,
            lambda _p: '```json\n{"person_id": "p1", "reason": "fenced"}\n```',
        )
        self.assertEqual(match.person_id, "p1")
        self.assertEqual(match.reason, "fenced")

    def test_cosine_identical_is_one_orthogonal_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_shortlist_ranks_by_cosine_similarity(self):
        def embed_fn(text):
            t = text.lower()
            return [float("music" in t or "piano" in t), float("python" in t)]

        chosen = shortlist(self.t1, self.people, k=2, embed_fn=embed_fn)
        self.assertEqual(chosen[0].id, "p1")
        self.assertEqual(chosen[1].id, "p2")

    def test_shortlist_empty_people_returns_empty(self):
        self.assertEqual(shortlist(self.t1, [], k=2), [])

    def test_checkpoint_mutates_caller_set_and_skips_failures(self):
        done = set()

        def llm_call(prompt):
            if '"id": "t1"' in prompt:
                return "not-json"
            return json.dumps({"person_id": "p2", "reason": "ok"})

        match_all([self.t1, self.t2], self.people, llm_call, workers=1, batch_size=2, done_ids=done)
        self.assertEqual(done, {"t2"})

    def test_skips_checkpointed_task_ids(self):
        calls = []
        inner = make_llm_call("p2")

        def llm_call(prompt):
            calls.append(prompt)
            return inner(prompt)

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

    def test_exclusive_assign_rematch_on_collision(self):
        def llm_call(prompt):
            person_id = "p1" if '"p1"' in prompt else "p2"
            return json.dumps({"person_id": person_id, "reason": "ok"})

        results = match_all([self.t1, self.t2], self.people, llm_call, k=2, workers=2, batch_size=2)
        self.assertIsNone(results[0]["error"])
        self.assertEqual(results[0]["person_id"], "p1")
        self.assertIsNone(results[1]["error"])
        self.assertEqual(results[1]["person_id"], "p2")

    def test_exclusive_assign_quarantines_when_no_free_person(self):
        people = [self.people[0]]
        results = match_all(
            [self.t1, self.t2],
            people,
            make_llm_call("p1", "only one"),
            k=1,
            workers=1,
            batch_size=2,
        )
        self.assertEqual(results[0]["person_id"], "p1")
        self.assertIsNone(results[0]["error"])
        self.assertIsNone(results[1]["person_id"])
        self.assertEqual(results[1]["error"], "no free person")

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
