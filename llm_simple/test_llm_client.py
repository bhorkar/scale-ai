import unittest

from llm_client import LLMClient
from parser import Person, Task


class FakeResponse:
    def __init__(self, body, status_code=200, text=""):
        self.body = body
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.body


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.task = Task("t1", "RAG", ["python"], ["ml"], "ml")
        self.people = [Person("p1", "Alice", ["python"], ["ml"])]

    def test_post_with_payload_returns_content(self):
        seen = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            seen["url"] = url
            seen["json"] = json
            seen["headers"] = headers
            seen["timeout"] = timeout
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"task": "t1", "people": "p1", "reason": "ok"}'
                            }
                        }
                    ]
                }
            )

        llm = LLMClient(api_key="sk-test", url="https://example.test", post=fake_post)
        payload = llm.build_payload("return a JSON object")
        data = llm.call(payload, task=self.task, people=self.people)
        self.assertEqual(data["people"], "p1")
        self.assertEqual(seen["json"], payload)
        self.assertEqual(seen["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(seen["url"], "https://example.test")
        self.assertEqual(seen["timeout"], 30)

    def test_errors_when_response_is_an_error(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(1)
            return FakeResponse({}, status_code=400, text="json must be in the messages")

        llm = LLMClient(api_key="sk-test", post=fake_post)
        with self.assertRaises(ValueError) as err:
            llm.call({"model": "x"}, task=self.task, people=self.people)
        self.assertIn("json must be in the messages", str(err.exception))
        self.assertEqual(len(calls), 1)

    def test_retries_temporary_then_succeeds(self):
        calls = []
        ok = {
            "choices": [
                {"message": {"content": '{"task": "t1", "people": "p1", "reason": "ok"}'}}
            ]
        }

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                return FakeResponse({}, status_code=429, text="rate limited")
            if len(calls) == 2:
                return FakeResponse({}, status_code=503, text="unavailable")
            return FakeResponse(ok)

        llm = LLMClient(api_key="sk-test", post=fake_post)
        data = llm.call({"model": "x"}, task=self.task, people=self.people)
        self.assertEqual(data["people"], "p1")
        self.assertEqual(len(calls), 3)

    def test_retries_exhausted_on_5xx(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(1)
            return FakeResponse({}, status_code=500, text="server error")

        llm = LLMClient(api_key="sk-test", post=fake_post, retries=3)
        with self.assertRaises(ValueError) as err:
            llm.call({"model": "x"}, task=self.task, people=self.people)
        self.assertIn("server error", str(err.exception))
        self.assertEqual(len(calls), 3)

    def test_call_validate_error(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResponse({"choices": [{"message": {"content": "not json"}}]})

        llm = LLMClient(api_key="sk-test", post=fake_post)
        with self.assertRaises(ValueError):
            llm.call({"model": "x"}, task=self.task, people=self.people)

    def test_validate_content_ok(self):
        llm = LLMClient(api_key="sk-test")
        data = llm.validate_content(
            '{"task": "t1", "people": "p1", "reason": "skills match"}',
            self.task,
            self.people,
        )
        self.assertEqual(data["people"], "p1")

    def test_validate_content_rejects_bad(self):
        llm = LLMClient(api_key="sk-test")
        with self.assertRaises(ValueError):
            llm.validate_content("not json", self.task, self.people)
        with self.assertRaises(ValueError):
            llm.validate_content(
                '{"task": "t9", "people": "p1", "reason": "x"}',
                self.task,
                self.people,
            )
        with self.assertRaises(ValueError):
            llm.validate_content(
                '{"task": "t1", "people": "p9", "reason": "x"}',
                self.task,
                self.people,
            )


if __name__ == "__main__":
    unittest.main()

