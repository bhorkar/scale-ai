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

    def test_call_returns_valid_match(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResponse(
                {"choices": [{"message": {"content": '{"task": "t1", "people": "p1", "reason": "skills match"}'}}]}
            )

        llm = LLMClient(api_key="sk-test", post=fake_post)
        data = llm.call({"model": "x"}, task=self.task, people=self.people)

        self.assertEqual(data, {"task": "t1", "people": "p1", "reason": "skills match"})

    def test_call_retries_transient_errors(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                return FakeResponse({}, status_code=429, text="rate limited")
            return FakeResponse(
                {"choices": [{"message": {"content": '{"task": "t1", "people": "p1", "reason": "ok"}'}}]}
            )

        llm = LLMClient(api_key="sk-test", post=fake_post)
        data = llm.call({"model": "x"}, task=self.task, people=self.people)

        self.assertEqual(data["people"], "p1")
        self.assertEqual(len(calls), 2)

    def test_call_rejects_invalid_content(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResponse({"choices": [{"message": {"content": "not json"}}]})

        llm = LLMClient(api_key="sk-test", post=fake_post)
        with self.assertRaises(ValueError):
            llm.call({"model": "x"}, task=self.task, people=self.people)


if __name__ == "__main__":
    unittest.main()
