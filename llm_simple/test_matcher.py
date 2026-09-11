import time
import unittest

from llm_client import LLMClient
from matcher import match_all
from parser import Person, Task


class FakeLlm:
    def build_prompt(self, task, people):
        return "Return a JSON object"

    def build_payload(self, prompt):
        return {"messages": [{"role": "user", "content": prompt}]}

    def call(self, payload, task=None, people=None):
        return {"task": task.id, "people": people[0].id, "reason": "skills match"}


class TestMatcher(unittest.TestCase):
    def test_parses_json_content(self):
        t1 = Task("t1", "RAG", ["python"], ["ml"], "ml")
        p1 = Person("p1", "Alice", ["python"], ["ml"])
        rows = match_all([t1], [p1], FakeLlm())
        self.assertEqual(rows[0]["people"], "p1")

    def test_bad_task_does_not_drop_the_next(self):
        class FlakyLlm(FakeLlm):
            def call(self, payload, task=None, people=None):
                if task.id == "t1":
                    raise ValueError("llm http 400")
                return {"task": task.id, "people": people[0].id, "reason": "ok"}

        t1 = Task("t1", "RAG", ["python"], ["ml"], "ml")
        t2 = Task("t2", "UI", ["react"], ["fe"], "fe")
        p1 = Person("p1", "Alice", ["python"], ["ml"])
        rows = match_all([t1, t2], [p1], FlakyLlm())
        self.assertTrue(rows[0]["error"])
        self.assertEqual(rows[1]["task"], "t2")

    def test_match_all_keeps_input_order(self):
        class SlowClient(LLMClient):
            def call(self, payload, task=None, people=None):
                time.sleep(0.05 if task.id == "t1" else 0.0)
                return {"task": task.id, "people": people[0].id, "reason": "ok"}

        t1 = Task("t1", "RAG", ["python"], ["ml"], "ml")
        t2 = Task("t2", "UI", ["react"], ["fe"], "fe")
        p1 = Person("p1", "Alice", ["python"], ["ml"])
        rows = match_all([t1, t2], [p1], SlowClient(api_key="sk-test"), workers=2)
        self.assertEqual([row["task"] for row in rows], ["t1", "t2"])

    def test_on_success_writes_finished_rows_only(self):
        seen = []

        class SlowClient(LLMClient):
            def call(self, payload, task=None, people=None):
                time.sleep(0.05 if task.id == "t1" else 0.0)
                if task.id == "t1":
                    raise ValueError("bad")
                return {"task": task.id, "people": people[0].id, "reason": "ok"}

        t1 = Task("t1", "RAG", ["python"], ["ml"], "ml")
        t2 = Task("t2", "UI", ["react"], ["fe"], "fe")
        p1 = Person("p1", "Alice", ["python"], ["ml"])
        rows = match_all(
            [t1, t2],
            [p1],
            SlowClient(api_key="sk-test"),
            workers=2,
            on_success=seen.append,
        )
        self.assertEqual([row["task"] for row in seen], ["t2"])
        self.assertTrue(rows[0]["error"])
        self.assertEqual(rows[1]["task"], "t2")


if __name__ == "__main__":
    unittest.main()

