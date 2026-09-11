import logging
import tempfile
import unittest
from pathlib import Path

from parser import _require, _split, parse_people, parse_tasks

HERE = Path(__file__).parent


def _tmp_csv(text):
    path = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv")
    path.write(text)
    path.close()
    return path.name


class TestParser(unittest.TestCase):
    def test_fixture_ids_and_parsed_list_fields(self):
        tasks = parse_tasks(HERE / "task.csv")
        people = parse_people(HERE / "people.csv")
        self.assertEqual([task.id for task in tasks], ["t1", "t2", "t3", "t4"])
        self.assertEqual([person.id for person in people], ["p1", "p2", "p3", "p4"])
        self.assertEqual(tasks[0].required_skills, ["python", "nlp"])
        self.assertEqual(tasks[0].required_experiences, ["ML engineer", "NLP research"])
        self.assertEqual(people[0].skills, ["python", "pytorch", "nlp"])
        self.assertEqual(
            people[0].experiences,
            ["ML engineer at Stripe (4y)", "NLP intern at Stanford"],
        )

    def test_skips_invalid_row_logs_error_line_keeps_later(self):
        path = _tmp_csv(
            "id,title,required_skills,required_experiences,domain\n"
            "t1,Ok,python,ml,ml\n"
            ",Missing id,python,ml,ml\n"
            "t3,Also ok,go,sre,infra\n"
        )
        with self.assertLogs("parser", level=logging.ERROR) as logs:
            tasks = parse_tasks(path)
        self.assertEqual([task.id for task in tasks], ["t1", "t3"])
        self.assertTrue(any("line 3" in line for line in logs.output))
        self.assertTrue(any(line.startswith("ERROR:") for line in logs.output))

    def test_missing_field_skips_row(self):
        path = _tmp_csv(
            "id,title,required_skills,required_experiences\n"
            "t1,Ok,python,ml\n"
        )
        with self.assertLogs("parser", level=logging.ERROR) as logs:
            tasks = parse_tasks(path)
        self.assertEqual(tasks, [])
        self.assertTrue(any("line 2" in line for line in logs.output))

    def test_whitespace_trim_and_pipe_list(self):
        self.assertEqual(_require({"id": " t1 "}, "id"), "t1")
        self.assertEqual(_split(" python | nlp | "), ["python", "nlp"])
        path = _tmp_csv(
            "id,title,required_skills,required_experiences,domain\n"
            " t2 ,Title, react | typescript ,frontend,fe\n"
        )
        tasks = parse_tasks(path)
        self.assertEqual(tasks[0].id, "t2")
        self.assertEqual(tasks[0].required_skills, ["react", "typescript"])


if __name__ == "__main__":
    unittest.main()

