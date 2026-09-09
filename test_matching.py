import os
import tempfile
import unittest

from parser import parse_people, parse_tasks


class TestParser(unittest.TestCase):
    def test_fixture_csvs(self):
        tasks = parse_tasks("task.csv")
        people = parse_people("people.csv")
        self.assertEqual([t.id for t in tasks], ["t1", "t2", "t3", "t4"])
        self.assertEqual(tasks[0].required_skills, ["python", "nlp"])
        self.assertEqual(people[0].id, "p1")
        self.assertEqual(people[0].skills, ["python", "pytorch", "nlp"])

    def test_skips_bad_row(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write(
            "id,title,required_skills,required_experiences,domain\n"
            "t1,ok,python,ml,ml\n"
            "t2,bad,,x,x\n"
            "t3,ok,go,sre,infra\n"
        )
        handle.close()
        with self.assertLogs("parser", level="ERROR"):
            tasks = parse_tasks(handle.name)
        os.unlink(handle.name)
        self.assertEqual([t.id for t in tasks], ["t1", "t3"])


if __name__ == "__main__":
    unittest.main()
