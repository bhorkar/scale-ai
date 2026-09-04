import tempfile
import unittest
from pathlib import Path

from parsing import Person, load_people, parse_person


def write_csv(content: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write(content)
    handle.close()
    return Path(handle.name)


class TestParsePerson(unittest.TestCase):
    def test_load_people_happy_path_matches_expected_row(self):
        path = write_csv(
            "id,name,skills,experience\n"
            '101,Alice,"Python, backend",5\n'
        )
        people = load_people(path)
        self.assertEqual(
            people,
            [Person(id="101", name="Alice", skills=["Python", "backend"], experience=5)],
        )

    def test_load_people_skips_bad_row_and_keeps_good_row(self):
        path = write_csv(
            "id,name,skills,experience\n"
            ",Alice,Python,5\n"
            "102,Bob,python,7\n"
        )
        people = load_people(path)
        self.assertEqual(
            people,
            [Person(id="102", name="Bob", skills=["python"], experience=7)],
        )

    def test_empty_id_raises_value_error(self):
        row = {"id": "", "name": "Alice", "skills": "Python", "experience": "5"}
        with self.assertRaises(ValueError):
            parse_person(row, line_number=2)

    def test_non_integer_experience_raises_value_error(self):
        row = {"id": "101", "name": "Alice", "skills": "Python", "experience": "five"}
        with self.assertRaises(ValueError):
            parse_person(row, line_number=2)


if __name__ == "__main__":
    unittest.main()
