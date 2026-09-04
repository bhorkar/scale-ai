import csv
import dataclasses
from pathlib import Path

@dataclasses.dataclass
class Person:
    id: str
    name: str
    skills: list[str]
    experience: int

@dataclasses.dataclass
class Task:
    id: str
    title: str
    description: str


REQUIRED_TASK_FIELDS = ["id", "title", "description"]
FIRST_DATA_LINE = 2
REQUIRED_PERSON_FIELDS = ["id", "name", "skills", "experience"]

def parse_person(row: dict, line_number: int) -> Person:
    missing = [
        field
        for field in REQUIRED_PERSON_FIELDS
        if not (row.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(f"line {line_number}: missing {missing}")
    return Person(
        id=row["id"].strip(),
        name=row["name"].strip(),
        skills=row["skills"].strip().split(","),
        experience=int(row["experience"].strip())
    )

def load_people(csv_path: str | Path) -> list[Person]:
    people: list[Person] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for line_number, row in enumerate(reader, start=FIRST_DATA_LINE):
            try:
                people.append(parse_person(row, line_number))
            except ValueError as exc:
                print(f"skipping person: {exc}")
    return people

def parse_task(row: dict, line_number: int) -> Task:
    missing = [
        field
        for field in REQUIRED_TASK_FIELDS
        if not (row.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(f"line {line_number}: missing {missing}")
    return Task(
        id=row["id"].strip(),
        title=row["title"].strip(),
        description=row["description"].strip(),
    )

def load_tasks(csv_path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for line_number, row in enumerate(reader, start=FIRST_DATA_LINE):
            try:
                tasks.append(parse_task(row, line_number))
            except ValueError as exc:
                print(f"skipping task: {exc}")
    return tasks


def main():
    from matching import match_all, openai_call

    people = load_people("people.csv")
    tasks = load_tasks("tasks.csv")
    for result in match_all(tasks, people, openai_call, k=2, workers=2, batch_size=2):
        print(result)


if __name__ == "__main__":
    main()


