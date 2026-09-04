import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Person:
    id: str
    name: str
    skills: list[str]
    experience: int

@dataclass
class Task:
    id: str
    title: str
    description: str


def parse_person(row: dict, line_number: int) -> Person:
    missing = [
        field
        for field in ["id", "name", "skills", "experience"]
        if not (row.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(f"line {line_number}: missing {missing}")
    return Person(
        id=row["id"].strip(),
        name=row["name"].strip(),
        skills=[part.strip() for part in row["skills"].split(",") if part.strip()],
        experience=int(row["experience"].strip()),
    )


def parse_task(row: dict, line_number: int) -> Task:
    missing = [
        field
        for field in ["id", "title", "description"]
        if not (row.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(f"line {line_number}: missing {missing}")
    return Task(
        id=row["id"].strip(),
        title=row["title"].strip(),
        description=row["description"].strip(),
    )


def load_people(csv_path: str | Path) -> list[Person]:
    people: list[Person] = []
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                people.append(parse_person(row, reader.line_num))
            except ValueError as exc:
                print(f"skipping person: {exc}")
    return people


def load_tasks(csv_path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                tasks.append(parse_task(row, reader.line_num))
            except ValueError as exc:
                print(f"skipping task: {exc}")
    return tasks
