import csv
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Task:
    id: str
    title: str
    required_skills: list
    required_experiences: list
    domain: str


@dataclass
class Person:
    id: str
    name: str
    skills: list
    experiences: list


def _require(row, key):
    value = (row[key] or "").strip()
    if not value:
        raise ValueError(f"{key} is empty")
    return value


def _split(value):
    return [part.strip() for part in value.split("|") if part.strip()]


def parse_csv(path, from_row, kind):
    records = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            try:
                records.append(from_row(row))
            except (KeyError, ValueError) as exc:
                log.error("skipping bad %s at line %s: %s", kind, line_no, exc)
    log.info("parsed %s valid %s", len(records), kind)
    return records


def _task_from_row(row):
    return Task(
        id=_require(row, "id"),
        title=_require(row, "title"),
        required_skills=_split(_require(row, "required_skills")),
        required_experiences=_split(_require(row, "required_experiences")),
        domain=_require(row, "domain"),
    )


def _person_from_row(row):
    return Person(
        id=_require(row, "id"),
        name=_require(row, "name"),
        skills=_split(_require(row, "skills")),
        experiences=_split(_require(row, "experiences")),
    )


def parse_tasks(path):
    return parse_csv(path, _task_from_row, "task")


def parse_people(path):
    return parse_csv(path, _person_from_row, "person")

