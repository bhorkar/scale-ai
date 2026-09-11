import csv
import logging

from llm_client import LLMClient
from matcher import match_all
from parser import parse_people, parse_tasks

log = logging.getLogger(__name__)

FIELDS = ["task_id", "people_id", "explanation"]


def write_row(writer, row):
    if row.get("error"):
        log.warning("write skip %s", row)
        return
    writer.writerow(
        {
            "task_id": row["task"],
            "people_id": row["people"],
            "explanation": row["reason"],
        }
    )

def write_matches(path, responses):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in responses:
            write_row(writer, row)


def run_matching(tasks_path, people_path, output_path, llm):
    tasks = parse_tasks(tasks_path)
    people = parse_people(people_path)
    if not tasks or not people:
        raise ValueError("parsing failed")
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        def on_success(row):
            write_row(writer, row)
            handle.flush()

        responses = match_all(tasks, people, llm, on_success=on_success)
    return responses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_matching("task.csv", "people.csv", "matches.csv", LLMClient())
import csv
import logging

from llm_client import LLMClient
from matcher import match_all
from parser import parse_people, parse_tasks

log = logging.getLogger(__name__)

FIELDS = ["task_id", "people_id", "explanation"]


def write_row(writer, row):
    if row.get("error"):
        log.warning("write skip %s", row)
        return
    writer.writerow(
        {
            "task_id": row["task"],
            "people_id": row["people"],
            "explanation": row["reason"],
        }
    )

def write_matches(path, responses):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in responses:
            write_row(writer, row)


def run_matching(tasks_path, people_path, output_path, llm):
    tasks = parse_tasks(tasks_path)
    people = parse_people(people_path)
    if not tasks or not people:
        raise ValueError("parsing failed")
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        def on_success(row):
            write_row(writer, row)
            handle.flush()

        responses = match_all(tasks, people, llm, on_success=on_success)
    return responses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_matching("task.csv", "people.csv", "matches.csv", LLMClient())

