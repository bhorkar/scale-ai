import csv
import logging

from llm_client import LLMClient
from matcher import match_all
from parser import parse_people, parse_tasks

log = logging.getLogger(__name__)


def write_matches(path: str, responses: list[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task_id", "people_id", "explanation"]
        )
        writer.writeheader()
        for row in responses:
            if row.get("error"):
                log.warning("write skip %s", row)
                continue
            writer.writerow(
                {
                    "task_id": row["task"],
                    "people_id": row["people"],
                    "explanation": row["reason"],
                }
            )


def run_matching(tasks_path: str, people_path: str, output_path: str, llm) -> list[dict]:
    tasks = parse_tasks(tasks_path)
    people = parse_people(people_path)
    if not tasks or not people:
        raise ValueError("parsing failed")
    responses = match_all(tasks, people, llm)
    write_matches(output_path, responses)
    return responses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_matching("task.csv", "people.csv", "matches.csv", LLMClient())
~                                                                                                                                                                                                                                                   
~                                                                                                                                                                                                                                                   
~                                                                                      
