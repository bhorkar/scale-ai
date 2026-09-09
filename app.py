import csv
import json
import logging

from llm_client import LLMClient
from matcher import match
from parser import parse_people, parse_tasks

log = logging.getLogger("app")

MATCH_FIELDS = ["task_id", "people_id", "explanation"]


class MatchingApp:
    def __init__(self, llm, task_parser=parse_tasks, people_parser=parse_people):
        self.llm = llm
        self.task_parser = task_parser
        self.people_parser = people_parser

    @staticmethod
    def write_matches(responses, path):
        rows = []
        for text in responses:
            try:
                data = json.loads(text)
                reason = data["reason"]
                explanation = reason["explanation"] if isinstance(reason, dict) else str(reason)
                rows.append(
                    {
                        "task_id": data["task_id"],
                        "people_id": data["person_id"],
                        "explanation": explanation,
                    }
                )
                log.info("match row task_id=%s people_id=%s", data["task_id"], data["person_id"])
            except (json.JSONDecodeError, KeyError, TypeError) as err:
                log.error("match parse failed error=%s text=%s", err, text)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        log.info("write done path=%s rows=%s", path, len(rows))
        return rows

    def run(self, tasks_path="task.csv", people_path="people.csv", matches_path="matches.csv"):
        log.info("load start tasks=%s people=%s", tasks_path, people_path)
        try:
            tasks = self.task_parser(tasks_path)
            people = self.people_parser(people_path)
        except (OSError, csv.Error, UnicodeError) as err:
            log.error("load failed error=%s", err)
            return [], [], []
        log.info("load done tasks=%s people=%s", len(tasks), len(people))
        responses = match(tasks, people, llm=self.llm)
        log.info("match done responses=%s", len(responses))
        self.write_matches(responses, matches_path)
        return tasks, people, responses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    MatchingApp(LLMClient()).run()
