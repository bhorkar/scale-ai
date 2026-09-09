import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from llm_client import build_payload

log = logging.getLogger("matcher")


class LLMError(Exception):
    pass

RESPONSE_SCHEMA = {
    "task_id": "string",
    "person_id": "string",
    "reason": {
        "matched_skills": ["string"],
        "explanation": "string",
    },
}


def build_prompt(task, people):
    people_payload = [asdict(person) for person in people]
    task_payload = asdict(task)
    log.info("prompt build task_id=%s people=%s", task.id, [person.id for person in people])
    return (
       f""" "Assign this task to exactly one person with the best matching skills.\n"
        "Do not assign more than one person.\n"
        "Use the people list and the task information below.\n"
        f"Task:\n{json.dumps(task_payload)}\n"
        f"People:\n{json.dumps(people_payload)}\n"
        "Respond with JSON only, no extra text, in this shape:\n"
        f"{json.dumps(RESPONSE_SCHEMA)}\n"
        "task_id and person_id are the assigned pair. reason explains the skill match." """
    )


def validate_response(text):
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as err:
        raise LLMError("invalid_json", str(err)) from err
    if not isinstance(data, dict):
        raise LLMError("not_object")
    for key in ("task_id", "person_id"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LLMError("invalid_field", key)
    reason = data.get("reason")
    if not isinstance(reason, dict):
        raise LLMError("invalid_field", "reason")
    explanation = reason.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise LLMError("invalid_field", "explanation")
    skills = reason.get("matched_skills")
    if skills is not None:
        if not isinstance(skills, list) or any(not isinstance(s, str) for s in skills):
            raise LLMError("invalid_field", "matched_skills")
    return data


def _match_one(llm, task, people):
    log.info("match start task_id=%s", task.id)
    text = llm.call(build_payload(build_prompt(task, people)))
    try:
        validate_response(text)
    except LLMError as err:
        log.error("match invalid task_id=%s error=%s", task.id, err)
        return None
    log.info("match done task_id=%s chars=%s", task.id, len(text))
    return text


def match(tasks, people, llm, workers=2):
    log.info("match pool workers=%s tasks=%s", workers, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(lambda task: _match_one(llm, task, people), tasks))
    log.info("match summary tasks=%s responses=%s", len(tasks), len(responses))
    return responses


