import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from parsing import Person, Task


@dataclass
class Match:
    task_id: str
    person_id: str
    reason: str


def build_prompt(task: Task, people: list[Person]) -> str:
    task_json = {
        "id": task.id,
        "title": task.title,
        "description": task.description,
    }
    people_json = [
        {
            "id": person.id,
            "name": person.name,
            "skills": person.skills,
            "experience": person.experience,
        }
        for person in people
    ]
    return (
        "Pick exactly one candidate for this task. Do not invent people.\n"
        'Return JSON only: {"person_id": "...", "reason": "..."}\n\n'
        f"TASK:\n{json.dumps(task_json)}\n\n"
        f"CANDIDATES:\n{json.dumps(people_json)}\n"
    )


def openai_call(prompt: str, retries: int = 3) -> str:
    """POST prompt to OpenAI. Retry only on 429 or timeout."""
    print(f"openai request start prompt_chars={len(prompt)}")
    body = json.dumps(
        {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    last_error = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
            text = payload["choices"][0]["message"]["content"]
            print(f"openai response: {text}")
            return text
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == retries:
                raise
            print(f"openai retry attempt={attempt} status=429")
            time.sleep(0.05 * attempt)
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == retries:
                raise
            print(f"openai retry attempt={attempt} timeout")
            time.sleep(0.05 * attempt)
    raise last_error


def parse_llm_json(raw):
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def validate_match(match: Match, people: list[Person]) -> None:
    allowed = {person.id for person in people}
    if match.person_id not in allowed:
        raise ValueError(f"unknown person_id {match.person_id}")


def shortlist(task: Task, people: list[Person], k: int = 3) -> list[Person]:
    """Top-k people by skill-token overlap with the task. No LLM."""
    desc = set((task.title + " " + task.description).lower().replace(",", " ").split())
    scored = []
    for person in people:
        skills = set(" ".join(person.skills).lower().replace(",", " ").split())
        scored.append((len(skills & desc), person))
    scored.sort(key=lambda item: -item[0])
    chosen = [person for _, person in scored[:k]]
    print(
        f"shortlist task={task.id} ids={[p.id for p in chosen]} "
        f"scores={[score for score, _ in scored[:k]]}"
    )
    return chosen


def match_person_to_task(task: Task, people: list[Person], llm_call, k: int = 3) -> Match:
    candidates = shortlist(task, people, k)
    print(f"matching task={task.id}")
    raw = llm_call(build_prompt(task, candidates))
    print(f"llm raw: {raw}")
    try:
        data = parse_llm_json(raw)
        match = Match(task.id, str(data["person_id"]), str(data["reason"]))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"task {task.id}: invalid LLM output: {exc}") from exc
    validate_match(match, candidates)
    print(f"matched task={match.task_id} person={match.person_id}")
    return match


def _result_for_task(task: Task, people: list[Person], llm_call, k: int) -> dict:
    try:
        match = match_person_to_task(task, people, llm_call, k)
        return {
            "task_id": match.task_id,
            "person_id": match.person_id,
            "reason": match.reason,
            "error": None,
        }
    except Exception as exc:
        print(f"failed task={task.id} error={exc}")
        return {
            "task_id": task.id,
            "person_id": None,
            "reason": None,
            "error": str(exc),
        }


def match_all(
    tasks: list[Task],
    people: list[Person],
    llm_call,
    k: int = 3,
    workers: int = 2,
    batch_size: int = 2,
    done_ids: set[str] | None = None,
) -> list[dict]:
    """One LLM call per task. Shortlist, then batched threads. Skip done_ids (checkpoint)."""
    done_ids = set(done_ids or [])
    pending = [task for task in tasks if task.id not in done_ids]
    print(
        f"match_all start pending={len(pending)} skipped={len(tasks) - len(pending)} "
        f"k={k} workers={workers} batch_size={batch_size}"
    )
    results = []
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        print(f"batch start offset={offset} tasks={[task.id for task in batch]}")
        slots = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_result_for_task, task, people, llm_call, k): idx
                for idx, task in enumerate(batch)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                slots[idx] = fut.result()
                print(f"worker done task={batch[idx].id}")
        for row in slots:
            results.append(row)
            done_ids.add(row["task_id"])
            print(f"commit task={row['task_id']}")
    print(f"match_all done total={len(results)}")
    return results
