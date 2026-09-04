import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from parsing import Person, Task


@dataclass
class Match:
    task_id: str
    person_id: str
    reason: str


def build_prompt(task: Task, people: list[Person]) -> str:
    return (
        "Pick exactly one candidate for this task. Do not invent people.\n"
        'Return JSON only: {"person_id": "...", "reason": "..."}\n\n'
        f"TASK:\n{json.dumps(asdict(task))}\n\n"
        f"CANDIDATES:\n{json.dumps([asdict(p) for p in people])}\n"
    )


def _openai_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def openai_call(prompt: str, retries: int = 3) -> str:
    """POST prompt to OpenAI. Retry only on 429 or timeout."""
    print(f"openai request start prompt_chars={len(prompt)}")
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            data = _openai_json("https://api.openai.com/v1/chat/completions", payload)
            text = data["choices"][0]["message"]["content"]
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


def person_text(person: Person) -> str:
    return " ".join([person.name, *person.skills])


def task_text(task: Task) -> str:
    return f"{task.title} {task.description}"


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(a @ b / denom)


def bag_of_words_embed(text: str, size: int = 32) -> np.ndarray:
    vec = np.zeros(size)
    for word in text.lower().replace(",", " ").split():
        vec[sum(map(ord, word)) % size] += 1
    return vec


def openai_embed(text: str) -> list[float]:
    print(f"openai embed start chars={len(text)}")
    data = _openai_json(
        "https://api.openai.com/v1/embeddings",
        {
            "model": os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            "input": text,
        },
    )
    return data["data"][0]["embedding"]


def shortlist(
    task: Task,
    people: list[Person],
    k: int = 3,
    embed_fn=None,
) -> list[Person]:
    """Top-k people by cosine(person embedding, task embedding)."""
    embed = embed_fn or bag_of_words_embed
    if not people:
        print(f"shortlist task={task.id} ids=[]")
        return []
    task_vec = np.asarray(embed(task_text(task)), float)
    matrix = np.asarray([embed(person_text(p)) for p in people], float)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(task_vec)
    sims = np.divide(matrix @ task_vec, denom, out=np.zeros(len(people)), where=denom != 0)
    order = np.argsort(-sims)[:k]
    chosen = [people[i] for i in order]
    print(f"shortlist task={task.id} ids={[p.id for p in chosen]} cosine={np.round(sims[order], 3).tolist()}")
    return chosen


def match_person_to_task(task: Task, people: list[Person], llm_call, k: int = 3, embed_fn=None) -> Match:
    candidates = shortlist(task, people, k, embed_fn=embed_fn)
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


def _result_for_task(task: Task, people: list[Person], llm_call, k: int, embed_fn) -> dict:
    try:
        match = match_person_to_task(task, people, llm_call, k, embed_fn=embed_fn)
        return {"task_id": match.task_id, "person_id": match.person_id, "reason": match.reason, "error": None}
    except Exception as exc:
        print(f"failed task={task.id} error={exc}")
        return {"task_id": task.id, "person_id": None, "reason": None, "error": str(exc)}


def match_all(
    tasks: list[Task],
    people: list[Person],
    llm_call,
    k: int = 3,
    workers: int = 2,
    batch_size: int = 2,
    done_ids: set[str] | None = None,
    embed_fn=None,
) -> list[dict]:
    """Score a batch in parallel. Assign 1:1 in task order. Skip done_ids."""
    checkpoint = done_ids if done_ids is not None else set()
    pending = [task for task in tasks if task.id not in checkpoint]
    assigned: set[str] = set()
    print(
        f"match_all start pending={len(pending)} skipped={len(tasks) - len(pending)} "
        f"k={k} workers={workers} batch_size={batch_size}"
    )
    results = []
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        print(f"batch start offset={offset} tasks={[task.id for task in batch]}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_result_for_task, task, people, llm_call, k, embed_fn) for task in batch]
            slots = [fut.result() for fut in futs]
        for task, row in zip(batch, slots):
            if row["error"] is None and row["person_id"] in assigned:
                remaining = [person for person in people if person.id not in assigned]
                print(f"collision task={task.id} person={row['person_id']} remaining={[p.id for p in remaining]}")
                if remaining:
                    row = _result_for_task(task, remaining, llm_call, k, embed_fn)
                else:
                    row = {"task_id": task.id, "person_id": None, "reason": None, "error": "no free person"}
            results.append(row)
            if row["error"] is None:
                checkpoint.add(row["task_id"])
                assigned.add(row["person_id"])
            print(f"commit task={row['task_id']} person={row['person_id']}")
    print(f"match_all done total={len(results)}")
    return results
