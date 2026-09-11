import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from parser import Person, Task

log = logging.getLogger(__name__)


def match_one(llm, task: Task, people: list[Person]) -> dict:
    log.info("match task %s", task.id)
    try:
        payload = llm.build_payload(llm.build_prompt(task, people))
        return llm.call(payload, task=task, people=people)
    except Exception as exc:
        log.warning("skip task %s: %s", task.id, exc)
        return {"error": str(exc), "task_id": task.id}


def match_all(
    tasks: list[Task],
    people: list[Person],
    llm,
    workers: int | None = None,
) -> list[dict]:
    workers = workers or getattr(llm, "workers", 1)
    started = time.perf_counter()
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(match_one, llm, task, people): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(pending):
            index = pending[future]
            results[index] = future.result()
    return results
