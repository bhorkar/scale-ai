import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


def match_one(llm, task, people):
    log.info("match task %s", task.id)
    try:
        payload = llm.build_payload(llm.build_prompt(task, people))
        return llm.call(payload, task=task, people=people)
    except Exception as exc:
        log.warning("skip task %s: %s", task.id, exc)
        return {"error": str(exc), "task_id": task.id}


def match_all(tasks, people, llm, workers=None, on_success=None):
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
            row = future.result()
            results[index] = row
            if on_success and not row.get("error"):
                log.info("write success task %s", row.get("task"))
                on_success(row)
    elapsed = time.perf_counter() - started
    log.info(
        "match_all tasks=%s workers=%s elapsed=%.3fs",
        len(tasks),
        workers,
        elapsed,
    )
    return results

