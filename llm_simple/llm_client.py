import json
import logging
import os

import requests

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key=None,
        url=None,
        model="gpt-4o-mini",
        post=requests.post,
        timeout=30,
        workers=1,
        retries=3,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.url = url or "https://api.openai.com/v1/chat/completions"
        self.model = model
        self.post = post
        self.timeout = timeout
        self.workers = workers
        self.retries = retries

    def build_prompt(self, task, people):
        return (
            "Match the task to the best person. "
            "Return a JSON object with keys task, people, and reason. "
            "task is the task id. people is one person id from the list. "
            "Reason explains the skills match between the task and the person.\n"
            f"Task: {task}\n"
            f"People: {people}"
        )

    def build_payload(self, prompt):
        if not prompt.strip():
            raise ValueError("prompt is empty")
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }

    def validate_content(self, text, task, people):
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            raise ValueError("content is not json")
        if not isinstance(data, dict):
            raise ValueError("content is not a json object")
        for key in ("task", "people", "reason"):
            if not data.get(key):
                raise ValueError(f"missing {key}")
        if data["task"] != task.id:
            raise ValueError("task does not match")
        allowed = [person.id for person in people]
        if data["people"] not in allowed:
            raise ValueError("people not in list")
        return data

    def call(self, payload, task=None, people=None):
        if not payload:
            raise ValueError("payload is missing")
        if not self.api_key:
            raise ValueError("api_key is missing")
        last_error = "llm request failed"
        for attempt in range(self.retries):
            response = self.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            status = response.status_code
            if status == 200:
                text = response.json()["choices"][0]["message"]["content"].strip()
                try:
                    return self.validate_content(text, task, people)
                except Exception as exc:
                    raise ValueError(str(exc))
            last_error = response.text or f"llm http {status}"
            if status == 429 or status >= 500:
                log.warning(
                    "retry attempt=%s/%s status=%s",
                    attempt + 1,
                    self.retries,
                    status,
                )
                continue
            raise ValueError(last_error)
        raise ValueError(last_error)

