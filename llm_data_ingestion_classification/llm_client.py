import logging
import os
import random
import time

import requests

log = logging.getLogger("llm_client")


def build_payload(prompt):
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    log.info("llm payload model=%s", payload["model"])
    return payload


class LLMClient:
    def __init__(
        self,
        base_url=None,
        payload=None,
        post=None,
        api_key=None,
        max_attempts=3,
        backoff=0.1,
        sleep=time.sleep,
        jitter=random.random,
    ):
        self.base_url = base_url or "https://api.openai.com/v1/chat/completions"
        self.payload = payload
        self.post = post or requests.post
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.sleep = sleep
        self.jitter = jitter
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

    def _once(self, body):
        log.info("Logging llm_request url=%s keys=%s", self.base_url, body)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.post(self.base_url, json=body, headers=headers, timeout=30)
        print(response.json())
        response.raise_for_status()
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise ValueError("malformed openai envelope") from err
        if not isinstance(text, str) or not text.strip():
            raise ValueError("malformed openai envelope")
        log.info("llm_response chars=%s", len(text))
        return text

    def call(self, payload=None):
        body = payload if payload is not None else self.payload
        if body is None:
            raise ValueError("payload is required")
        last_err = None
        for attempt in range(self.max_attempts):
            try:
                return self._once(body)
            except ValueError:
                raise
            except Exception as err:
                last_err = err
                log.warning("llm retry attempt=%s/%s error=%s", attempt + 1, self.max_attempts, err)
                if attempt >= self.max_attempts - 1:
                    break
                delay = self.backoff * (2 ** attempt) * self.jitter()
                log.info("llm backoff seconds=%s", delay)
                self.sleep(delay)
        raise last_err
