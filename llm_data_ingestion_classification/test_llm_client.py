import unittest

from llm_client import LLMClient


class FakeResponse:
    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakePost:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.attempts = 0

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.attempts += 1
        return self.responses.pop(0)


def success(text):
    return FakeResponse({"choices": [{"message": {"content": text}}]})


def make_client(post):
    return LLMClient(
        post=post,
        api_key="test-key",
        max_attempts=2,
        sleep=lambda _: None,
    )


class TestLLMClient(unittest.TestCase):
    def test_success(self):
        post = FakePost(success("hello"))

        result = make_client(post).call({"model": "test-model"})

        self.assertEqual(result, "hello")
        self.assertEqual(post.attempts, 1)

    def test_failure_after_two_attempts(self):
        post = FakePost(FakeResponse({}, 500), FakeResponse({}, 500))

        with self.assertRaises(RuntimeError):
            make_client(post).call({"model": "test-model"})

        self.assertEqual(post.attempts, 2)

    def test_retries_500_then_succeeds(self):
        post = FakePost(FakeResponse({}, 500), success("done"))

        result = make_client(post).call({"model": "test-model"})

        self.assertEqual(result, "done")
        self.assertEqual(post.attempts, 2)


if __name__ == "__main__":
    unittest.main()
