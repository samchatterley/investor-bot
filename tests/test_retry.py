import unittest

from utils.retry import with_retry


class TestWithRetry(unittest.TestCase):
    def test_succeeds_on_first_attempt(self):
        calls = []

        @with_retry(max_attempts=3, delay=0)
        def fn():
            calls.append(1)
            return "ok"

        result = fn()
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_and_eventually_succeeds(self):
        calls = []

        @with_retry(max_attempts=3, delay=0)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "done"

        result = fn()
        self.assertEqual(result, "done")
        self.assertEqual(len(calls), 3)

    def test_raises_after_all_attempts_exhausted(self):
        @with_retry(max_attempts=3, delay=0)
        def fn():
            raise RuntimeError("always fails")

        with self.assertRaises(RuntimeError):
            fn()

    def test_only_catches_specified_exceptions(self):
        @with_retry(max_attempts=3, delay=0, exceptions=(ValueError,))
        def fn():
            raise TypeError("different error")

        with self.assertRaises(TypeError):
            fn()

    def test_attempt_count_on_exhaustion(self):
        attempts = []

        @with_retry(max_attempts=4, delay=0)
        def fn():
            attempts.append(1)
            raise ValueError("fail")

        with self.assertRaises(ValueError):
            fn()
        self.assertEqual(len(attempts), 4)

    def test_return_value_preserved(self):
        @with_retry(max_attempts=2, delay=0)
        def fn():
            return {"key": 42}

        self.assertEqual(fn(), {"key": 42})

    def test_passes_args_and_kwargs(self):
        @with_retry(max_attempts=2, delay=0)
        def fn(a, b=0):
            return a + b

        self.assertEqual(fn(3, b=4), 7)

    def test_zero_max_attempts_returns_none(self):
        # Branch 20->exit: range(1, 0+1) is empty → for loop exits naturally
        @with_retry(max_attempts=0, delay=0)
        def fn():
            return "never called"  # pragma: no cover

        result = fn()
        self.assertIsNone(result)
