"""Tests for data/finbert.py — lazy-loaded FinBERT sentiment classifier."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import data.finbert as finbert


def _make_pipeline_output(pos: float, neg: float, neu: float) -> list[list[dict]]:
    """Build a mock pipeline return value matching return_all_scores=True format."""
    return [
        [
            {"label": "positive", "score": pos},
            {"label": "negative", "score": neg},
            {"label": "neutral", "score": neu},
        ]
    ]


class TestIsAvailable(unittest.TestCase):
    def setUp(self) -> None:
        # Reset module-level state before every test
        finbert._pipeline = None

    def tearDown(self) -> None:
        finbert._pipeline = None

    def test_returns_false_when_pipeline_is_false(self) -> None:
        finbert._pipeline = False
        self.assertFalse(finbert.is_available())

    def test_triggers_load_and_returns_true_on_success(self) -> None:
        mock_pipe = MagicMock()
        with patch(
            "data.finbert._load_pipeline",
            side_effect=lambda: setattr(finbert, "_pipeline", mock_pipe),
        ):
            result = finbert.is_available()
        self.assertTrue(result)
        self.assertIs(finbert._pipeline, mock_pipe)

    def test_already_loaded_returns_true(self) -> None:
        finbert._pipeline = MagicMock()
        self.assertTrue(finbert.is_available())

    def test_load_fails_returns_false(self) -> None:
        with patch(
            "data.finbert._load_pipeline", side_effect=lambda: setattr(finbert, "_pipeline", False)
        ):
            result = finbert.is_available()
        self.assertFalse(result)


class TestClassifyText(unittest.TestCase):
    def setUp(self) -> None:
        finbert._pipeline = None

    def tearDown(self) -> None:
        finbert._pipeline = None

    def test_returns_none_for_short_text(self) -> None:
        # Under 20 chars stripped
        self.assertIsNone(finbert.classify_text("too short"))

    def test_returns_none_for_whitespace_only(self) -> None:
        self.assertIsNone(finbert.classify_text("   " * 10))

    def test_returns_none_when_pipeline_unavailable(self) -> None:
        finbert._pipeline = False
        long_text = "a" * 25
        self.assertIsNone(finbert.classify_text(long_text))

    def test_returns_none_when_pipeline_fails_to_load(self) -> None:
        with patch(
            "data.finbert._load_pipeline", side_effect=lambda: setattr(finbert, "_pipeline", False)
        ):
            self.assertIsNone(finbert.classify_text("a" * 25))

    def test_correct_dict_for_mocked_pipeline(self) -> None:
        mock_pipe = MagicMock(return_value=_make_pipeline_output(0.8, 0.1, 0.1))
        finbert._pipeline = mock_pipe
        result = finbert.classify_text(
            "Revenue beat consensus expectations significantly this quarter"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "positive")
        self.assertAlmostEqual(result["positive"], 0.8)
        self.assertAlmostEqual(result["negative"], 0.1)
        self.assertAlmostEqual(result["neutral"], 0.1)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_negative_label_wins(self) -> None:
        mock_pipe = MagicMock(return_value=_make_pipeline_output(0.1, 0.8, 0.1))
        finbert._pipeline = mock_pipe
        result = finbert.classify_text("Revenue missed expectations significantly this quarter")
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "negative")

    def test_neutral_label_wins(self) -> None:
        mock_pipe = MagicMock(return_value=_make_pipeline_output(0.1, 0.1, 0.8))
        finbert._pipeline = mock_pipe
        result = finbert.classify_text(
            "Revenue results were broadly in line with analyst estimates"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "neutral")

    def test_returns_none_on_pipeline_exception(self) -> None:
        mock_pipe = MagicMock(side_effect=RuntimeError("cuda error"))
        finbert._pipeline = mock_pipe
        result = finbert.classify_text(
            "Revenue beat consensus expectations significantly this quarter"
        )
        self.assertIsNone(result)

    def test_text_truncated_to_2000_chars(self) -> None:
        mock_pipe = MagicMock(return_value=_make_pipeline_output(0.6, 0.2, 0.2))
        finbert._pipeline = mock_pipe
        long_text = "x" * 5000
        finbert.classify_text(long_text)
        call_arg = mock_pipe.call_args[0][0]
        self.assertEqual(len(call_arg), 2000)

    def test_scores_normalised_when_not_summing_to_one(self) -> None:
        # Scores sum to 2.0 — should be normalised
        mock_pipe = MagicMock(return_value=_make_pipeline_output(1.6, 0.2, 0.2))
        finbert._pipeline = mock_pipe
        result = finbert.classify_text(
            "Revenue beat consensus expectations significantly this quarter"
        )
        self.assertIsNotNone(result)
        total = result["positive"] + result["negative"] + result["neutral"]
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertAlmostEqual(result["positive"], 1.6 / 2.0)

    def test_triggers_load_when_pipeline_none(self) -> None:
        mock_pipe = MagicMock(return_value=_make_pipeline_output(0.7, 0.2, 0.1))
        with patch(
            "data.finbert._load_pipeline",
            side_effect=lambda: setattr(finbert, "_pipeline", mock_pipe),
        ):
            result = finbert.classify_text(
                "Earnings per share exceeded analyst consensus forecasts today"
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "positive")


class TestClassifyTexts(unittest.TestCase):
    def setUp(self) -> None:
        finbert._pipeline = None

    def tearDown(self) -> None:
        finbert._pipeline = None

    def test_returns_one_result_per_input(self) -> None:
        mock_pipe = MagicMock(return_value=_make_pipeline_output(0.7, 0.2, 0.1))
        finbert._pipeline = mock_pipe
        texts = [
            "Revenue beat expectations by a wide margin this quarter",
            "short",
            "Earnings per share missed analyst consensus estimates significantly",
        ]
        results = finbert.classify_texts(texts)
        self.assertEqual(len(results), 3)
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])  # too short
        self.assertIsNotNone(results[2])

    def test_empty_list_returns_empty(self) -> None:
        results = finbert.classify_texts([])
        self.assertEqual(results, [])

    def test_all_none_when_pipeline_unavailable(self) -> None:
        finbert._pipeline = False
        texts = ["a" * 25, "b" * 25]
        results = finbert.classify_texts(texts)
        self.assertEqual(results, [None, None])


class TestLoadPipeline(unittest.TestCase):
    def setUp(self) -> None:
        finbert._pipeline = None

    def tearDown(self) -> None:
        finbert._pipeline = None

    def test_sets_false_on_import_error(self) -> None:
        with patch("builtins.__import__", side_effect=ImportError("no transformers")):
            finbert._load_pipeline()
        self.assertIs(finbert._pipeline, False)

    def test_sets_pipeline_on_success(self) -> None:
        mock_pipe_instance = MagicMock()
        fake_transformers = MagicMock()
        fake_transformers.pipeline.return_value = mock_pipe_instance
        import sys

        original = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        try:
            finbert._load_pipeline()
        finally:
            if original is None:
                sys.modules.pop("transformers", None)
            else:  # pragma: no cover
                sys.modules["transformers"] = original
        self.assertIs(finbert._pipeline, mock_pipe_instance)

    def test_sets_false_on_pipeline_creation_error(self) -> None:
        """If transformers imports but pipeline() raises, _pipeline must be False."""
        fake_transformers = MagicMock()
        fake_transformers.pipeline.side_effect = OSError("model not found")
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            # Re-run _load_pipeline with patched sys.modules so the import succeeds
            # but the pipeline call raises
            import sys

            original = sys.modules.get("transformers")
            sys.modules["transformers"] = fake_transformers
            try:
                finbert._load_pipeline()
            finally:
                if original is None:  # pragma: no cover
                    sys.modules.pop("transformers", None)
                else:
                    sys.modules["transformers"] = original
        self.assertIs(finbert._pipeline, False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
