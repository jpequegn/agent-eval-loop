"""Tests for failure-mode detectors."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from eval.detectors import (
    COLLAPSE_VARIANCE_THRESHOLD,
    COLLAPSE_WINDOW,
    COHERENCE_CHECK_INTERVAL,
    COHERENCE_LOW_THRESHOLD,
    GAMING_SCORE_RISE_THRESHOLD,
    GAMING_WINDOW,
    AnomalyKind,
    CritiqueCollapseDetector,
    DetectorSuite,
    PromptCoherenceChecker,
    ScoreGamingDetector,
)


def _mock_client(response_json: dict) -> MagicMock:
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json.dumps(response_json))]
    mock.messages.create.return_value = mock_resp
    return mock


# ---------------------------------------------------------------------------
# CritiqueCollapseDetector
# ---------------------------------------------------------------------------

class TestCritiqueCollapseDetector:
    def _det(self) -> CritiqueCollapseDetector:
        return CritiqueCollapseDetector()

    def test_no_anomaly_with_insufficient_history(self) -> None:
        det = self._det()
        scores = [[3.0, 4.0], [3.5, 4.5]]   # only 2 iterations
        assert det.check(2, scores) is None

    def test_no_anomaly_when_variance_high(self) -> None:
        det = self._det()
        # Wide spread → high variance
        scores = [[1.0, 5.0, 1.0, 5.0]] * COLLAPSE_WINDOW
        assert det.check(COLLAPSE_WINDOW, scores) is None

    def test_anomaly_when_all_scores_identical(self) -> None:
        det = self._det()
        # All episodes score exactly 4.0 → zero variance
        scores = [[4.0, 4.0, 4.0, 4.0]] * COLLAPSE_WINDOW
        result = det.check(COLLAPSE_WINDOW, scores)
        assert result is not None
        assert result.kind == AnomalyKind.CRITIQUE_COLLAPSE

    def test_anomaly_when_variance_just_below_threshold(self) -> None:
        det = self._det()
        tiny = COLLAPSE_VARIANCE_THRESHOLD / 2
        scores = [[4.0, 4.0 + tiny]] * COLLAPSE_WINDOW
        result = det.check(COLLAPSE_WINDOW, scores)
        assert result is not None

    def test_no_anomaly_single_episode_per_iteration(self) -> None:
        # Can't compute stdev with one data point — should skip safely
        det = self._det()
        scores = [[4.0]] * COLLAPSE_WINDOW
        assert det.check(COLLAPSE_WINDOW, scores) is None

    def test_anomaly_message_contains_threshold(self) -> None:
        det = self._det()
        scores = [[4.0, 4.0, 4.0]] * COLLAPSE_WINDOW
        result = det.check(COLLAPSE_WINDOW, scores)
        assert str(COLLAPSE_VARIANCE_THRESHOLD) in result.message

    def test_uses_only_last_window_iterations(self) -> None:
        det = self._det()
        # First 5 iterations have high variance, last COLLAPSE_WINDOW have low
        high_var = [1.0, 5.0, 1.0, 5.0]
        low_var = [4.0, 4.0, 4.0, 4.0]
        scores = [high_var] * 5 + [low_var] * COLLAPSE_WINDOW
        result = det.check(len(scores) - 1, scores)
        assert result is not None


# ---------------------------------------------------------------------------
# ScoreGamingDetector
# ---------------------------------------------------------------------------

class TestScoreGamingDetector:
    def test_no_anomaly_insufficient_history(self) -> None:
        det = ScoreGamingDetector(client=MagicMock())
        assert det.check(4, [3.0, 3.5], [3.0, 3.5]) is None

    def test_no_anomaly_both_rising(self) -> None:
        det = ScoreGamingDetector(client=MagicMock())
        auto = [3.0, 3.2, 3.4, 3.6, 3.8]
        spot = [3.0, 3.1, 3.3, 3.5, 3.7]
        assert det.check(4, auto, spot) is None

    def test_no_anomaly_auto_rises_spot_flat(self) -> None:
        det = ScoreGamingDetector(client=MagicMock())
        auto = [3.0, 3.2, 3.4, 3.6, 3.6 + GAMING_SCORE_RISE_THRESHOLD - 0.01]
        spot = [3.0, 3.0, 3.0, 3.0, 3.0]
        # auto rise is just below threshold
        assert det.check(4, auto, spot) is None

    def test_anomaly_when_auto_rises_spot_falls(self) -> None:
        det = ScoreGamingDetector(client=MagicMock())
        auto = [3.0] + [3.0 + GAMING_SCORE_RISE_THRESHOLD + 0.1] * (GAMING_WINDOW - 1)
        spot = [4.0] + [3.5] * (GAMING_WINDOW - 1)
        result = det.check(GAMING_WINDOW - 1, auto, spot)
        assert result is not None
        assert result.kind == AnomalyKind.SCORE_GAMING

    def test_spot_check_parses_response(self) -> None:
        mock_client = _mock_client({"score": 4, "reason": "Good summary."})
        det = ScoreGamingDetector(client=mock_client)
        score, reason = det.spot_check("transcript text", {"key_topics": []})
        assert score == 4.0
        assert "Good" in reason

    def test_spot_check_strips_markdown_fence(self) -> None:
        raw = '```json\n{"score": 3, "reason": "OK."}\n```'
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=raw)]
        mock_client.messages.create.return_value = mock_resp
        det = ScoreGamingDetector(client=mock_client)
        score, _ = det.spot_check("t", {})
        assert score == 3.0


# ---------------------------------------------------------------------------
# PromptCoherenceChecker
# ---------------------------------------------------------------------------

class TestPromptCoherenceChecker:
    def test_should_check_every_n_iterations(self) -> None:
        checker = PromptCoherenceChecker(client=MagicMock())
        assert checker.should_check(COHERENCE_CHECK_INTERVAL - 1) is True
        assert checker.should_check(0) is False
        assert checker.should_check(COHERENCE_CHECK_INTERVAL) is False

    def test_no_anomaly_when_score_high(self) -> None:
        mock_client = _mock_client({"score": 5, "issues": []})
        checker = PromptCoherenceChecker(client=mock_client)
        assert checker.check(9, "Clear prompt.") is None

    def test_anomaly_when_score_low(self) -> None:
        mock_client = _mock_client({"score": 2, "issues": ["contradictory instructions"]})
        checker = PromptCoherenceChecker(client=mock_client)
        result = checker.check(9, "Confusing prompt.")
        assert result is not None
        assert result.kind == AnomalyKind.PROMPT_DRIFT
        assert "contradictory" in result.message

    def test_anomaly_at_threshold_boundary(self) -> None:
        mock_client = _mock_client({"score": COHERENCE_LOW_THRESHOLD - 0.1, "issues": ["vague"]})
        checker = PromptCoherenceChecker(client=mock_client)
        result = checker.check(9, "Some prompt.")
        assert result is not None

    def test_no_anomaly_at_exact_threshold(self) -> None:
        mock_client = _mock_client({"score": COHERENCE_LOW_THRESHOLD, "issues": []})
        checker = PromptCoherenceChecker(client=mock_client)
        assert checker.check(9, "OK prompt.") is None


# ---------------------------------------------------------------------------
# DetectorSuite
# ---------------------------------------------------------------------------

class TestDetectorSuite:
    def _suite(self) -> DetectorSuite:
        return DetectorSuite(
            collapse=CritiqueCollapseDetector(),
            gaming=ScoreGamingDetector(client=MagicMock()),
            coherence=PromptCoherenceChecker(client=_mock_client({"score": 5, "issues": []})),
        )

    def test_run_returns_empty_list_when_no_anomalies(self) -> None:
        suite = self._suite()
        result = suite.run(
            iteration=1,
            recent_episode_scores=[[3.0, 4.0], [3.5, 4.5]],
            avg_scores=[3.5, 4.0],
            spot_check_scores=[3.5, 4.0],
            system_prompt="Good prompt.",
        )
        assert result == []

    def test_run_detects_collapse(self) -> None:
        suite = self._suite()
        low_var = [[4.0, 4.0, 4.0]] * COLLAPSE_WINDOW
        result = suite.run(
            iteration=COLLAPSE_WINDOW,
            recent_episode_scores=low_var,
            avg_scores=[4.0] * COLLAPSE_WINDOW,
            spot_check_scores=[4.0] * COLLAPSE_WINDOW,
            system_prompt="Some prompt.",
        )
        kinds = [a.kind for a in result]
        assert AnomalyKind.CRITIQUE_COLLAPSE in kinds

    def test_coherence_only_checked_at_interval(self) -> None:
        # Mock the coherence checker to count calls
        mock_coherence = MagicMock()
        mock_coherence.should_check.return_value = False
        suite = DetectorSuite(
            collapse=CritiqueCollapseDetector(),
            gaming=ScoreGamingDetector(client=MagicMock()),
            coherence=mock_coherence,
        )
        suite.run(5, [[4.0, 4.0]], [4.0], [4.0], "prompt")
        mock_coherence.check.assert_not_called()
