"""Tests for the skill registry, skill definitions, and SkillAgent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.skill_agent import SkillAgent
from agents.task_agent import TaskResult
from eval.skills import (
    BUILTIN_SKILLS,
    PODCAST_SUMMARIZER_V1,
    QUOTE_FINDER_V1,
    TOPIC_EXTRACTOR_V1,
    BenchmarkCase,
    BenchmarkResult,
    Skill,
    SkillRegistry,
    run_benchmarks,
)


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------

class TestSkill:
    def test_id_combines_name_and_version(self) -> None:
        s = Skill(
            name="my-skill", version="v2", model="claude-sonnet-4-6",
            system_prompt="p", output_schema={}, benchmarks=[], gotchas=[],
        )
        assert s.id == "my-skill-v2"

    def test_round_trip_serialization(self) -> None:
        skill = PODCAST_SUMMARIZER_V1
        restored = Skill.from_dict(skill.to_dict())
        assert restored.id == skill.id
        assert restored.system_prompt == skill.system_prompt
        assert len(restored.benchmarks) == len(skill.benchmarks)
        assert len(restored.gotchas) == len(skill.gotchas)

    def test_from_dict_preserves_benchmarks(self) -> None:
        d = TOPIC_EXTRACTOR_V1.to_dict()
        restored = Skill.from_dict(d)
        assert restored.benchmarks[0].name == TOPIC_EXTRACTOR_V1.benchmarks[0].name

    def test_to_dict_contains_all_fields(self) -> None:
        d = QUOTE_FINDER_V1.to_dict()
        for key in ("name", "version", "model", "system_prompt", "output_schema", "benchmarks", "gotchas"):
            assert key in d


# ---------------------------------------------------------------------------
# Built-in skills
# ---------------------------------------------------------------------------

class TestBuiltinSkills:
    @pytest.mark.parametrize("skill", BUILTIN_SKILLS)
    def test_has_system_prompt(self, skill: Skill) -> None:
        assert len(skill.system_prompt) > 50

    @pytest.mark.parametrize("skill", BUILTIN_SKILLS)
    def test_has_gotchas(self, skill: Skill) -> None:
        assert len(skill.gotchas) >= 3, f"{skill.id} needs at least 3 gotchas"

    @pytest.mark.parametrize("skill", BUILTIN_SKILLS)
    def test_has_benchmarks(self, skill: Skill) -> None:
        assert len(skill.benchmarks) >= 1

    @pytest.mark.parametrize("skill", BUILTIN_SKILLS)
    def test_output_schema_has_required_field(self, skill: Skill) -> None:
        assert "required" in skill.output_schema or "properties" in skill.output_schema

    def test_podcast_summarizer_schema_keys(self) -> None:
        required = PODCAST_SUMMARIZER_V1.output_schema["required"]
        for key in ("key_topics", "main_argument", "notable_quotes", "actionable_takeaways"):
            assert key in required

    def test_topic_extractor_schema_has_topics(self) -> None:
        assert "topics" in TOPIC_EXTRACTOR_V1.output_schema["required"]

    def test_quote_finder_schema_has_quotes(self) -> None:
        assert "quotes" in QUOTE_FINDER_V1.output_schema["required"]


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistry:
    def test_default_registry_has_three_builtin_skills(self) -> None:
        reg = SkillRegistry()
        assert len(reg.list_ids()) == 3

    def test_get_returns_skill(self) -> None:
        reg = SkillRegistry()
        skill = reg.get("podcast-summarizer-v1")
        assert skill.name == "podcast-summarizer"

    def test_get_raises_on_unknown_id(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent-v99")

    def test_register_adds_skill(self) -> None:
        reg = SkillRegistry()
        new_skill = Skill(
            name="custom", version="v1", model="claude-haiku-4-5-20251001",
            system_prompt="Do something.", output_schema={}, benchmarks=[], gotchas=["none"],
        )
        reg.register(new_skill)
        assert "custom-v1" in reg.list_ids()

    def test_list_ids_sorted(self) -> None:
        reg = SkillRegistry()
        ids = reg.list_ids()
        assert ids == sorted(ids)

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        reg = SkillRegistry()
        path = tmp_path / "skills.json"
        reg.save(path)

        loaded = SkillRegistry.load(path)
        assert loaded.list_ids() == reg.list_ids()

    def test_loaded_skills_preserve_gotchas(self, tmp_path: Path) -> None:
        reg = SkillRegistry()
        path = tmp_path / "skills.json"
        reg.save(path)
        loaded = SkillRegistry.load(path)
        orig = reg.get("podcast-summarizer-v1")
        restored = loaded.get("podcast-summarizer-v1")
        assert restored.gotchas == orig.gotchas

    def test_save_produces_valid_json(self, tmp_path: Path) -> None:
        reg = SkillRegistry()
        path = tmp_path / "skills.json"
        reg.save(path)
        data = json.loads(path.read_text())
        assert "podcast-summarizer-v1" in data


# ---------------------------------------------------------------------------
# run_benchmarks
# ---------------------------------------------------------------------------

class TestRunBenchmarks:
    def test_passes_when_all_keys_present(self) -> None:
        skill = PODCAST_SUMMARIZER_V1
        output = {
            "key_topics": ["AI"],
            "main_argument": "Agents are the future.",
            "notable_quotes": ["I just prompted it."],
            "actionable_takeaways": ["Build in public"],
        }
        results = run_benchmarks(skill, output)
        assert all(r.passed for r in results)

    def test_fails_when_key_missing(self) -> None:
        skill = PODCAST_SUMMARIZER_V1
        output = {"key_topics": ["AI"]}   # missing 3 keys
        results = run_benchmarks(skill, output)
        failing = [r for r in results if not r.passed]
        assert len(failing) >= 1
        assert any("missing key" in r.reason for r in failing)

    def test_fails_when_list_empty(self) -> None:
        skill = PODCAST_SUMMARIZER_V1
        output = {
            "key_topics": [],   # empty list
            "main_argument": "Something",
            "notable_quotes": ["q"],
            "actionable_takeaways": ["a"],
        }
        results = run_benchmarks(skill, output)
        failing = [r for r in results if not r.passed]
        assert any("empty" in r.reason for r in failing)

    def test_fails_when_string_empty(self) -> None:
        skill = PODCAST_SUMMARIZER_V1
        output = {
            "key_topics": ["AI"],
            "main_argument": "",   # empty string
            "notable_quotes": ["q"],
            "actionable_takeaways": ["a"],
        }
        results = run_benchmarks(skill, output)
        failing = [r for r in results if not r.passed]
        assert any("empty" in r.reason for r in failing)

    def test_result_has_skill_id(self) -> None:
        results = run_benchmarks(TOPIC_EXTRACTOR_V1, {"topics": [{"topic": "AI"}]})
        assert all(r.skill_id == "topic-extractor-v1" for r in results)


# ---------------------------------------------------------------------------
# SkillAgent
# ---------------------------------------------------------------------------

def _mock_task_result(output: dict) -> MagicMock:
    r = MagicMock(spec=TaskResult)
    r.output = output
    r.input_tokens = 100
    r.output_tokens = 50
    r.latency_ms = 200
    r.cost_usd = 0.001
    r.model = "claude-sonnet-4-6"
    return r


class TestSkillAgent:
    def test_loads_skill_from_registry(self) -> None:
        reg = SkillRegistry()
        agent = SkillAgent.from_registry("podcast-summarizer-v1", registry=reg)
        assert agent.skill.name == "podcast-summarizer"

    def test_system_prompt_matches_skill(self) -> None:
        agent = SkillAgent(skill=PODCAST_SUMMARIZER_V1)
        assert agent.system_prompt == PODCAST_SUMMARIZER_V1.system_prompt

    def test_system_prompt_is_mutable(self) -> None:
        agent = SkillAgent(skill=PODCAST_SUMMARIZER_V1)
        agent.system_prompt = "New prompt"
        assert agent.system_prompt == "New prompt"

    def test_run_delegates_to_task_agent(self) -> None:
        agent = SkillAgent(skill=PODCAST_SUMMARIZER_V1)
        good_output = {
            "key_topics": ["AI"], "main_argument": "Agents.",
            "notable_quotes": ["quote"], "actionable_takeaways": ["act"],
        }
        agent._agent = MagicMock()
        agent._agent.run.return_value = _mock_task_result(good_output)

        result = agent.run("transcript text")
        agent._agent.run.assert_called_once_with("transcript text")
        assert result.output == good_output

    def test_run_with_benchmark_returns_results(self) -> None:
        agent = SkillAgent(skill=PODCAST_SUMMARIZER_V1)
        good_output = {
            "key_topics": ["AI"], "main_argument": "Agents.",
            "notable_quotes": ["quote"], "actionable_takeaways": ["act"],
        }
        agent._agent = MagicMock()
        agent._agent.run.return_value = _mock_task_result(good_output)

        task_result, bench_results = agent.run_with_benchmark("transcript text")
        assert all(r.passed for r in bench_results)

    def test_run_with_benchmark_catches_missing_keys(self) -> None:
        agent = SkillAgent(skill=PODCAST_SUMMARIZER_V1)
        bad_output = {"key_topics": ["AI"]}   # incomplete
        agent._agent = MagicMock()
        agent._agent.run.return_value = _mock_task_result(bad_output)

        _, bench_results = agent.run_with_benchmark("t")
        assert any(not r.passed for r in bench_results)

    def test_raises_on_unknown_skill_id(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            SkillAgent.from_registry("nonexistent-v99")
