"""SkillAgent: TaskAgent wrapper that is initialised from a named Skill.

Lets the eval loop hot-swap skills by name rather than by raw prompt strings.
"""

from __future__ import annotations

from typing import Any

import anthropic

from agents.task_agent import TaskAgent, TaskResult
from eval.skills import Skill, SkillRegistry

_DEFAULT_REGISTRY = SkillRegistry()


class SkillAgent:
    """Wraps TaskAgent, loading system_prompt and model from a Skill."""

    def __init__(
        self,
        skill: Skill,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.skill = skill
        self._agent = TaskAgent(
            system_prompt=skill.system_prompt,
            model=skill.model,
            client=client,
        )

    @classmethod
    def from_registry(
        cls,
        skill_id: str,
        registry: SkillRegistry | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> "SkillAgent":
        """Load a skill by ID from a registry (default: builtin registry)."""
        reg = registry or _DEFAULT_REGISTRY
        skill = reg.get(skill_id)
        return cls(skill=skill, client=client)

    @property
    def system_prompt(self) -> str:
        return self._agent.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._agent.system_prompt = value

    def run(self, transcript: str) -> TaskResult:
        return self._agent.run(transcript)

    def run_with_benchmark(self, transcript: str) -> tuple[TaskResult, list]:
        """Run the skill and validate output against its benchmark suite."""
        from eval.skills import run_benchmarks
        result = self._agent.run(transcript)
        benchmark_results = run_benchmarks(self.skill, result.output)
        return result, benchmark_results
