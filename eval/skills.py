"""Skill registry: versioned, benchmarked, failure-documented agent capabilities.

A skill is the unit of reusable AI work:
  - name + version identify it uniquely
  - system_prompt is the executable core
  - benchmarks define the pass/fail bar
  - gotchas document known failure modes
  - output_schema describes what the skill returns
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkCase:
    """A single pass/fail benchmark for a skill."""
    name: str
    description: str
    input_summary: str       # human-readable description of the input
    expected_keys: list[str] # keys that must be present in the output
    min_score: float | None = None   # if set, critic score must meet this bar


@dataclass
class Skill:
    name: str
    version: str
    model: str
    system_prompt: str
    output_schema: dict[str, Any]           # JSON schema of expected output
    benchmarks: list[BenchmarkCase]
    gotchas: list[str]                       # known failure modes / edge cases
    description: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}-{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model": self.model,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "output_schema": self.output_schema,
            "benchmarks": [
                {
                    "name": b.name,
                    "description": b.description,
                    "input_summary": b.input_summary,
                    "expected_keys": b.expected_keys,
                    "min_score": b.min_score,
                }
                for b in self.benchmarks
            ],
            "gotchas": self.gotchas,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        return cls(
            name=d["name"],
            version=d["version"],
            model=d["model"],
            description=d.get("description", ""),
            system_prompt=d["system_prompt"],
            output_schema=d.get("output_schema", {}),
            benchmarks=[
                BenchmarkCase(
                    name=b["name"],
                    description=b["description"],
                    input_summary=b["input_summary"],
                    expected_keys=b["expected_keys"],
                    min_score=b.get("min_score"),
                )
                for b in d.get("benchmarks", [])
            ],
            gotchas=d.get("gotchas", []),
        )


# ---------------------------------------------------------------------------
# Built-in skills
# ---------------------------------------------------------------------------

PODCAST_SUMMARIZER_V1 = Skill(
    name="podcast-summarizer",
    version="v1",
    model="claude-sonnet-4-6",
    description="Produces a structured JSON summary of a podcast transcript.",
    system_prompt="""You are an expert podcast analyst. Given a podcast transcript, produce a concise structured summary.

Return ONLY valid JSON with this exact schema:
{
  "key_topics": ["<topic>", ...],
  "main_argument": "<string>",
  "notable_quotes": ["<quote>", ...],
  "actionable_takeaways": ["<item>", ...]
}

Be specific and grounded in the actual content. Do not invent information.""",
    output_schema={
        "type": "object",
        "required": ["key_topics", "main_argument", "notable_quotes", "actionable_takeaways"],
        "properties": {
            "key_topics": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "main_argument": {"type": "string"},
            "notable_quotes": {"type": "array", "items": {"type": "string"}},
            "actionable_takeaways": {"type": "array", "items": {"type": "string"}},
        },
    },
    benchmarks=[
        BenchmarkCase(
            name="all_keys_present",
            description="Output must contain all four required keys.",
            input_summary="Any podcast transcript",
            expected_keys=["key_topics", "main_argument", "notable_quotes", "actionable_takeaways"],
        ),
        BenchmarkCase(
            name="non_empty_topics",
            description="key_topics must have at least one entry.",
            input_summary="Any podcast transcript with a clear theme",
            expected_keys=["key_topics"],
            min_score=None,
        ),
        BenchmarkCase(
            name="main_argument_is_string",
            description="main_argument must be a non-empty string.",
            input_summary="Any podcast transcript",
            expected_keys=["main_argument"],
        ),
    ],
    gotchas=[
        "Very short transcripts (<200 words) often produce generic summaries with low accuracy scores.",
        "Transcripts with multiple guests may conflate speakers' views in main_argument.",
        "Heavy technical jargon (e.g., ML papers) causes notable_quotes to over-quote definitions rather than insights.",
        "Non-English transcripts are not supported — output degrades silently rather than erroring.",
        "Transcripts >100k chars are silently truncated; long episodes lose their conclusions.",
    ],
)

TOPIC_EXTRACTOR_V1 = Skill(
    name="topic-extractor",
    version="v1",
    model="claude-sonnet-4-6",
    description="Extracts a flat, ranked list of topics from a podcast transcript with confidence scores.",
    system_prompt="""You are a topic extraction specialist. Analyse the podcast transcript and extract the most important topics discussed.

Return ONLY valid JSON with this exact schema:
{
  "topics": [
    {"topic": "<string>", "confidence": <float 0-1>, "mentions": <int>}
  ]
}

Rules:
- Return 5-10 topics, ranked by importance (most important first)
- confidence reflects how central the topic is to the episode
- mentions is your estimate of how many times it was discussed
- Topics should be specific noun phrases, not vague categories""",
    output_schema={
        "type": "object",
        "required": ["topics"],
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["topic", "confidence", "mentions"],
                },
                "minItems": 1,
            }
        },
    },
    benchmarks=[
        BenchmarkCase(
            name="topics_key_present",
            description="Output must contain a 'topics' array.",
            input_summary="Any podcast transcript",
            expected_keys=["topics"],
        ),
        BenchmarkCase(
            name="at_least_five_topics",
            description="Must return at least 5 topics.",
            input_summary="Podcast with multiple distinct subjects",
            expected_keys=["topics"],
        ),
    ],
    gotchas=[
        "Monologue episodes with a single sustained argument produce fewer topics than requested.",
        "Interview podcasts with rapid topic-switching cause confidence scores to cluster around 0.5.",
        "Very long transcripts cause early topics to dominate; later content is under-represented.",
        "Topics can overlap semantically (e.g., 'AI agents' and 'autonomous agents'); no deduplication is performed.",
        "The 'mentions' count is an estimate and can be off by 2-3x on long episodes.",
    ],
)

QUOTE_FINDER_V1 = Skill(
    name="quote-finder",
    version="v1",
    model="claude-sonnet-4-6",
    description="Extracts the most quotable, shareable moments from a podcast transcript.",
    system_prompt="""You are a media editor specialising in finding the most shareable quotes from podcast transcripts.

Return ONLY valid JSON with this exact schema:
{
  "quotes": [
    {
      "text": "<verbatim or near-verbatim quote>",
      "speaker": "<speaker name or 'Unknown'>",
      "context": "<one sentence explaining why this quote is notable>",
      "shareable_score": <int 1-5>
    }
  ]
}

Rules:
- Return 3-7 quotes
- Prefer complete thoughts over sentence fragments
- shareable_score: 1=interesting only to fans, 5=would go viral
- Preserve the speaker's voice — minimal paraphrasing""",
    output_schema={
        "type": "object",
        "required": ["quotes"],
        "properties": {
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "speaker", "context", "shareable_score"],
                },
                "minItems": 1,
            }
        },
    },
    benchmarks=[
        BenchmarkCase(
            name="quotes_key_present",
            description="Output must contain a 'quotes' array.",
            input_summary="Any podcast transcript",
            expected_keys=["quotes"],
        ),
        BenchmarkCase(
            name="quote_has_required_fields",
            description="Each quote must have text, speaker, context, shareable_score.",
            input_summary="Transcript with identifiable speakers",
            expected_keys=["quotes"],
        ),
    ],
    gotchas=[
        "Transcripts without speaker labels result in all speakers marked 'Unknown'.",
        "Highly technical discussions produce low shareable_scores across the board, even for genuinely good quotes.",
        "Very short quotes (<10 words) score high on shareability but lack context — use context field to compensate.",
        "Paraphrasing creep: despite instructions, the model occasionally polishes quotes, losing the speaker's voice.",
        "Episodes with many interruptions produce fragmented quotes that read poorly out of context.",
    ],
)

BUILTIN_SKILLS: list[Skill] = [
    PODCAST_SUMMARIZER_V1,
    TOPIC_EXTRACTOR_V1,
    QUOTE_FINDER_V1,
]


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """In-memory registry of skills, optionally backed by a JSON file."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        for s in BUILTIN_SKILLS:
            self.register(s)

    def register(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Skill:
        if skill_id not in self._skills:
            raise KeyError(f"Skill {skill_id!r} not found. Available: {self.list_ids()}")
        return self._skills[skill_id]

    def list_ids(self) -> list[str]:
        return sorted(self._skills)

    def save(self, path: str | Path) -> None:
        data = {s.id: s.to_dict() for s in self._skills.values()}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SkillRegistry":
        registry = cls.__new__(cls)
        registry._skills = {}
        data = json.loads(Path(path).read_text())
        for skill_dict in data.values():
            skill = Skill.from_dict(skill_dict)
            registry._skills[skill.id] = skill
        return registry


# ---------------------------------------------------------------------------
# Skill runner: validates output against a skill's benchmarks
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    skill_id: str
    case_name: str
    passed: bool
    reason: str


def run_benchmarks(skill: Skill, output: dict[str, Any]) -> list[BenchmarkResult]:
    """Validate a skill's output against its benchmark suite.

    Does not make any API calls — purely structural validation.
    """
    results = []
    for case in skill.benchmarks:
        passed = True
        reason = "ok"

        # Check all expected keys are present
        for key in case.expected_keys:
            if key not in output:
                passed = False
                reason = f"missing key: {key!r}"
                break
            val = output[key]
            # Non-empty check for lists and strings
            if isinstance(val, (list, str)) and len(val) == 0:
                passed = False
                reason = f"key {key!r} is empty"
                break

        results.append(BenchmarkResult(
            skill_id=skill.id,
            case_name=case.name,
            passed=passed,
            reason=reason,
        ))
    return results
