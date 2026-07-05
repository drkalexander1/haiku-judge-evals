"""Inspect task: each --model blindly judges every pair of haikus for the same
scenario (subject + prompt variant), picking a winner and rating it.

Pairing is within scenario_id -- for N author models per scenario that's
C(N, 2) pairs. Which author lands on the "A" side vs "B" side is decided by a
stable hash of (scenario_id, model_x, model_y), so the assignment is fixed
across all judge models (apples-to-apples comparison) and reproducible across
re-runs, but effectively random per pair (for position-bias sanity checks).

Run:
    inspect eval src/inspect_eval.py \
      --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
      --log-dir logs/frontier-judged

    python -m src.report logs/frontier-judged --output results/frontier-judged
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import json_schema

from src.schema import HaikuToJudge, JudgePairRating, load_haikus_to_judge, load_judge_prompt_template

SCORER_NAME = "judge_pair_scorer"
INVALID_PAIR_SCORE = {
    "preferred_side": None,
    "preferred_author": None,
    "preferred_rating": None,
    "syllable_judgment_correct_left": None,
    "syllable_judgment_correct_right": None,
}

_RESPONSE_SCHEMA = ResponseSchema(
    name="judge_pair_rating",
    json_schema=json_schema(JudgePairRating),
    strict=True,
)


def _stable_order(scenario_id: str, model_x: str, model_y: str) -> tuple[str, str]:
    """Deterministically decide which model is shown as Haiku A vs B."""
    key = f"{scenario_id}::{model_x}::{model_y}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    swap = int(digest[:8], 16) % 2 == 1
    return (model_y, model_x) if swap else (model_x, model_y)


def pair_dataset(haikus_path=None) -> MemoryDataset:
    haikus = load_haikus_to_judge(haikus_path)
    by_scenario: dict[str, list[HaikuToJudge]] = {}
    for h in haikus:
        by_scenario.setdefault(h.scenario_id, []).append(h)

    template = load_judge_prompt_template()
    samples = []
    for scenario_id, items in sorted(by_scenario.items()):
        items_sorted = sorted(items, key=lambda h: h.author_model)
        for h_x, h_y in combinations(items_sorted, 2):
            pair_map = {h_x.author_model: h_x, h_y.author_model: h_y}
            left_model, right_model = _stable_order(scenario_id, h_x.author_model, h_y.author_model)
            left, right = pair_map[left_model], pair_map[right_model]

            samples.append(
                Sample(
                    input=template.format(
                        subject=left.subject,
                        haiku_a_text=left.full_text(),
                        haiku_b_text=right.full_text(),
                    ),
                    id=f"{scenario_id}__{left.author_model}_vs_{right.author_model}",
                    metadata={
                        "scenario_id": scenario_id,
                        "subject": left.subject,
                        "stratum": left.stratum,
                        "prompt_variant": left.prompt_variant,
                        "author_left": left.author_model,
                        "author_right": right.author_model,
                        "syllable_perfect_actual_left": left.syllable_perfect_actual,
                        "syllable_perfect_actual_right": right.syllable_perfect_actual,
                    },
                )
            )
    return MemoryDataset(samples, name="haiku_pairs")


@scorer(name=SCORER_NAME, metrics=[])
def judge_pair_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            rating = JudgePairRating.model_validate_json(state.output.completion)
        except (ValueError, json.JSONDecodeError) as exc:
            return Score(value=INVALID_PAIR_SCORE, explanation=f"Invalid rating JSON: {exc}")

        meta = state.metadata
        preferred_author = meta["author_left"] if rating.preferred == "A" else meta["author_right"]
        syllable_judgment_correct_left = int(rating.syllable_correct_a == meta["syllable_perfect_actual_left"])
        syllable_judgment_correct_right = int(rating.syllable_correct_b == meta["syllable_perfect_actual_right"])

        return Score(
            value={
                "preferred_side": rating.preferred,
                "preferred_author": preferred_author,
                "preferred_rating": rating.preferred_rating,
                "syllable_judgment_correct_left": syllable_judgment_correct_left,
                "syllable_judgment_correct_right": syllable_judgment_correct_right,
            },
            answer=json.dumps(rating.model_dump()),
        )

    return score


@task
def judge_eval(haikus_path: str | None = None) -> Task:
    return Task(
        dataset=pair_dataset(haikus_path),
        solver=generate(),
        scorer=judge_pair_scorer(),
        config=GenerateConfig(response_schema=_RESPONSE_SCHEMA),
    )
