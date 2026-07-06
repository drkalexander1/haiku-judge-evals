"""Inspect task: each --model blindly judges every pair of haikus for the same
scenario (subject + prompt variant), picking a winner and rating it.

Pairing is within scenario_id -- for N author models per scenario that's
C(N, 2) pairs. Each pair is judged in BOTH position orientations (Haiku A/B
order swapped) -- the "Mirror Test" -- so report.py can discard votes that
flip based on position alone rather than trust a single hashed ordering.

Default judging is one direct side-by-side call per orientation (120 calls per
judge at 3 authors x 20 scenarios). Pass ``-T prepair=true`` to use PRePair
instead: two isolated pointwise critiques plus a final decision from those
critiques only (360 calls per judge at the same scale).

Run:
    inspect eval src/inspect_eval.py \
      --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
      --log-dir logs/frontier-judged

    # Full PRePair protocol (~3x the default cost)
    inspect eval src/inspect_eval.py -T prepair=true \
      --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
      --log-dir logs/frontier-judged-prepair

    python -m src.report logs/frontier-judged --output results/frontier-judged
"""

from __future__ import annotations

import json
from itertools import combinations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ResponseSchema
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import json_schema

from src.schema import (
    HaikuToJudge,
    JudgePairRating,
    PointwiseCritique,
    load_haikus_to_judge,
    load_judge_prompt_template,
    load_pointwise_prompt_template,
    load_prepair_final_prompt_template,
)

SCORER_NAME = "judge_pair_scorer"
INVALID_PAIR_SCORE = {
    "preferred_side": None,
    "preferred_author": None,
    "preferred_rating": None,
    "syllable_judgment_correct_left": None,
    "syllable_judgment_correct_right": None,
}

_DECISION_SCHEMA = ResponseSchema(
    name="judge_pair_rating",
    json_schema=json_schema(JudgePairRating),
    strict=True,
)
_CRITIQUE_SCHEMA = ResponseSchema(
    name="pointwise_critique",
    json_schema=json_schema(PointwiseCritique),
    strict=True,
)


def pair_dataset(haikus_path=None) -> MemoryDataset:
    haikus = load_haikus_to_judge(haikus_path)
    by_scenario: dict[str, list[HaikuToJudge]] = {}
    for h in haikus:
        by_scenario.setdefault(h.scenario_id, []).append(h)

    samples = []
    for scenario_id, items in sorted(by_scenario.items()):
        items_sorted = sorted(items, key=lambda h: h.author_model)
        for h_x, h_y in combinations(items_sorted, 2):
            pair_id = f"{h_x.author_model}__vs__{h_y.author_model}"
            for orientation, (left, right) in (("fwd", (h_x, h_y)), ("swap", (h_y, h_x))):
                samples.append(
                    Sample(
                        input=(
                            f"Blind-judge {scenario_id}: {left.author_model} vs "
                            f"{right.author_model} (orientation={orientation})"
                        ),
                        id=f"{scenario_id}__{pair_id}__{orientation}",
                        metadata={
                            "scenario_id": scenario_id,
                            "subject": left.subject,
                            "stratum": left.stratum,
                            "prompt_variant": left.prompt_variant,
                            "pair_id": pair_id,
                            "orientation": orientation,
                            "author_left": left.author_model,
                            "author_right": right.author_model,
                            "haiku_left_text": left.full_text(),
                            "haiku_right_text": right.full_text(),
                            "syllable_perfect_actual_left": left.syllable_perfect_actual,
                            "syllable_perfect_actual_right": right.syllable_perfect_actual,
                        },
                    )
                )
    return MemoryDataset(samples, name="haiku_pairs")


@solver
def pairwise_solver():
    """One blind side-by-side A/B pick per orientation (Mirror Test only)."""

    template = load_judge_prompt_template()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        state.messages = [
            ChatMessageUser(
                content=template.format(
                    subject=meta["subject"],
                    haiku_a_text=meta["haiku_left_text"],
                    haiku_b_text=meta["haiku_right_text"],
                )
            )
        ]
        return await generate(state, response_schema=_DECISION_SCHEMA)

    return solve


@solver
def prepair_solver():
    """Two isolated pointwise critiques, then a final pairwise decision that only
    sees those critiques -- never the raw haiku text side by side."""

    pointwise_template = load_pointwise_prompt_template()
    final_template = load_prepair_final_prompt_template()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata

        state.messages = [
            ChatMessageUser(
                content=pointwise_template.format(
                    subject=meta["subject"], haiku_text=meta["haiku_left_text"], label="A"
                )
            )
        ]
        state = await generate(state, response_schema=_CRITIQUE_SCHEMA)
        critique_a_raw = state.output.completion

        state.messages = [
            ChatMessageUser(
                content=pointwise_template.format(
                    subject=meta["subject"], haiku_text=meta["haiku_right_text"], label="B"
                )
            )
        ]
        state = await generate(state, response_schema=_CRITIQUE_SCHEMA)
        critique_b_raw = state.output.completion

        state.messages = [
            ChatMessageUser(
                content=final_template.format(
                    subject=meta["subject"], critique_a=critique_a_raw, critique_b=critique_b_raw
                )
            )
        ]
        state = await generate(state, response_schema=_DECISION_SCHEMA)

        state.metadata["critique_a_raw"] = critique_a_raw
        state.metadata["critique_b_raw"] = critique_b_raw
        return state

    return solve


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
            answer=json.dumps(
                {
                    "rating": rating.model_dump(),
                    "critique_a": meta.get("critique_a_raw"),
                    "critique_b": meta.get("critique_b_raw"),
                }
            ),
        )

    return score


@task
def judge_eval(haikus_path: str | None = None, prepair: bool = False) -> Task:
    solver = prepair_solver() if prepair else pairwise_solver()
    return Task(
        dataset=pair_dataset(haikus_path),
        solver=solver,
        scorer=judge_pair_scorer(),
    )
