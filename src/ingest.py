"""Build data/haikus_to_judge.jsonl from an existing Haiku-evals generation run.

Reads predictions.jsonl (+ scenarios_snapshot.yaml, if present, for the exact
scenario metadata used at generation time) from a source results/<run>/ dir --
typically ../Haiku-evals/results/<run>. Computes ground-truth syllable
correctness here so the judge eval can later check whether a judge's own
syllable_correct call agrees with reality.

The judge prompt (prompts/judge_v1.txt) never sees author_model -- that field
is only used at analysis time.

Usage:
    python -m src.ingest --source ../Haiku-evals/results/frontier --output data/haikus_to_judge.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.schema import DATA_DIR, HaikuToJudge
from src.syllables_util import syllable_perfect


def _load_scenario_meta(source: Path) -> dict[str, dict]:
    snap = source / "scenarios_snapshot.yaml"
    if not snap.exists():
        raise FileNotFoundError(
            f"Missing {snap}; expected a Haiku-evals results/<run>/ directory "
            "with scenarios_snapshot.yaml"
        )
    with snap.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {item["id"]: item for item in raw}


def build_haikus_to_judge(source: Path) -> list[HaikuToJudge]:
    pred_path = source / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    scenario_meta = _load_scenario_meta(source)
    items: list[HaikuToJudge] = []
    with pred_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scenario = scenario_meta.get(rec["scenario_id"])
            if scenario is None:
                continue
            lines = [
                rec["prediction"]["line1"],
                rec["prediction"]["line2"],
                rec["prediction"]["line3"],
            ]
            items.append(
                HaikuToJudge(
                    judge_sample_id=f"{rec['scenario_id']}__{rec['model']}__{i}",
                    scenario_id=rec["scenario_id"],
                    subject=scenario["subject"],
                    stratum=scenario["stratum"],
                    prompt_variant=scenario["prompt_variant"],
                    author_model=rec["model"],
                    line1=lines[0],
                    line2=lines[1],
                    line3=lines[2],
                    syllable_perfect_actual=syllable_perfect(lines),
                )
            )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build haikus_to_judge.jsonl from a source generation run")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a Haiku-evals results/<run>/ dir (needs predictions.jsonl + scenarios_snapshot.yaml)",
    )
    parser.add_argument("--output", type=Path, default=DATA_DIR / "haikus_to_judge.jsonl")
    args = parser.parse_args(argv)

    items = build_haikus_to_judge(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

    authors = sorted({item.author_model for item in items})
    print(f"Wrote {len(items)} haikus to judge from {len(authors)} author models to {args.output}")
    print(f"Author models: {authors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
