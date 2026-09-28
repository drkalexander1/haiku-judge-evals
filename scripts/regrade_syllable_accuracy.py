"""Re-grade a finished run's syllable accuracy against the corrected counts.

The published syllable_accuracy.csv files were graded against the old
heuristic counter, and that old answer is baked into the scores in the .eval
logs, so re-running report.py can't fix them. This script works from
results/<run>/pairs.csv instead:

1. Recompute the old ground truth with the old counter (syllables.estimate on
   every word, as src/syllables_util.py did before the cmudict change).
2. Back out each judge's yes/no call: syllable_judgment_correct == 1 means it
   agreed with the old truth, 0 means it disagreed.
3. Check that this reproduces the published syllable_accuracy.csv exactly.
4. Grade the same calls against syllable_perfect_actual in
   data/haikus_to_judge.jsonl (recounted with cmudict).

Writes results/<run>/syllable_accuracy_corrected.csv and leaves the published
files alone. No API calls.

Usage:
    python scripts/regrade_syllable_accuracy.py frontier-judged week-2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import syllables

from src.schema import load_haikus_to_judge

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)


def _old_count(text: str) -> int:
    """The pre-cmudict counter, kept here only to reproduce old results."""
    cleaned = _PUNCT_RE.sub(" ", text).strip()
    total = 0
    for word in cleaned.split():
        w = word.strip("-'")
        if w:
            total += max(1, syllables.estimate(w))
    return total


def _truth_table() -> pd.DataFrame:
    rows = []
    for h in load_haikus_to_judge():
        lines = [h.line1, h.line2, h.line3]
        rows.append(
            {
                "scenario_id": h.scenario_id,
                "author": h.author_model,
                "old_truth": [_old_count(line) for line in lines] == [5, 7, 5],
                "new_truth": h.syllable_perfect_actual,
            }
        )
    return pd.DataFrame(rows)


def regrade(run: str) -> pd.DataFrame:
    pairs = pd.read_csv(RESULTS / run / "pairs.csv")
    truth = _truth_table()

    calls = []
    for side in ("left", "right"):
        part = pairs[["judge_model", "scenario_id", f"author_{side}", f"syllable_judgment_correct_{side}"]]
        calls.append(part.set_axis(["judge_model", "scenario_id", "author", "correct_old"], axis=1))
    calls = pd.concat(calls, ignore_index=True).dropna(subset=["correct_old"])
    calls = calls.merge(truth, on=["scenario_id", "author"], how="left", validate="many_to_one")
    if calls["old_truth"].isna().any():
        raise ValueError(f"{run}: pairs.csv has haikus that aren't in data/haikus_to_judge.jsonl")
    calls["said_575"] = calls["old_truth"].where(calls["correct_old"] == 1, ~calls["old_truth"])

    published = pd.read_csv(RESULTS / run / "syllable_accuracy.csv").set_index("judge_model")
    rows = []
    for judge, g in calls.groupby("judge_model"):
        reproduced = (g["said_575"] == g["old_truth"]).mean()
        expected = published.loc[judge, "syllable_judgment_accuracy"]
        if abs(reproduced - expected) > 1e-9:
            raise ValueError(f"{run}/{judge}: reproduced {reproduced} != published {expected}")

        errors = g[~g["new_truth"]]
        correct = g[g["new_truth"]]
        rows.append(
            {
                "judge_model": judge,
                "n_calls": len(g),
                "accuracy_published": expected,
                "accuracy_corrected": (g["said_575"] == g["new_truth"]).mean(),
                "said_575_rate": g["said_575"].mean(),
                "errors_caught": int((~errors["said_575"]).sum()),
                "errors_total": len(errors),
                "catch_rate": (~errors["said_575"]).mean() if len(errors) else None,
                "false_flags": int((~correct["said_575"]).sum()),
                "correct_total": len(correct),
                "false_flag_rate": (~correct["said_575"]).mean() if len(correct) else None,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-grade syllable accuracy against the corrected counts")
    parser.add_argument("runs", nargs="+", help="Run names under results/, e.g. frontier-judged week-2")
    args = parser.parse_args(argv)

    for run in args.runs:
        table = regrade(run)
        out = RESULTS / run / "syllable_accuracy_corrected.csv"
        table.to_csv(out, index=False, lineterminator="\n")
        print(f"\n{run}: reproduced published accuracy for all {len(table)} judges")
        print(table.to_string(index=False))
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
