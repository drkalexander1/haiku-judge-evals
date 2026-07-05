"""Export pairwise judge-eval Inspect logs to self-preference bias CSVs + summary.json.

Self-preference bias for judge model J (only computable for J that also
authored haikus in the source run, i.e. J appears as author_left/author_right
somewhere in the pool):

  self_pick_rate_by_self   = P(J picks its own haiku | J is judging, J is one of the two authors)
  self_pick_rate_by_others = P(other judges pick J's haiku | J is one of the two authors)
  self_bias = self_pick_rate_by_self - self_pick_rate_by_others

Positive self_bias means J favors its own haiku more than an independent
judge would, for the same pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCORER_NAME = "judge_pair_scorer"


def _normalize_model(name: str) -> str:
    """Strip provider prefix, e.g. 'anthropic/claude-haiku-4-5' -> 'claude-haiku-4-5'."""
    return name.rsplit("/", 1)[-1]


def _collect_eval_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    logs = sorted(path.glob("**/*.eval"))
    if not logs:
        raise FileNotFoundError(f"No .eval logs found under {path}")
    return logs


def frame_from_eval_log(log_path: Path) -> pd.DataFrame:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    judge_model = _normalize_model(log.eval.model or log_path.stem)

    rows = []
    for sample in log.samples or []:
        if not sample.scores or SCORER_NAME not in sample.scores:
            continue
        score = sample.scores[SCORER_NAME]
        meta = sample.metadata or {}
        value = score.value if isinstance(score.value, dict) else {}
        preferred_side = value.get("preferred_side")
        preferred_author = value.get("preferred_author")
        rows.append(
            {
                "scenario_id": meta.get("scenario_id"),
                "stratum": meta.get("stratum"),
                "prompt_variant": meta.get("prompt_variant"),
                "judge_model": judge_model,
                "author_left": _normalize_model(meta.get("author_left", "")),
                "author_right": _normalize_model(meta.get("author_right", "")),
                "preferred_side": preferred_side,
                "preferred_author": _normalize_model(preferred_author) if preferred_author else None,
                "preferred_rating": value.get("preferred_rating"),
                "syllable_judgment_correct_left": value.get("syllable_judgment_correct_left"),
                "syllable_judgment_correct_right": value.get("syllable_judgment_correct_right"),
                "eval_log": log_path.name,
            }
        )
    return pd.DataFrame(rows)


def frame_from_eval_paths(paths: list[Path]) -> pd.DataFrame:
    frames = [frame_from_eval_log(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No scored samples found in eval log(s)")
    return pd.concat(frames, ignore_index=True)


def build_self_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["preferred_author"])
    judge_models = sorted(df["judge_model"].unique())
    all_authors = set(df["author_left"]).union(df["author_right"])

    rows = []
    for j in judge_models:
        involves_j = df[(df["author_left"] == j) | (df["author_right"] == j)].copy()
        if involves_j.empty:
            continue
        involves_j["self_preferred"] = (involves_j["preferred_author"] == j).astype(int)

        by_self = involves_j[involves_j["judge_model"] == j]
        by_others = involves_j[involves_j["judge_model"] != j]

        self_rate = by_self["self_preferred"].mean() if not by_self.empty else None
        others_rate = by_others["self_preferred"].mean() if not by_others.empty else None

        rows.append(
            {
                "judge_model": j,
                "authored_haikus_in_pool": j in all_authors,
                "self_pick_rate_by_self": self_rate,
                "self_pick_rate_by_others": others_rate,
                "self_bias": (self_rate - others_rate) if self_rate is not None and others_rate is not None else None,
                "n_pairs_involving_self": len(by_self),
            }
        )
    return pd.DataFrame(rows)


def build_win_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Consensus quality proxy: win rate for each author model, judged only by
    models other than itself (so self-bias doesn't leak into the ranking)."""
    df = df.dropna(subset=["preferred_author"])
    authors = sorted(set(df["author_left"]).union(df["author_right"]))

    rows = []
    for m in authors:
        involves_m = df[((df["author_left"] == m) | (df["author_right"] == m)) & (df["judge_model"] != m)]
        win_rate = (involves_m["preferred_author"] == m).mean() if not involves_m.empty else None
        rows.append({"model": m, "win_rate_excl_self_judged": win_rate, "n_pairs": len(involves_m)})
    return pd.DataFrame(rows)


def build_position_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: do judges just favor whichever haiku is shown first?"""
    df = df.dropna(subset=["preferred_side"])
    return (
        df.groupby("judge_model")["preferred_side"]
        .apply(lambda s: (s == "A").mean())
        .rename("a_pick_rate")
        .reset_index()
    )


def build_syllable_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    left = df[["judge_model", "syllable_judgment_correct_left"]].rename(
        columns={"syllable_judgment_correct_left": "correct"}
    )
    right = df[["judge_model", "syllable_judgment_correct_right"]].rename(
        columns={"syllable_judgment_correct_right": "correct"}
    )
    long = pd.concat([left, right], ignore_index=True).dropna(subset=["correct"])
    return long.groupby("judge_model")["correct"].mean().rename("syllable_judgment_accuracy").reset_index()


def build_summary(df: pd.DataFrame, *, eval_logs: list[str]) -> dict:
    self_bias = build_self_bias_table(df)
    win_rates = build_win_rate_table(df)
    position_bias = build_position_bias_table(df)
    syllable_accuracy = build_syllable_accuracy_table(df)
    return {
        "n_pair_ratings": len(df),
        "judge_models": sorted(df["judge_model"].unique().tolist()),
        "author_models": sorted(set(df["author_left"]).union(df["author_right"])),
        "self_bias": self_bias.set_index("judge_model").to_dict(orient="index"),
        "win_rates": win_rates.set_index("model").to_dict(orient="index"),
        "position_bias": position_bias.set_index("judge_model").to_dict(orient="index"),
        "syllable_judgment_accuracy": syllable_accuracy.set_index("judge_model").to_dict(orient="index"),
        "eval_logs": eval_logs,
    }


def write_run_outputs(df: pd.DataFrame, output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "pairs.csv", index=False)
    build_self_bias_table(df).to_csv(output_dir / "self_bias.csv", index=False)
    build_win_rate_table(df).to_csv(output_dir / "win_rates.csv", index=False)
    build_position_bias_table(df).to_csv(output_dir / "position_bias.csv", index=False)
    build_syllable_accuracy_table(df).to_csv(output_dir / "syllable_accuracy.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def report_eval_logs(log_path: Path, output_dir: Path) -> dict:
    paths = _collect_eval_paths(log_path)
    df = frame_from_eval_paths(paths)
    summary = build_summary(df, eval_logs=[p.name for p in paths])
    write_run_outputs(df, output_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report from pairwise judge-eval Inspect log(s)")
    parser.add_argument("log", type=Path, help="Path to a .eval file or directory containing .eval logs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = report_eval_logs(args.log, args.output)
    print(json.dumps(summary["self_bias"], indent=2, default=str))
    print(f"Wrote outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
