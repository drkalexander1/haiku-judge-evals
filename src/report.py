"""Export pairwise judge-eval Inspect logs to self-preference bias CSVs + summary.json.

Each (scenario, author pair) is judged in both position orientations (the
Mirror Test -- see inspect_eval.py). A "vote" only counts if the judge picked
the same author regardless of which side it was shown on; position-driven
flips are discarded as noise before any bias/quality metric is computed.
`position_consistency.csv` / `position_bias.csv` report how often that
happens per judge model.

Self-preference bias for judge model J (only computable for J that also
authored haikus in the source run, i.e. J appears as author_x/author_y
somewhere in the pool), computed over position-consistent votes only:

  self_pick_rate_by_self   = P(J picks its own haiku | J is judging, J is one of the two authors)
  self_pick_rate_by_others = P(other judges pick J's haiku | J is one of the two authors)
  self_bias = self_pick_rate_by_self - self_pick_rate_by_others

Positive self_bias means J favors its own haiku more than an independent
judge would, for the same pair.

`bradley_terry.csv` converts position-consistent, non-self-judged wins into
an Elo-scaled Bradley-Terry rating per author model -- a more principled
"who's actually better" ranking than a raw win-rate tally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
                "pair_id": meta.get("pair_id"),
                "orientation": meta.get("orientation"),
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


def build_position_consistency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two position-swapped orientations for each (judge, pair) into
    one vote (the Mirror Test). A vote only counts (`consistent=True`) if the judge
    picked the same author regardless of presentation order."""
    df = df.dropna(subset=["preferred_author"])
    rows = []
    for (scenario_id, judge_model, pair_id), g in df.groupby(["scenario_id", "judge_model", "pair_id"]):
        if len(g) != 2 or set(g["orientation"]) != {"fwd", "swap"}:
            continue  # incomplete pair (missing/invalid orientation) -- skip
        authors_picked = set(g["preferred_author"])
        consistent = len(authors_picked) == 1
        author_x, author_y = pair_id.split("__vs__")
        rows.append(
            {
                "scenario_id": scenario_id,
                "judge_model": judge_model,
                "pair_id": pair_id,
                "author_x": author_x,
                "author_y": author_y,
                "stratum": g["stratum"].iloc[0],
                "prompt_variant": g["prompt_variant"].iloc[0],
                "consistent": consistent,
                "preferred_author": g["preferred_author"].iloc[0] if consistent else None,
            }
        )
    return pd.DataFrame(rows)


def build_self_bias_table(votes: pd.DataFrame) -> pd.DataFrame:
    votes = votes[votes["consistent"]]
    judge_models = sorted(votes["judge_model"].unique())
    all_authors = set(votes["author_x"]).union(votes["author_y"])

    rows = []
    for j in judge_models:
        involves_j = votes[(votes["author_x"] == j) | (votes["author_y"] == j)].copy()
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
                "n_position_consistent_pairs_involving_self": len(by_self),
            }
        )
    return pd.DataFrame(rows)


def build_win_rate_table(votes: pd.DataFrame) -> pd.DataFrame:
    """Consensus quality proxy: win rate for each author model, over position-consistent
    votes judged only by *other* models (keeps self-bias from leaking into the ranking)."""
    votes = votes[votes["consistent"]]
    authors = sorted(set(votes["author_x"]).union(votes["author_y"]))

    rows = []
    for m in authors:
        involves_m = votes[((votes["author_x"] == m) | (votes["author_y"] == m)) & (votes["judge_model"] != m)]
        win_rate = (involves_m["preferred_author"] == m).mean() if not involves_m.empty else None
        rows.append({"model": m, "win_rate_excl_self_judged": win_rate, "n_pairs": len(involves_m)})
    return pd.DataFrame(rows)


def build_position_bias_table(df: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    """Sanity checks against position bias: `a_pick_rate` should hover near 50% if a judge
    isn't just favoring whichever haiku is shown first; `flip_rate` is the fraction of pairs
    where swapping position alone changed the judge's pick (discarded as noise elsewhere)."""
    df = df.dropna(subset=["preferred_side"])
    a_pick_rate = df.groupby("judge_model")["preferred_side"].apply(lambda s: (s == "A").mean()).rename("a_pick_rate")
    flip_rate = votes.groupby("judge_model")["consistent"].apply(lambda s: 1 - s.mean()).rename("flip_rate")
    return pd.concat([a_pick_rate, flip_rate], axis=1).reset_index()


def build_syllable_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    left = df[["judge_model", "syllable_judgment_correct_left"]].rename(
        columns={"syllable_judgment_correct_left": "correct"}
    )
    right = df[["judge_model", "syllable_judgment_correct_right"]].rename(
        columns={"syllable_judgment_correct_right": "correct"}
    )
    long = pd.concat([left, right], ignore_index=True).dropna(subset=["correct"])
    return long.groupby("judge_model")["correct"].mean().rename("syllable_judgment_accuracy").reset_index()


def fit_bradley_terry(win_matrix: pd.DataFrame, iterations: int = 1000, tol: float = 1e-8) -> pd.Series:
    """Zermelo/MM algorithm. win_matrix.loc[i, j] = number of times i beat j.
    Returns a per-model strength (mean-normalized to 1, not yet Elo-scaled)."""
    models = list(win_matrix.index)
    n = len(models)
    w = win_matrix.to_numpy(dtype=float)
    strength = np.ones(n)

    for _ in range(iterations):
        new_strength = np.zeros(n)
        for i in range(n):
            numerator = w[i].sum()
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                total = w[i, j] + w[j, i]
                if total > 0:
                    denom += total / (strength[i] + strength[j])
            new_strength[i] = numerator / denom if denom > 0 else strength[i]
        new_strength = new_strength / new_strength.mean()
        if np.max(np.abs(new_strength - strength)) < tol:
            strength = new_strength
            break
        strength = new_strength

    return pd.Series(strength, index=models)


def build_bradley_terry_table(votes: pd.DataFrame) -> pd.DataFrame:
    """Elo-scaled Bradley-Terry ratings from position-consistent, non-self-judged wins."""
    votes = votes[votes["consistent"]]
    votes = votes[(votes["judge_model"] != votes["author_x"]) & (votes["judge_model"] != votes["author_y"])]
    models = sorted(set(votes["author_x"]).union(votes["author_y"]))
    if not models:
        return pd.DataFrame(columns=["model", "bt_strength", "elo", "n_wins"])

    win_matrix = pd.DataFrame(0.0, index=models, columns=models)
    for _, row in votes.iterrows():
        winner = row["preferred_author"]
        loser = row["author_y"] if winner == row["author_x"] else row["author_x"]
        win_matrix.loc[winner, loser] += 1

    strength = fit_bradley_terry(win_matrix)
    elo = 1500 + 400 * np.log10(strength)
    n_wins = win_matrix.sum(axis=1)

    return (
        pd.DataFrame({"model": models, "bt_strength": strength.values, "elo": elo.values, "n_wins": n_wins.values})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )


def build_summary(df: pd.DataFrame, votes: pd.DataFrame, *, eval_logs: list[str]) -> dict:
    self_bias = build_self_bias_table(votes)
    win_rates = build_win_rate_table(votes)
    position_bias = build_position_bias_table(df, votes)
    syllable_accuracy = build_syllable_accuracy_table(df)
    bradley_terry = build_bradley_terry_table(votes)
    return {
        "n_pair_ratings": len(df),
        "n_position_consistent_votes": int(votes["consistent"].sum()),
        "judge_models": sorted(df["judge_model"].unique().tolist()),
        "author_models": sorted(set(df["author_left"]).union(df["author_right"])),
        "self_bias": self_bias.set_index("judge_model").to_dict(orient="index"),
        "win_rates": win_rates.set_index("model").to_dict(orient="index"),
        "position_bias": position_bias.set_index("judge_model").to_dict(orient="index"),
        "syllable_judgment_accuracy": syllable_accuracy.set_index("judge_model").to_dict(orient="index"),
        "bradley_terry": bradley_terry.set_index("model").to_dict(orient="index"),
        "eval_logs": eval_logs,
    }


def write_run_outputs(df: pd.DataFrame, votes: pd.DataFrame, output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "pairs.csv", index=False)
    votes.to_csv(output_dir / "position_consistency.csv", index=False)
    build_self_bias_table(votes).to_csv(output_dir / "self_bias.csv", index=False)
    build_win_rate_table(votes).to_csv(output_dir / "win_rates.csv", index=False)
    build_position_bias_table(df, votes).to_csv(output_dir / "position_bias.csv", index=False)
    build_syllable_accuracy_table(df).to_csv(output_dir / "syllable_accuracy.csv", index=False)
    build_bradley_terry_table(votes).to_csv(output_dir / "bradley_terry.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def report_eval_logs(log_path: Path, output_dir: Path) -> dict:
    paths = _collect_eval_paths(log_path)
    df = frame_from_eval_paths(paths)
    votes = build_position_consistency_table(df)
    summary = build_summary(df, votes, eval_logs=[p.name for p in paths])
    write_run_outputs(df, votes, output_dir, summary)
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
