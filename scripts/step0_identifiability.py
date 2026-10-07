"""Step 0: can this pool identify beta_form and beta_topic separately?

L1err is |line1-5| + |line2-7| + |line3-5|, same as haiku-evals
syllable_l1_error. subject_cosine_full is recomputed with all-MiniLM-L6-v2
(the haiku-evals local embedder). The original generation-run scores are
not in this repo.

Needs sentence-transformers, which isn't a core dependency:
    pip install -e ".[step0]"
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import load_haikus_to_judge
from src.syllables_util import TARGET_SYLLABLES, line_syllable_counts

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "step0"


def syllable_l1_error(lines: list[str], target: tuple[int, ...] = TARGET_SYLLABLES) -> int:
    counts = line_syllable_counts(lines)
    if len(counts) != len(target):
        return sum(abs(c - t) for c, t in zip(counts, target)) + 50
    return sum(abs(c - t) for c, t in zip(counts, target))


def embed_cosines(texts: list[str], subjects: list[str]) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    unique = list(dict.fromkeys(texts + subjects))
    vecs = model.encode(unique, convert_to_numpy=True, show_progress_bar=False)
    index = {t: vecs[i] for i, t in enumerate(unique)}

    out = []
    for text, subject in zip(texts, subjects):
        a = np.asarray(index[text], dtype=np.float64)
        b = np.asarray(index[subject], dtype=np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out.append(0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb)))
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    haikus = load_haikus_to_judge()
    rows = []
    for h in haikus:
        lines = [h.line1, h.line2, h.line3]
        counts = line_syllable_counts(lines)
        rows.append(
            {
                "judge_sample_id": h.judge_sample_id,
                "scenario_id": h.scenario_id,
                "subject": h.subject,
                "stratum": h.stratum,
                "prompt_variant": h.prompt_variant,
                "author_model": h.author_model,
                "line1": h.line1,
                "line2": h.line2,
                "line3": h.line3,
                "syl1": counts[0] if len(counts) > 0 else None,
                "syl2": counts[1] if len(counts) > 1 else None,
                "syl3": counts[2] if len(counts) > 2 else None,
                "L1err": syllable_l1_error(lines),
                "syllable_perfect_actual": h.syllable_perfect_actual,
                "text": h.full_text(),
            }
        )
    haiku_df = pd.DataFrame(rows)

    print("Embedding 60 haikus + subjects with all-MiniLM-L6-v2 …")
    haiku_df["cos"] = embed_cosines(haiku_df["text"].tolist(), haiku_df["subject"].tolist())

    by_scenario: dict[str, list[dict]] = {}
    for rec in haiku_df.to_dict(orient="records"):
        by_scenario.setdefault(rec["scenario_id"], []).append(rec)

    pair_rows = []
    for scenario_id, items in sorted(by_scenario.items()):
        items = sorted(items, key=lambda r: r["author_model"])
        for left, right in combinations(items, 2):
            pair_rows.append(
                {
                    "scenario_id": scenario_id,
                    "subject": left["subject"],
                    "stratum": left["stratum"],
                    "prompt_variant": left["prompt_variant"],
                    "author_left": left["author_model"],
                    "author_right": right["author_model"],
                    "L1err_left": left["L1err"],
                    "L1err_right": right["L1err"],
                    "cos_left": left["cos"],
                    "cos_right": right["cos"],
                    "delta_form": left["L1err"] - right["L1err"],
                    "delta_topic": left["cos"] - right["cos"],
                    "delta_form_model": right["L1err"] - left["L1err"],
                    "delta_topic_model": left["cos"] - right["cos"],
                }
            )
    pairs = pd.DataFrame(pair_rows)

    # Identifiability uses one orientation per author-pair (60 rows).
    # delta_form is L1err_left - L1err_right, so a positive value means left
    # is further from 5-7-5. The *_model columns are oriented so positive means
    # left is better on that channel (lower L1err, higher cos), matching the
    # logit's (L1err_right - L1err_left) and (cos_left - cos_right).
    r = pearson(pairs["delta_form_model"].to_numpy(), pairs["delta_topic_model"].to_numpy())
    r_abs = pearson(pairs["delta_form"].abs().to_numpy(), pairs["delta_topic"].abs().to_numpy())

    form_gaps = {
        "ge_1": int((pairs["delta_form"].abs() >= 1).sum()),
        "ge_2": int((pairs["delta_form"].abs() >= 2).sum()),
        "zero": int((pairs["delta_form"] == 0).sum()),
    }
    topic_gaps = {
        "ge_0.02": int((pairs["delta_topic"].abs() >= 0.02).sum()),
        "ge_0.05": int((pairs["delta_topic"].abs() >= 0.05).sum()),
        "ge_0.10": int((pairs["delta_topic"].abs() >= 0.10).sum()),
        "lt_0.02": int((pairs["delta_topic"].abs() < 0.02).sum()),
    }

    # Tradeoff = form and topic favor different haikus. Use the *_model
    # columns, where both channels share a direction: on the raw deltas,
    # opposite signs mean the same haiku wins both.
    form_m, topic_m = pairs["delta_form_model"], pairs["delta_topic_model"]
    crossed = int((((form_m > 0) & (topic_m < 0)) | ((form_m < 0) & (topic_m > 0))).sum())
    same_sign = int((((form_m > 0) & (topic_m > 0)) | ((form_m < 0) & (topic_m < 0))).sum())

    by_author = (
        haiku_df.groupby("author_model")
        .agg(
            n=("L1err", "size"),
            L1err_mean=("L1err", "mean"),
            L1err_std=("L1err", "std"),
            perfect=("syllable_perfect_actual", "sum"),
            cos_mean=("cos", "mean"),
            cos_std=("cos", "std"),
        )
        .reset_index()
    )

    summary = {
        "n_haikus": int(len(haiku_df)),
        "n_pairs": int(len(pairs)),
        "n_perfect_575": int(haiku_df["syllable_perfect_actual"].sum()),
        "L1err_mean": float(haiku_df["L1err"].mean()),
        "L1err_std": float(haiku_df["L1err"].std()),
        "L1err_min": int(haiku_df["L1err"].min()),
        "L1err_max": int(haiku_df["L1err"].max()),
        "cos_mean": float(haiku_df["cos"].mean()),
        "cos_std": float(haiku_df["cos"].std()),
        "cos_min": float(haiku_df["cos"].min()),
        "cos_max": float(haiku_df["cos"].max()),
        "delta_form_mean": float(pairs["delta_form"].mean()),
        "delta_form_std": float(pairs["delta_form"].std()),
        "delta_form_min": int(pairs["delta_form"].min()),
        "delta_form_max": int(pairs["delta_form"].max()),
        "delta_topic_mean": float(pairs["delta_topic"].mean()),
        "delta_topic_std": float(pairs["delta_topic"].std()),
        "delta_topic_min": float(pairs["delta_topic"].min()),
        "delta_topic_max": float(pairs["delta_topic"].max()),
        "corr_delta_form_delta_topic": r,
        "corr_abs_delta_form_abs_delta_topic": r_abs,
        "pairs_form_gap_ge_1": form_gaps["ge_1"],
        "pairs_form_gap_ge_2": form_gaps["ge_2"],
        "pairs_form_tied": form_gaps["zero"],
        "pairs_topic_gap_ge_0.02": topic_gaps["ge_0.02"],
        "pairs_topic_gap_ge_0.05": topic_gaps["ge_0.05"],
        "pairs_topic_gap_ge_0.10": topic_gaps["ge_0.10"],
        "pairs_topic_near_tie_lt_0.02": topic_gaps["lt_0.02"],
        "pairs_tradeoff_opposite_sign": crossed,
        "pairs_same_sign": same_sign,
        "pairs_either_zero": int(((pairs["delta_form"] == 0) | (pairs["delta_topic"].abs() < 1e-12)).sum()),
        "embedder": "local:all-MiniLM-L6-v2 (recomputed; not joined from haiku-evals predictions.jsonl)",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    haiku_df.drop(columns=["text"]).to_csv(OUT / "haikus.csv", index=False)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    by_author.to_csv(OUT / "by_author.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(by_author.to_string(index=False))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
