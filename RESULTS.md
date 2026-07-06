# Results — frontier judged run (July 2026)

Run: `results/frontier-judged` · Logs: `logs/frontier-judged/` (gitignored)

**Protocol.** Default mirror-only pairwise judging (`prepair=false`): one blind side-by-side A/B call per orientation. Three judge models (`gpt-4o-mini`, `claude-haiku-4-5`, `claude-sonnet-4-6`) each judged all 120 mirror samples (60 unique pairs × 2 orientations) — **360 LLM calls** total. Haiku pool: 60 haikus from 20 scenarios × 3 author models (bundled `data/haikus_to_judge.jsonl`).

Read the [Caveats](README.md#caveats) before citing anything here; n is small and the mirror filter discards most raw votes.

## Headline

**Sonnet shows a large self-preference signal; Haiku does not.** When position-consistent votes only are counted, `claude-sonnet-4-6` picks its own haiku ~90% of the time in self-involved pairs (+0.35 self-bias vs. independent judges), while `claude-haiku-4-5` is flat (−0.01). The bigger methodological surprise is **positional instability**: swapping A/B order alone flipped the winner in **39%** of pair-level judgments, so only **110 / 360** raw ratings survive the mirror test — direct side-by-side judging is noisier than the design assumed.

## 1. Self-preference bias (position-consistent votes only)

| Judge | Self-pick rate | Others pick its haiku | Self-bias | n (self-involved pairs) |
|-------|---------------:|----------------------:|----------:|------------------------:|
| claude-sonnet-4-6 | **0.90** | 0.55 | **+0.35** | 29 |
| gpt-4o-mini | 0.67 | 0.49 | +0.18 | 24 |
| claude-haiku-4-5 | 0.27 | 0.28 | −0.01 | 26 |

Raw (pre-mirror) self-pick rates tell a similar story for Sonnet (79% of 80 self-involved ratings) and GPT-4o-mini (60%), but Haiku is already low at 35%.

**Interpretation.** Sonnet both *wins* head-to-head quality comparisons (below) and *favors its own output* when it is the judge — the two findings coexist. Haiku is the opposite pattern: lowest quality ranking, no detectable nepotism. GPT-4o-mini sits in the middle on both dimensions.

**Caveats.** Self-bias is computed on 24–29 position-consistent, self-involved pairs per model — enough to illustrate the metric, not enough for tight confidence intervals. All judges share the same two provider families (Anthropic + OpenAI); there is no fully independent third-party judge in the loop.

## 2. Quality ranking (non-self-judged, position-consistent)

| Author model | Win rate (excl. self-judged) | Bradley-Terry Elo | n wins |
|--------------|----------------------------:|------------------:|-------:|
| claude-sonnet-4-6 | **0.55** | **1528** | 9 |
| gpt-4o-mini | 0.49 | 1526 | 14 |
| claude-haiku-4-5 | 0.28 | 1429 | 8 |

Sonnet and GPT-4o-mini are close on Elo; Haiku trails clearly. This ranking uses only votes from *other* judges, so Sonnet's self-bias does not inflate its quality score.

## 3. Positional bias and mirror-test attrition

| Judge | A-pick rate | Flip rate | Position-consistent pairs |
|-------|------------:|----------:|--------------------------:|
| claude-haiku-4-5 | **0.71** | 0.42 | 35 / 60 |
| claude-sonnet-4-6 | 0.62 | 0.27 | 44 / 60 |
| gpt-4o-mini | **0.36** | **0.48** | 31 / 60 |

A well-calibrated judge should have `a_pick_rate ≈ 0.50`. None do. Haiku and Sonnet favor whichever haiku is shown as **A**; GPT-4o-mini favors **B** (A-pick rate 0.36).

**Flip rate** = fraction of (judge, pair) rows where swapping A/B order changed the winner. Aggregate: **70 / 180 = 39%** flipped; only **110 / 180 = 61%** passed the mirror test. GPT-4o-mini is the noisiest judge (48% flip). This is the main reason effective sample size is much smaller than the 360 calls suggest.

The mirror test is doing real work here — without it, positional noise would contaminate both self-bias and quality estimates. But at this noise level, a single-pass side-by-side protocol (without PRePair) may be too unstable for fine-grained claims.

## 4. Syllable judgment accuracy

| Judge | Syllable judgment accuracy |
|-------|---------------------------:|
| claude-sonnet-4-6 | 0.30 |
| claude-haiku-4-5 | 0.27 |
| gpt-4o-mini | 0.18 |

Programmatic ground truth: only **10 / 60 (17%)** of haikus in the pool are exactly 5-7-5 (`syllable_perfect_actual`). Judges agree with ground truth on syllable correctness just **18–30%** of the time (per-line calls aggregated across both haikus in each rating). Judges are poor syllable counters in this side-by-side setting — or they apply a looser standard than the programmatic counter — but the pool is also mostly non-conforming, so random guessing would not score much worse.

This metric is a sanity check on judge competence, not the primary self-bias signal.

## 5. What this run can and can't claim

**Defensible (as design portfolio + illustrative findings):**

- The eval pipeline works end-to-end: ingest → blind pairwise judging → mirror filter → self-bias / win-rate / Bradley-Terry export.
- Sonnet shows a large positive self-bias and Haiku does not, under the stated protocol and mirror filter.
- Positional instability is severe under direct side-by-side judging — a concrete motivation for the mirror test (and optionally PRePair).
- Quality ranking (Sonnet ≈ GPT-4o-mini > Haiku) is directionally consistent across win-rate and Bradley-Terry, using non-self-judged votes only.

**Not defensible without caveats or more data:**

- Precise self-bias magnitudes (n ≈ 24–29 per model after filtering).
- Cross-provider generalization — no Gemini or other out-of-family judge.
- Claims that side-by-side judging is sufficient for stable preferences — 39% flip rate argues against it.
- Comparison to PRePair — this run used direct pairwise only; a separate `-T prepair=true` run would be needed to test whether isolated critiques reduce flips or self-bias.

## Reproduce

```bash
source .venv/bin/activate
inspect eval src/inspect_eval.py \
  --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
  --log-dir logs/frontier-judged
python -m src.report logs/frontier-judged --output results/frontier-judged
```
