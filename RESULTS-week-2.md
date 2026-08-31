# Results — week 2 PRePair run (August 2026)

This is **week 2**: same haiku pool as [week 1](RESULTS.md), judged with PRePair plus an out-of-family Gemini vote. Week 1 stays at `results/frontier-judged`.

Run: `results/week-2` · Logs: `logs/week-2/` (gitignored)

**Protocol.** PRePair (`prepair=true`): isolated pointwise critiques of Haiku A and B, then a final A/B decision from those critiques only — the judge never sees the two haikus side by side (Jeong et al., BlackboxNLP 2025). Four judge models (`gpt-4o-mini`, `claude-haiku-4-5`, `claude-sonnet-4-6`, `gemini-3.5-flash-lite`) each judged all 120 mirror samples (60 unique pairs × 2 orientations) — **1,440 LLM calls** total (3 PRePair steps × 480 ratings). Gemini did not author any haikus in the pool; it is an independent "others" vote only. Haiku pool: the same 60 haikus as week 1 (`data/haikus_to_judge.jsonl`).

`google/gemini-2.5-flash` is closed to new users. `google/gemini-3.6-flash` smoke-tested cleanly (`logs/week-2-gemini`) but did not survive a full PRePair run: disabling thinking to hold cost down was rejected with a 400 `INVALID_ARGUMENT` on `thinkingConfig.thinkingBudget: 0` (`logs/week-2-gemini-nothink`), and the full run at default settings was abandoned before completing. Week 2 uses `google/gemini-3.5-flash-lite` as the Flash-class out-of-family peer.

Read the [Caveats](README.md#caveats) before citing anything here; n is small and the mirror filter still discards a large share of raw votes.

## Headline

**PRePair wipes out the week-1 self-preference signal.** Sonnet's +0.35 self-bias (90% self-pick) falls to **−0.03** (67% self-pick, matching how others and Gemini already rank it). GPT-4o-mini's +0.18 falls to **+0.04**. Gemini's independent pick rates track in-family "others" closely, so week 1's nepotism was not an Anthropic+OpenAI family-collusion artifact — it was the Comparative Trap.

**Flip rate barely moved** (39% → 34% among the three overlapping judges). PRePair is the self-bias control, not a positional-stability fix. Haiku's A-pick rate does calibrate (0.71 → 0.51); GPT-4o-mini just swaps which side it favors (0.36 → 0.63).

## 1. Self-preference bias vs week 1 (position-consistent votes only)

| Judge | Week 1 self-pick | Week 2 self-pick | Week 1 others | Week 2 others | Week 2 Gemini | Week 1 self-bias | Week 2 self-bias | n (week 2) |
|-------|-----------------:|-----------------:|--------------:|--------------:|--------------:|-----------------:|-----------------:|-----------:|
| claude-sonnet-4-6 | **0.90** | 0.67 | 0.55 | 0.70 | 0.70 | **+0.35** | **−0.03** | 30 |
| gpt-4o-mini | 0.67 | 0.41 | 0.49 | 0.37 | 0.33 | +0.18 | +0.04 | 22 |
| claude-haiku-4-5 | 0.27 | 0.35 | 0.28 | 0.46 | 0.50 | −0.01 | −0.11 | 23 |

`self_pick_rate_by_independent` is Gemini only (it never authored). It agrees with the mixed others column to within a few points, so adding an out-of-family judge does **not** rewrite who is actually preferred — it confirms the PRePair quality ranking.

**Interpretation.** Under side-by-side judging, Sonnet both won and favored itself. Under PRePair it still *wins* (below) but no longer *over-picks* itself relative to anyone else, including Gemini. GPT-4o-mini's moderate nepotism likewise disappears. Haiku stays non-nepotistic and is slightly anti-self (−0.11) — it under-picks its own haikus relative to others.

Gemini has no self-bias row: it never authored.

**Caveats.** Self-bias is still n ≈ 22–30 position-consistent self-involved pairs per model. Week 1 vs week 2 is also a protocol change, not a paired re-grade of identical judge traces — LLM sampling variance is in both numbers.

## 2. Quality ranking (non-self-judged, position-consistent)

| Author model | Week 1 win rate | Week 2 win rate | Week 1 Elo | Week 2 Elo | Week 2 n wins |
|--------------|----------------:|----------------:|-----------:|-----------:|--------------:|
| claude-sonnet-4-6 | 0.55 | **0.70** | 1528 | **1587** | 35 |
| claude-haiku-4-5 | 0.28 | 0.46 | 1429 | 1452 | 24 |
| gpt-4o-mini | 0.49 | 0.37 | 1526 | 1409 | 21 |

Week 1 had Sonnet ≈ GPT-4o-mini > Haiku. Week 2, with Gemini in the non-self-judged pool and no side-by-side anchoring, is **Sonnet ≫ Haiku > GPT-4o-mini**. Sonnet's quality lead is clearer once self-votes and comparative-trap noise are removed; GPT-4o-mini's week-1 near-tie with Sonnet does not survive PRePair.

## 3. Positional bias and mirror-test attrition

| Judge | Week 1 A-pick | Week 2 A-pick | Week 1 flip | Week 2 flip | Week 2 consistent |
|-------|--------------:|--------------:|------------:|------------:|------------------:|
| claude-haiku-4-5 | 0.71 | **0.51** | 0.42 | 0.38 | 37 / 60 |
| claude-sonnet-4-6 | 0.62 | 0.57 | 0.27 | **0.20** | 48 / 60 |
| gpt-4o-mini | 0.36 | 0.63 | 0.48 | 0.43 | 34 / 60 |
| gemini-3.5-flash-lite | — | 0.65 | — | 0.40 | 36 / 60 |

Among the three overlapping judges, aggregate flip rate is **61 / 180 = 34%** (week 1: **70 / 180 = 39%**). All four judges: **85 / 240 = 35%** flipped; **155 / 240** pair-level judgments passed the mirror test (**155 / 480** sample ratings).

PRePair did **not** deliver the positional-stability win the week-1 writeup hoped for. Isolated critiques still get a left/right label in the final decision prompt, and three of four judges still A-prefer (GPT-4o-mini reversed from B-prefer to A-prefer). Haiku is the exception: A-pick 0.51 is actually calibrated.

The mirror test is still doing real work. Do not drop it for PRePair runs.

## 4. Syllable judgment accuracy

| Judge | Week 1 | Week 2 |
|-------|-------:|-------:|
| claude-sonnet-4-6 | 0.30 | 0.38 |
| gpt-4o-mini | 0.18 | 0.36 |
| gemini-3.5-flash-lite | — | 0.28 |
| claude-haiku-4-5 | 0.27 | 0.17 |

PRePair's pointwise step asks for per-line syllable counts before the final decision. Sonnet and GPT-4o-mini improve vs week 1; Haiku gets worse. All remain poor counters against programmatic 5-7-5 (only 10 / 60 haikus in the pool are exact). This is still a competence sanity check, not the self-bias claim.

## 5. Answers to week 1's open questions

1. **Did PRePair cut flip rate vs 39%?** Only a little (34% among the same three judges). Not enough to treat PRePair as a positional-bias fix.
2. **Does Sonnet's +0.35 self-bias hold when the judge never sees the two haikus side by side?** No. It goes to approximately zero (−0.03), while Sonnet remains the quality leader.
3. **Does Gemini as independent "others" move `self_pick_rate_by_others`?** Not in a way that changes the story. Gemini's pick rates (0.70 / 0.50 / 0.33 for Sonnet / Haiku / GPT) match the mixed-others column. Week 1's in-family "others" were not secretly colluding.

## 6. What this run can and can't claim

**Defensible (as design portfolio + illustrative findings):**

- PRePair is a self-preference control: the week-1 Sonnet/GPT nepotism signal is protocol-dependent, not a stable model trait under this pool.
- An out-of-family Gemini judge agrees with in-family others on who should win, so the remaining ranking is not just Anthropic+OpenAI family bias.
- Sonnet is the quality leader under both protocols; the size of that lead is protocol-dependent.
- The Mirror Test is still required under PRePair (35% flips).

**Not defensible without caveats or more data:**

- Precise self-bias magnitudes (n ≈ 22–30 per model after filtering).
- Claims that PRePair removes positional bias — it mostly doesn't.
- Claims that GPT-4o-mini is "worse at haiku" in general — week 1 and week 2 disagree on its rank vs Haiku.
- Generalization beyond this 60-haiku pool and these four judges.

## Reproduce

```bash
source .venv/bin/activate
inspect eval src/inspect_eval.py -T prepair=true \
  --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6,google/gemini-3.5-flash-lite \
  --log-dir logs/week-2
python -m src.report logs/week-2 --output results/week-2
```
