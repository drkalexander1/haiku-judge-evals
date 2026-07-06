# Haiku Judge Evals

Blind pairwise LLM-judge eval for self-preference bias: when a model judges two anonymous haikus for the same subject, does it pick its own more often than an independent judge would?

Companion to [Haiku-evals](../Haiku-evals), which generates the haikus this eval judges.

## Design

1. **Ingest** haikus from a completed Haiku-evals generation run (`results/<run>/predictions.jsonl` + `scenarios_snapshot.yaml`). Each haiku keeps its true author model as hidden ground truth (`author_model`), never shown to the judge.
2. **Pair, within scenario, both orientations.** Every `scenario_id` (a subject + prompt variant) has one haiku per author model. For N authors that's C(N, 2) pairs per scenario -- at 3 authors x 20 scenarios, 60 pairs. Each pair is judged in **both** position orientations (Haiku A/B order swapped) -- the "Mirror Test" -- so a vote only counts if the judge picks the same author regardless of which side it's shown on. Position-driven flips are discarded as noise rather than trusted. This doubles sample count to 120 per judge.
3. **Judge, blind.** Every judge model in the `--model` list judges every pair -- including pairs where one of the haikus is its own -- without being told who wrote what. **Default:** one direct side-by-side A/B call per orientation (`prompts/judge_pairwise_v1.txt`) -- 120 calls per judge at 3 authors x 20 scenarios, 360 total for three judges. **Optional PRePair** (`-T prepair=true`): each orientation runs three model calls instead of one -- isolated critiques of Haiku A and B, then a final decision from those critiques only (Jeong et al., BlackboxNLP 2025 "Comparative Trap"). That triples cost to 360 calls per judge but breaks side-by-side stylistic anchoring.
4. **Analyze.** For the eval to measure self-preference for a given model, that model needs to appear in **both** the source generation run's author list and this eval's `--model` list.

Pairwise comparison (vs. absolute 1-10 scoring of each haiku independently) sidesteps a real problem: models differ in scale calibration (some hand out 8s liberally, others are harsh graders), which can swamp a subtle preference signal. A binary A/B pick cancels that out. The cost: pairing scales as C(N, 2) in the number of author models, not N -- fine at 3 authors, worth remembering before scaling the model set up.

This design targets three documented LLM-judge biases: **self-preference/nepotism bias** (Panickssery et al., NeurIPS 2024 -- judges recognize and favor their own generations), the **Comparative Trap** (addressed optionally by PRePair), and **positional bias** (addressed by the Mirror Test).

## Metrics (`src/report.py`)

| Output | Metric | What it measures |
|---|---|---|
| `self_bias.csv` | `self_pick_rate_by_self` | P(J picks its own haiku \| J is judging a pair it authored) |
| `self_bias.csv` | `self_pick_rate_by_others` | P(other judges pick J's haiku, same pairs) |
| `self_bias.csv` | `self_bias` | `self_pick_rate_by_self - self_pick_rate_by_others` -- positive means J favors its own haiku more than an independent judge would |
| `win_rates.csv` | `win_rate_excl_self_judged` | Consensus quality proxy: how often a model's haiku wins, judged only by *other* models, over position-consistent votes only (keeps self-bias from leaking into the quality ranking) |
| `position_bias.csv` | `a_pick_rate` | Sanity check -- should hover near 50% if a judge isn't just favoring whichever haiku is shown first |
| `position_bias.csv` | `flip_rate` | Mirror Test result: fraction of pairs where swapping position alone changed the judge's pick (discarded as noise elsewhere) |
| `position_consistency.csv` | `consistent` | Per (judge, pair) row: whether both orientations agreed on the winner -- the raw Mirror Test data behind every other table |
| `bradley_terry.csv` | `elo` | Elo-scaled Bradley-Terry rating per author model, fit from position-consistent, non-self-judged wins -- a more principled "who's actually better" ranking than a raw win tally |
| `syllable_accuracy.csv` | `syllable_judgment_accuracy` | How often a judge's 5-7-5 call agrees with the programmatic ground truth |

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
cp .env.example .env   # add API keys

# 1. Pull haikus from an existing Haiku-evals run
python -m src.ingest --source ../Haiku-evals/results/frontier

# 2. Every listed model judges every pair, blind (Mirror Test; 360 LLM calls for 3 judges x 20 scenarios)
inspect eval src/inspect_eval.py \
  --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
  --log-dir logs/frontier-judged

# Optional: full PRePair protocol (~1,080 LLM calls at the same scale)
# inspect eval src/inspect_eval.py -T prepair=true \
#   --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
#   --log-dir logs/frontier-judged-prepair

# 3. Export self-bias tables
python -m src.report logs/frontier-judged --output results/frontier-judged
```

See [RESULTS.md](RESULTS.md) for a worked example from the frontier judged run.

## Outputs (`results/<run>/`)

- `pairs.csv` -- every (judge, pair) rating, joined with ground truth
- `self_bias.csv`, `win_rates.csv`, `position_bias.csv`, `syllable_accuracy.csv` -- see Metrics above
- `summary.json` -- all of the above, nested

## Caveats

- `self_bias` is only defined for judge models that also authored haikus in the source run -- use the same model set for generation and judging to get a full picture.
- One rating per (judge, pair, orientation); no repeated sampling, so per-judge bias estimates carry sampling noise from LLM output variance. Re-run with `--epochs` in Inspect if you need error bars.
- Pair count grows as C(N, 2) in the number of author models, x2 for orientation, x3 more if you opt into PRePair -- fine for a handful of models (3 authors x 20 scenarios = 120 calls per judge by default), worth reconsidering (e.g. sampling a subset of pairs) if the model set grows much larger.
- Default side-by-side judging is cheaper but more exposed to the Comparative Trap; PRePair is available when you want that control and can pay ~3x the judge cost.
- All judges in the current default `--model` list are Anthropic + OpenAI mini-tier models; there's no fully "un-invested" third-party judge (e.g. Gemini) in the loop yet to fully rule out family-level bias rather than model-level bias.

## License

MIT
