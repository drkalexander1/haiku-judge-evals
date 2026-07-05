# Haiku Judge Evals

Blind pairwise LLM-judge eval for self-preference bias: when a model judges two anonymous haikus for the same subject, does it pick its own more often than an independent judge would?

Companion to [Haiku-evals](../Haiku-evals), which generates the haikus this eval judges.

## Design

1. **Ingest** haikus from a completed Haiku-evals generation run (`results/<run>/predictions.jsonl` + `scenarios_snapshot.yaml`). Each haiku keeps its true author model as hidden ground truth (`author_model`), never shown to the judge.
2. **Pair, within scenario.** Every `scenario_id` (a subject + prompt variant) has one haiku per author model. For N authors that's C(N, 2) pairs per scenario -- at 3 authors x 20 scenarios, 60 pairs. Which author lands on the "A" side vs "B" side is decided by a stable hash of `(scenario_id, model_x, model_y)`, so the assignment is fixed across every judge (apples-to-apples) and reproducible, but effectively random pair-to-pair (useful for a position-bias sanity check).
3. **Judge, blind.** Every judge model in the `--model` list judges every pair -- including pairs where one of the haikus is its own -- without being told who wrote what. One response per pair:
   - `preferred`: "A" or "B"
   - `preferred_rating`: 1-10 quality rating of whichever haiku it picked
   - `syllable_correct_a` / `syllable_correct_b`: the judge's own call on 5-7-5 form for each haiku (checked against `syllable_perfect_actual`, computed programmatically at ingest time)
4. **Analyze.** For the eval to measure self-preference for a given model, that model needs to appear in **both** the source generation run's author list and this eval's `--model` list.

Pairwise comparison (vs. absolute 1-10 scoring of each haiku independently) sidesteps a real problem: models differ in scale calibration (some hand out 8s liberally, others are harsh graders), which can swamp a subtle preference signal. A binary A/B pick cancels that out. The cost: pairing scales as C(N, 2) in the number of author models, not N -- fine at 3 authors, worth remembering before scaling the model set up.

## Metrics (`src/report.py`)

| Output | Metric | What it measures |
|---|---|---|
| `self_bias.csv` | `self_pick_rate_by_self` | P(J picks its own haiku \| J is judging a pair it authored) |
| `self_bias.csv` | `self_pick_rate_by_others` | P(other judges pick J's haiku, same pairs) |
| `self_bias.csv` | `self_bias` | `self_pick_rate_by_self - self_pick_rate_by_others` -- positive means J favors its own haiku more than an independent judge would |
| `win_rates.csv` | `win_rate_excl_self_judged` | Consensus quality proxy: how often a model's haiku wins, judged only by *other* models (keeps self-bias from leaking into the quality ranking) |
| `position_bias.csv` | `a_pick_rate` | Sanity check -- should hover near 50% if a judge isn't just favoring whichever haiku is shown first |
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

# 2. Every listed model judges every pair, blind
inspect eval src/inspect_eval.py \
  --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
  --log-dir logs/frontier-judged

# 3. Export self-bias tables
python -m src.report logs/frontier-judged --output results/frontier-judged
```

## Outputs (`results/<run>/`)

- `pairs.csv` -- every (judge, pair) rating, joined with ground truth
- `self_bias.csv`, `win_rates.csv`, `position_bias.csv`, `syllable_accuracy.csv` -- see Metrics above
- `summary.json` -- all of the above, nested

## Caveats

- `self_bias` is only defined for judge models that also authored haikus in the source run -- use the same model set for generation and judging to get a full picture.
- One rating per (judge, pair); no repeated sampling, so per-judge bias estimates carry sampling noise from LLM output variance. Re-run with `--epochs` in Inspect if you need error bars.
- Pair count grows as C(N, 2) in the number of author models -- fine for a handful of models, worth reconsidering (e.g. sampling a subset of pairs) if the model set grows much larger.

## License

MIT
