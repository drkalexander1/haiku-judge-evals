# Persona judges — design sketch

**Status:** signed off by Daniel, 2026-09-22. Predictions are **frozen:
direction only, no numeric β thresholds.** Nothing built, nothing generated.
This is week 3 on the haiku-judge branch (week 1 direct pairwise, week 2
PRePair + Gemini). Paul takes the Inspect build from this freeze.

**Round number:** week 3. Step 0 (rerun 2026-10-01 with the CMU counter):
the topic channel is identifiable on the existing pool; the form channel is
thin. **Keep this pool** (Daniel, 2026-10-07): no extra off-form haikus.
P2/P3 may be inconclusive; that is a reported limitation, not a new
generation run. Paul takes the Inspect build.

## The question

**When you give a judge a persona, what does it actually change?**

A persona is an instruction, not a sample of a population. This eval does not
ask whether a "haiku master" persona judges like real haiku masters — there is
no human panel here, and that is the silicon-sampling frame already rejected in
the water-quality valuation work. It asks something checkable without one:

> Does the persona move the things it *should* move, and leave alone the things
> it *shouldn't*?

Haiku is a good place to ask this because the domain has two channels the
judge cannot argue with — **syllable counts** (programmatic) and **subject
grounding** (embedding cosine) — sitting alongside pure taste. So a persona's
effect can be read against ground truth instead of against another model's
opinion.

## Three kinds of outcome

| kind | what | should a persona move it? | how it's read |
|---|---|---|---|
| **Taste weights** | how much the judge's picks track form vs. subject fit | **Yes — in the direction the persona states** | revealed-preference weights (below) |
| **Competence** | syllable-count accuracy vs. programmatic truth | **No** | accuracy delta vs. no-persona baseline |
| **Nuisance biases** | position bias (A-pick, flip rate) and self-preference | **No** | delta vs. baseline |

The finding is wherever a persona breaks this table: moves a taste weight the
wrong way (the instruction isn't being followed), or moves competence or bias
(the persona is leaking into things it has no business touching). A persona
that erases Sonnet's +0.35 week-1 self-preference would be the most
interesting single result, because it would say the persona reaches the
self-recognition channel. That's the same channel PRePair broke in week 2.

## Revealed-preference weights — the primary measure

Each judgment is a binary choice between two haikus, so fit a conditional
logit per judge × persona:

```
P(pick left) = σ( β0 + β_form · (L1err_right − L1err_left)
                     + β_topic · (cos_left − cos_right) )
```

- `β_form` — how much lower syllable error wins the pick
- `β_topic` — how much subject grounding wins the pick
- `β0` — position bias, **estimated as a parameter instead of discarded**. That
  keeps all 120 observations per judge rather than the ~61% that survive the
  mirror filter. Report mirror-filtered numbers too, for comparability with
  weeks 1–2.

A persona's effect is the **sign** of the shift in `β_form` / `β_topic`
relative to the no-persona baseline, within judge. Any signed movement is
the datum — a small move the right way is a weak hit, a large move the
wrong way is a miss. Report the βs, the shift vs P0, and intervals or a
paired contrast; do not discard a correctly-signed shift as noise, and do
not pre-register a magnitude cutoff. Step 0 can tell us the channels
separate. It cannot tell us how hard a model follows an instruction.

Every pair is judged under every condition, so this is a paired design and
the baseline judgment of the same pair is available as a covariate.

This is the random-utility / discrete-choice machinery from stated-preference
economics, pointed at a model. It's the same toolkit the valuation eval uses,
which is worth saying out loud in a writeup.

**What the judge returns (already defined in week 1):** JSON with
`preferred` (`A` or `B`), `preferred_rating` (1–10 on the winner), and
`syllable_correct_a` / `syllable_correct_b` (booleans). Preferences are
comparable because every pair is judged under every persona. The 1–10 is a
secondary intensity signal, not the primary outcome.

## Conditions

| condition | persona text (draft — to be written and length-matched) | frozen prediction (sign only) |
|---|---|---|
| **P0 none** | — | baseline |
| **P1 placebo** | a named person with no aesthetic values ("You are Sam. You live in a mid-sized city.") | no change on anything — controls for *having* a persona |
| **P2 formalist** | values strict 5-7-5; a haiku that breaks form fails regardless of imagery | `β_form` ↑ |
| **P3 imagist** | values vivid concrete imagery; treats English syllable counting as an arbitrary convention | `β_form` ↓ toward 0 |
| **P4 literalist** | values clear subject grounding; a haiku that is not clearly about the assigned subject fails | `β_topic` ↑ |

All persona texts **length-matched**, same reason as the CC3D placebo arms:
unmatched length confounds every contrast with prompt size. For every persona,
competence and nuisance biases have a null prediction ("no change"), tested
two-sided. A formalist that counts syllables *better* is still a finding (a
persona working as an effort lever), just a different one.

P4 is written as a **literalist**, not "a teacher." "Teacher" is already a
persona (and an underspecified one — English teacher? any subject?). The
prediction is about subject grounding, so the instruction should say that
directly.

**Deliberately excluded: demographic personas** (age, nationality, education).
They import stereotype questions and the population-proxy frame, and they
don't come with a checkable prediction. That is a different eval; it needs
human ratings and should be scoped separately.

## Protocol

- **Direct pairwise with the Mirror Test** (week-1 protocol), not PRePair.
  Self-preference only exists to perturb under direct judging. PRePair
  already took it to ~0 in week 2.
- **Judges:** gpt-4o-mini, claude-haiku-4-5, claude-sonnet-4-6 (all three
  authored in the pool, so self-bias is defined). Gemini flash-lite optional
  as the out-of-family check.
- **Pool:** the existing 60 haikus (`data/haikus_to_judge.jsonl`).
- **Scale:** 120 samples × 5 conditions × 3 judges = **1,800 calls**, about 5×
  week 1. Short prompts, so small.
- **Re-run P0 in the same batch.** Don't reuse week-1 logs as the baseline: a
  different run date means different sampling, and possibly different
  snapshots.
- **Run conditions blocked**, not interleaved: all P0, then all P1, and so
  on. Same model snapshot either way; blocked is easier to debug if a
  condition fails partway.

## Step 0: rerun 2026-10-01 with the CMU counter. Topic identifiable, form thin.

`L1err` is `|s1−5|+|s2−7|+|s3−5|` via `src/syllables_util.py`, which counts
with the CMU Pronouncing Dictionary since #5. Cosine is
`subject_cosine_full` recomputed with `all-MiniLM-L6-v2` (same embedder as
haiku-evals; the original generation-run scores are not in this repo).
Tables: `results/step0/`. The first run (2026-09-22) used the old
`syllables.estimate()` counter, which overcounted silent endings; its
numbers are kept below for comparison.

| check | first run (old counter) | rerun (CMU counter) |
|---|---|---|
| n | 60 haikus, 60 unique author-pairs | same |
| exact 5-7-5 | 10 / 60. Mean L1err 1.55 (range 0-4) | **53 / 60**. Mean L1err 0.12 (range 0-1) |
| mean cosine | 0.41 (range 0.16-0.62) | same |
| corr(Δform, Δtopic) | −0.10 | **+0.31** |
| form gap ≥ 1 | 45 / 60 | **12 / 60**, every one a 1-syllable gap |
| \|Δtopic\| ≥ 0.05 | 38 / 60 | 38 / 60 |
| tradeoff pairs (form and topic favor different haikus) | 27 / 60 (reported as 18) | **4 / 60** |

The first run reported 18 tradeoff pairs. That count compared the raw
deltas, where L1err is lower-is-better and cosine is higher-is-better, so
"opposite sign" meant the same haiku won both channels. The script now
counts on the `*_model` deltas, which share a direction.

**Topic is identifiable; form is thin.** For each judge, only 24 of its 120
judgments (12 pairs × 2 orientations) carry any form information, and each
one only asks "does the judge prefer the exact 5-7-5 haiku over the one
that's a syllable off?" Only 4 pairs separate form from topic, and where
form does differ, the better-formed haiku tends to also be the more
on-topic one (r = +0.31). So the P2 and P3 predictions on `β_form` may come
out inconclusive on this pool. P4 (`β_topic`), the P1 placebo, position
bias and self-preference don't need form variation. The competence
catch rate rests on just 7 off-form haikus (28 calls per judge), so it's
thin too.

Authors still don't dominate both channels. gpt-4o-mini is now closest to
5-7-5 (mean L1err 0.05, 19 of 20 exact), Haiku and Sonnet are at 0.15
(17 of 20), and Sonnet is still slightly highest on cosine (0.43).

**Off-form decision (Daniel, 2026-10-07): option A, keep this pool.** Do not
edit haikus into minimal pairs (that would break self-preference) and do
not add a looser-form generation run. Form is thin; report that. P4, P1,
position bias, and self-preference do not need more form variation. Paul
builds on the existing 60.

Step 0 is why magnitudes stay unfrozen: these numbers say whether the βs
can be told apart, not how large a persona should move them. There is no
prior persona run to take an effect size from.

## Build (small)

1. `prompts/personas/{none,placebo,formalist,imagist,literalist}.txt`
2. `-T persona=<name>` on the task. `pairwise_solver` prepends a
   `ChatMessageSystem` with the persona text when persona ≠ none.
3. `report.py`: persona column throughout, plus a `persona_effects.csv`
   (β's by judge × persona, and guardrail deltas vs. P0).
4. The prediction table above is frozen in this commit, **before** the
   first generation — same move as R10. Sign only; no numeric cutoffs.

Paul carries the Inspect build from this freeze.
Daniel owns the design, the frozen predictions, and the writeup framing.

## Split of work

- **Daniel:** design, Step 0 (done), frozen prediction table, writeup
  framing.
- **Paul:** Inspect build (persona prompt files, `-T persona=`, report
  column) from this freeze. Optional informal UI prompting on free models
  in parallel — qualitative only.
- **Compute:** ~1,800 short calls; a few dollars. Either of us can pay.
- **Repo:** this one. Branch + PR into `haiku-judge-evals` so the existing
  data and report pipeline stay in place. Credit Paul in the README when
  the build lands.

## Why this matters beyond haiku

On Aug 4 Daniel told Bobby he doesn't fully trust an LLM to score LLM output
that "looks right". An expert-persona close-reader is the obvious tool for
CC3D failure-mode discovery. This eval is the cheap sandbox, with ground
truth, where you find out whether a persona changes **what a judge can see**
or only **what it says it values**. Don't build that bridge now. It's the
reason the haiku result is worth having.

Paul's broader frame, which this sandbox is meant to inform rather than
replace: LLM judges are already part of scaled oversight because there isn't
enough human bandwidth. Role-assignment is a common user pattern; longer,
more specific prompts tend to be more reliable; and "who is grading" matters
most when the criterion is subjective. Existing multi-judge setups
(pairwise debate plus a third judge) usually don't specify a persona beyond
for/against. A later design could treat personas as a counsel — product /
engineering / data science each advocating from their strengths — but that
is not this round.

Lit check happens at writeup time, not before building. Persona effects on LLM
judges and on reasoning are an active area, so don't claim novelty.

---

## Paul's notes (2026-09)

Folded in from the annotated copy, so they live on this PR.

**Hypothesis (Paul).** Assigning a persona should make judging more
consistent across models (lower between-model variance) and reduce
self-preference vs. the no-persona control.

Settled: that is not a co-primary. This round treats directional
taste-weight shifts as the thing a persona is launched to move. Models
trending together on those signs, and do-no-harm on self-preference, are
guardrails. Between-model agreement (models picking the same winner more
often) is not a study question. It remains measurable from the same run.

**Stated vs. assigned persona.** First log whatever persona the judge
chooses, then assign one, then maybe join several as a counsel. Two
prompt-order variants for the log arm: judge first then ask what persona was
used, or state a persona first then judge. Length-matching still matters if
that arm is in scope.

**P4 "teacher" and extra riffs.** "Teacher" is already a persona. Extra
sketches: an engineering student peer-reviewing against given haiku
standards (strict); a teacher reading a short book of haiku over breakfast
(same role, relaxed, not grading). Core contrast: creative vs. strict.

**Prompt order.** Does `You are XYZ, judge this` vs. `judge this, acting as
XYZ` change the pick?

**Sampling.** Sequential mix of P0–P4 vs. all P0, then all P1, etc.

**Replies to the original open questions.** Three-outcome framing is fine;
demographics can stay out. No strong opinion on git structure. Happy to
chip in on compute. (The "freeze expected β thresholds" note is superseded:
signs are frozen, magnitudes are not.)

---

## Frozen calls (Daniel sign-off, 2026-09-22)

1. **Assigned-persona 5-arm design (P0–P4) for this round.** Defer "state
   your persona," prompt-order swaps, and counsel-of-personas.
2. **P4 is a literalist, not a teacher.**
3. **Run conditions blocked** (all P0, then all P1, …).
4. **Predictions are signs, not magnitudes.** P1 no change; P2 `β_form` ↑;
   P3 `β_form` ↓ toward 0; P4 `β_topic` ↑. Any signed movement is data.
   No numeric cutoff — Step 0 cannot supply one.
5. **Paul takes the Inspect build from this commit.**
6. **Guardrails, not a co-primary.** Same-direction taste-weight shifts
   across models, and do-no-harm on self-preference. Not studying whether
   models agree with each other more. Not re-opening week-2 self-preference
   as a primary.
7. **Keep the existing 60-haiku pool (2026-10-07).** Option A. Form is
   thin (12 pairs, all 1-syllable, 4 real tradeoffs); P2/P3 may be
   inconclusive and that is fine. Competence is catch-rate vs false-flag
   against CMU counts. No new generation run. Paul takes the Inspect build.
