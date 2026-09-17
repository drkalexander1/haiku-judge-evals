# Persona judges — design sketch

**Status:** design review with [Paul Davidson](https://github.com/PaulsForge).
Nothing built, nothing run. This is the third run on the haiku-judge branch
(week 1 direct pairwise, week 2 PRePair + Gemini). Comment on the PR rather
than in a sidecar copy of this file.

**Round number:** TBD (week 3 if the existing pool is identifiable; otherwise
a designed stimulus set comes first — see Step 0).

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

A persona's effect is the shift in `β_form` / `β_topic` relative to the
no-persona baseline, within judge. Every pair is judged under every
condition, so this is a paired design and the baseline judgment of the same
pair is available as a covariate.

This is the random-utility / discrete-choice machinery from stated-preference
economics, pointed at a model. It's the same toolkit the valuation eval uses,
which is worth saying out loud in a writeup.

**What the judge returns (already defined in week 1):** JSON with
`preferred` (`A` or `B`), `preferred_rating` (1–10 on the winner), and
`syllable_correct_a` / `syllable_correct_b` (booleans). Preferences are
comparable because every pair is judged under every persona. The 1–10 is a
secondary intensity signal, not the primary outcome.

## Conditions

| condition | persona text (draft — to be written and length-matched) | frozen prediction |
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

## Step 0 — free, and gates everything else

Before any API call: from the existing pool, compute `L1err` and `cos` per
haiku, then check
1. the correlation between Δform and Δtopic across the 60 pairs, and
2. how many pairs have a real gap on each.

Only 10 of 60 haikus are exact 5-7-5, and the features may track author
(Sonnet better on both). If the two features are highly correlated or barely
vary, the β's aren't identifiable from this pool. Then Phase 2 comes
first: a **designed stimulus set** that crosses form × grounding
(perfect-form/off-subject, broken-form/on-subject, …) so the weights separate
by construction.

`haikus_to_judge.jsonl` carries only `syllable_perfect_actual`. `L1err` can
be recomputed with `src/syllables_util.py`, and `cos` joined from the haiku-evals
run (verify the field is in its `predictions.jsonl`).

Numeric β thresholds get frozen **after** Step 0, once we know whether the
features separate. Directional predictions (the table above) can be frozen
now; magnitudes cannot.

Daniel owns Step 0. It is a design gate, not an Inspect build. After it
lands, we look at the table together on this PR.

## Build (small)

1. `prompts/personas/{none,placebo,formalist,imagist,literalist}.txt`
2. `-T persona=<name>` on the task. `pairwise_solver` prepends a
   `ChatMessageSystem` with the persona text when persona ≠ none.
3. `report.py`: persona column throughout, plus a `persona_effects.csv`
   (β's by judge × persona, and guardrail deltas vs. P0).
4. Freeze the prediction table above in a commit **before** the first
   generation — same move as R10.

Paul carries the Inspect build once Step 0 says the pool is usable.
Daniel owns the design, the frozen predictions, and the writeup framing.

## Split of work

- **Daniel:** design, Step 0 identifiability check, frozen prediction table,
  writeup framing.
- **Paul:** Inspect build (persona prompt files, `-T persona=`, report
  column) after the freeze commit. Optional informal UI prompting on free
  models in parallel for an early qualitative read — not a substitute for
  Step 0.
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

That is a different primary claim than the β table above. This round still
treats directional taste-weight shifts as the thing a persona is launched to
move, and between-model agreement / self-preference as outcomes we will
report (self-preference is already a guardrail: predicted not to move). If
the interesting result is "personas make judges agree with each other," that
is measurable from the same run and can be promoted in the writeup if it
shows up. Confirming that split is one of the asks on this PR.

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
freeze expected β thresholds ahead of time; demographics can stay out. No
strong opinion on git structure. Happy to chip in on compute. Next week is
probably okay; unclear how to contribute to Step 0 — maybe look at outputs
together, or do informal UI prompting on free models for an early read.

---

## Proposed calls for this PR

Please comment if any of these is the wrong call. Otherwise this is what
Step 0 and the later build will follow.

1. **Keep the assigned-persona 5-arm design (P0–P4) for this round.** Defer
   "state your persona," prompt-order swaps, and counsel-of-personas to a
   later round. Those are real questions; they are also extra cells, and
   they don't have the same checkable β predictions.
2. **Keep P4 as a literalist, not a teacher.** Breakfast-teacher and
   engineering-student can wait, or replace a cell later if P4 looks too
   close to P2 (both "strict").
3. **Run conditions blocked** (all P0, then all P1, …).
4. **Freeze directional predictions now; freeze numeric β thresholds after
   Step 0.**
5. **Daniel does Step 0; Paul takes the Inspect build after the freeze
   commit.** Informal UI prompting is optional and parallel, not the next
   required step.

Still open, and useful to settle on this PR: is Paul's between-model
consistency claim a co-primary we should pre-register, or a secondary we
report and promote if it shows up?
