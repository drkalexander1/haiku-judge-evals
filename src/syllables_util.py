"""English syllable counting for 5-7-5 haiku form scoring.

Counts come from the CMU Pronouncing Dictionary (``cmudict``). A word CMU
doesn't list falls back to ``syllables.estimate()``, a spelling heuristic.
The heuristic is much less accurate: it overcounts words with silent endings
like "waves" (2) or "embrace" (3).

Words with more than one pronunciation ("hour" is 1 or 2 syllables) are
resolved toward the target, so a line counts as 5 if any mix of listed
pronunciations gives 5. Known shortcomings:

- It's lenient. CMU lists some unusual variants (e.g. "spring" as 2
  syllables), and a variant like that can hide a line that's really off.
- Words missing from CMU (names, coinages, typos) fall back to the heuristic.

This used to be a copy of ../haiku-evals/src/syllables_util.py. haiku-evals
still uses the heuristic for every word, so the two repos now disagree on
most haikus.
"""

from __future__ import annotations

import re
from functools import lru_cache

import cmudict
import syllables

TARGET_SYLLABLES = (5, 7, 5)

_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def word_syllable_options(word: str) -> set[int]:
    """Every syllable count CMU lists for a word, e.g. "hour" -> {1, 2}.

    A hyphenated word missing from CMU is split and its parts combined
    ("tick-tock" -> {2}). Anything else falls back to the heuristic.
    """
    w = word.strip("-'").lower()
    prons = _cmu().get(w)
    if prons:
        return {sum(1 for phone in pron if phone[-1].isdigit()) for pron in prons}
    parts = [p for p in w.split("-") if p]
    if len(parts) > 1:
        totals = {0}
        for part in parts:
            totals = {t + n for t in totals for n in word_syllable_options(part)}
        return totals
    return {max(1, syllables.estimate(w))}


def line_syllable_options(text: str) -> set[int]:
    """Every total a line can reach, picking one pronunciation per word."""
    cleaned = _PUNCT_RE.sub(" ", text).strip()
    totals = {0}
    for word in cleaned.split():
        if not word.strip("-'"):
            continue
        totals = {t + n for t in totals for n in word_syllable_options(word)}
    return totals


def count_syllables(text: str, target: int | None = None) -> int:
    """Syllables in a line: the reading closest to ``target`` (ties go to the
    lower count), or the lowest reading if no target is given."""
    options = line_syllable_options(text)
    if target is None:
        return min(options)
    return min(options, key=lambda n: (abs(n - target), n))


def line_syllable_counts(lines: list[str], target: tuple[int, ...] = TARGET_SYLLABLES) -> list[int]:
    return [count_syllables(line, target[i] if i < len(target) else None) for i, line in enumerate(lines)]


def syllable_perfect(lines: list[str], target: tuple[int, ...] = TARGET_SYLLABLES) -> bool:
    counts = line_syllable_counts(lines, target)
    return len(counts) == len(target) and counts == list(target)
