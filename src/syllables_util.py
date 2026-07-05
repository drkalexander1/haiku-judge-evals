"""English syllable counting for 5-7-5 haiku form scoring.

Copied from ../Haiku-evals/src/syllables_util.py so this repo has no
cross-repo import dependency. Keep in sync if the counting logic changes.
"""

from __future__ import annotations

import re

import syllables

TARGET_SYLLABLES = (5, 7, 5)

_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)


def count_syllables(text: str) -> int:
    """Estimate syllables in a line (punctuation stripped, words summed)."""
    cleaned = _PUNCT_RE.sub(" ", text).strip()
    if not cleaned:
        return 0
    total = 0
    for word in cleaned.split():
        w = word.strip("-'")
        if not w:
            continue
        total += max(1, syllables.estimate(w))
    return total


def line_syllable_counts(lines: list[str]) -> list[int]:
    return [count_syllables(line) for line in lines]


def syllable_perfect(lines: list[str], target: tuple[int, ...] = TARGET_SYLLABLES) -> bool:
    counts = line_syllable_counts(lines)
    return len(counts) == len(target) and counts == list(target)
