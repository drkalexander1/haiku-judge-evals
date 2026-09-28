"""Recompute syllable_perfect_actual in data/haikus_to_judge.jsonl in place.

ingest.py normally sets this field, but it needs the source haiku-evals run,
which isn't in this repo. This script rewrites only that field, using the
haiku text already in the file, so the pool can be recounted whenever
src/syllables_util.py changes. Every other field is left as is.

Usage:
    python scripts/recount_syllables.py            # rewrite the file
    python scripts/recount_syllables.py --dry-run  # only list what would change
"""

from __future__ import annotations

import argparse

from src.schema import HAIKUS_PATH, load_haikus_to_judge
from src.syllables_util import line_syllable_counts, syllable_perfect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recount syllable_perfect_actual from the stored haiku text")
    parser.add_argument("--dry-run", action="store_true", help="List changes without writing the file")
    args = parser.parse_args(argv)

    haikus = load_haikus_to_judge()
    changed = 0
    for h in haikus:
        lines = [h.line1, h.line2, h.line3]
        new = syllable_perfect(lines)
        if new != h.syllable_perfect_actual:
            changed += 1
            print(f"{h.syllable_perfect_actual!s:>5} -> {new!s:<5} {line_syllable_counts(lines)}  {h.judge_sample_id}")
            h.syllable_perfect_actual = new

    print(f"{changed} of {len(haikus)} changed; {sum(h.syllable_perfect_actual for h in haikus)} now exactly 5-7-5")
    if not args.dry_run:
        with HAIKUS_PATH.open("w", encoding="utf-8", newline="\n") as f:  # keep LF on Windows
            for h in haikus:
                f.write(h.model_dump_json() + "\n")
        print(f"Wrote {HAIKUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
