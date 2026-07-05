"""Pydantic models for the judge eval: haikus-to-judge and pairwise judge ratings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HAIKUS_PATH = DATA_DIR / "haikus_to_judge.jsonl"
PROMPT_PATH = ROOT / "prompts" / "judge_pairwise_v1.txt"
POINTWISE_PROMPT_PATH = ROOT / "prompts" / "judge_pointwise_v1.txt"
PREPAIR_FINAL_PROMPT_PATH = ROOT / "prompts" / "judge_pairwise_prepair_v1.txt"


class HaikuToJudge(BaseModel):
    """One haiku pulled from a source generation run, ready to be judged blind.

    author_model and syllable_perfect_actual are ground truth kept out of the
    judge prompt -- used only at analysis time to measure self-preference bias
    and judge accuracy.
    """

    judge_sample_id: str
    scenario_id: str
    subject: str
    stratum: str
    prompt_variant: str
    author_model: str
    line1: str
    line2: str
    line3: str
    syllable_perfect_actual: bool

    def full_text(self) -> str:
        return "\n".join([self.line1, self.line2, self.line3])


class JudgePairRating(BaseModel):
    preferred: Literal["A", "B"] = Field(description="Which haiku the judge prefers overall")
    preferred_rating: int = Field(ge=1, le=10, description="Quality rating (1-10) of whichever haiku was preferred")
    syllable_correct_a: bool = Field(description="True if Haiku A follows the 5-7-5 syllable pattern")
    syllable_correct_b: bool = Field(description="True if Haiku B follows the 5-7-5 syllable pattern")


class PointwiseCritique(BaseModel):
    """A single haiku analyzed in isolation, with no visibility of the haiku it will
    eventually be compared against -- the PRePair step that breaks the "Comparative Trap"
    (Jeong et al. 2025) by forcing rubric-grounded analysis before any side-by-side framing."""

    line1_syllables: int
    line2_syllables: int
    line3_syllables: int
    syllable_correct: bool = Field(description="True if the pattern is exactly 5-7-5")
    has_kigo_or_season_word: bool = Field(description="True if there's a traditional seasonal word/image")
    imagery_assessment: str = Field(description="1-2 sentence assessment of imagery and evocativeness")
    overall_quality_1_to_10: int = Field(ge=1, le=10, description="Quality of this haiku alone, on its own merits")


def load_haikus_to_judge(path: Path | None = None) -> list[HaikuToJudge]:
    path = path or HAIKUS_PATH
    items: list[HaikuToJudge] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(HaikuToJudge.model_validate_json(line))
    return items


def load_judge_prompt_template(path: Path | None = None) -> str:
    path = path or PROMPT_PATH
    return path.read_text(encoding="utf-8")


def load_pointwise_prompt_template(path: Path | None = None) -> str:
    path = path or POINTWISE_PROMPT_PATH
    return path.read_text(encoding="utf-8")


def load_prepair_final_prompt_template(path: Path | None = None) -> str:
    path = path or PREPAIR_FINAL_PROMPT_PATH
    return path.read_text(encoding="utf-8")
