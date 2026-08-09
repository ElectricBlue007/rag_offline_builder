"""
evaluation/eval_dataset.py

Defines the structure for a hand-curated evaluation set and provides
JSON load/save utilities. This is deliberately simple (a flat JSON list)
rather than anything fancier — the evaluation set is meant to be
readable and editable by a human, not a generated artifact.

The original evaluation plan calls for 20-30 questions mixing English
and Spanish, and mixing questions that hit tables vs. prose vs. image
descriptions. evaluation/sample_eval_questions.json ships a real
starter set (12 questions) built directly from this project's own
fixture files (Binder2.md, Testing.md) as a genuine, answerable
template — expand it with your own documents' real content for a more
robust evaluation.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union


@dataclass
class EvalQuestion:
    question: str
    ground_truth: str
    # Free-form tags for breaking down results by coverage — not used
    # by the scoring logic itself, purely for the human reviewing
    # results to see "are my Spanish questions scoring worse than
    # English ones?" etc.
    language: str = "en"
    category: str = ""  # e.g. "table", "prose", "image_description", "multi_hop"


def load_eval_questions(path: Union[str, Path]) -> list[EvalQuestion]:
    """Load a list of EvalQuestion from a JSON file (a flat list of objects)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalQuestion(**item) for item in data]


def save_eval_questions(questions: list[EvalQuestion], path: Union[str, Path]) -> None:
    """Save a list of EvalQuestion to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(q) for q in questions]
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
