"""Educational Tools
====================
Helpers for explanations and lightweight progress tracking used in examples.

Functions:
- explain_concept(name, description, key_points=None)
- format_equation(latex_str)
- checkpoint(progress_log: list, label: str)
- create_quiz(questions)
- administer_quiz(quiz)
- grade_quiz(quiz, answers)

These keep side-effects minimal (stdout printing) so they are safe in notebooks.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Dict
import textwrap
from dataclasses import dataclass


def explain_concept(
    name: str, description: str, key_points: Optional[List[str]] = None
) -> str:
    """Return a formatted explanation block for a concept."""
    header = f"📘 Concept: {name}\n" + "-" * (len(name) + 11)
    desc = textwrap.fill(description, width=88)
    points = ""
    if key_points:
        bullets = "\n".join(f"  • {p}" for p in key_points)
        points = f"\nKey Points:\n{bullets}"
    block = f"{header}\n{desc}{points}\n"
    print(block)
    return block


def format_equation(latex_str: str) -> str:
    """Return a LaTeX equation string (placeholder for richer renderers)."""
    eq = f"Equation: ${latex_str}$"
    print(eq)
    return eq


def checkpoint(progress_log: List[str], label: str) -> None:
    """Append a progress label and print a concise status."""
    progress_log.append(label)
    print(f"✅ Reached checkpoint: {label} (total {len(progress_log)})")


# -------------------- Quiz Utilities --------------------
@dataclass
class QuizQuestion:
    prompt: str
    choices: Sequence[str]
    answer_index: int
    explanation: str = ""

    def display(self) -> None:
        print(self.prompt)
        for i, c in enumerate(self.choices, 1):
            print(f"  {i}. {c}")


def create_quiz(questions: Sequence[QuizQuestion]) -> List[QuizQuestion]:
    """Create a quiz object (list wrapper for now)."""
    return list(questions)


def administer_quiz(quiz: Sequence[QuizQuestion]) -> List[int]:
    """Run quiz in console and collect selected indices (1-based)."""
    responses: List[int] = []
    for q in quiz:
        q.display()
        while True:
            try:
                raw = input("Your answer (number): ").strip()
                choice = int(raw)
                if 1 <= choice <= len(q.choices):
                    responses.append(choice - 1)
                    break
                else:
                    print("Please enter a valid choice number.")
            except Exception:
                print("Invalid input; enter the choice number.")
        print()
    return responses


def grade_quiz(
    quiz: Sequence[QuizQuestion], responses: Sequence[int]
) -> Dict[str, float]:
    """Grade quiz and print per-question feedback.
    Returns summary dict with score and percentage.
    """
    correct = 0
    for i, (q, r) in enumerate(zip(quiz, responses), 1):
        is_correct = r == q.answer_index
        if is_correct:
            correct += 1
        status = (
            "✅ Correct"
            if is_correct
            else f"❌ Incorrect (correct: {q.choices[q.answer_index]})"
        )
        print(f"Q{i}: {status}")
        if q.explanation:
            print("   " + textwrap.fill(q.explanation, width=86))
    total = len(quiz)
    pct = (correct / total * 100) if total else 0.0
    print(f"\nScore: {correct}/{total} ({pct:.1f}%)")
    return {"correct": correct, "total": total, "percent": pct}


if __name__ == "__main__":
    log: List[str] = []
    explain_concept(
        "Superposition",
        "A qubit can exist in a linear combination of |0> and |1> states.",
        ["Amplitude", "Measurement collapse", "Basis vectors"],
    )
    format_equation(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle")
    checkpoint(log, "Introduced superposition")

    sample_quiz = create_quiz(
        [
            QuizQuestion(
                prompt="What does measurement do to a qubit in superposition?",
                choices=[
                    "Leaves it unchanged",
                    "Collapses it to a basis state",
                    "Clones the superposition",
                    "Amplifies amplitudes",
                ],
                answer_index=1,
                explanation="Standard projective measurement collapses the state to |0> or |1> proportional to squared amplitudes.",
            )
        ]
    )
    # Skip interactive administer in module self-test to avoid blocking.
    grade_quiz(sample_quiz, [1])
