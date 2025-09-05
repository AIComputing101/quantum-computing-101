# Quantum Computing 101 - Shared Utilities

This package contains shared utilities and helper functions used across example modules. All modules are lightweight and safe for notebook usage.

## 📁 Contents

### `visualization.py`
- Bloch sphere & state comparison
- Circuit diagram with explanation blocks
- Measurement result plotting & probability evolution
- Quantum vs Classical distribution comparisons

### `quantum_helpers.py`
- Common state preparations (Bell, plus, random rotation)
- Measurement helper
- Single-qubit state analysis (probabilities + Bloch vector)

### `classical_helpers.py`
- Function timing & benchmarking
- Basic statistics utilities
- Hamming distance
- Probability distribution normalization

### `educational_tools.py`
- Concept explanation formatting
- Lightweight equation wrapper
- Progress checkpoints
- Quiz creation, administration & grading utilities

### `cli.py`
- Command line interface to list and run example scripts (`quantum101` entry point)

## 🔧 Usage

Preferred (package import after installation or adding `examples` to `PYTHONPATH`):
```python
from utils import (
    create_bell_state, plot_bloch_comparison,
    explain_concept, create_quiz, QuizQuestion, grade_quiz
)

bell = create_bell_state()
explain_concept("Entanglement", "Non-classical correlations between qubits.")
quiz = create_quiz([
    QuizQuestion("State of Bell pair?", ["|01>+|10>", "|00>+|11>", "|00>", "|11>"], 1)
])
result = grade_quiz(quiz, [1])
```

Ad-hoc (inside an example without installation):
```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from visualization import plot_bloch_comparison
```

## 📋 Dependencies

Core utilities rely on the project example dependencies:
- qiskit
- matplotlib
- numpy
- seaborn (visual style)

Optional:
- networkx (only for certain advanced visual examples, not required by helpers)

## ✅ Design Principles
- Minimal external dependencies
- Clear, documented function contracts
- Safe to import in constrained notebook kernels
- Easy to extend without breaking examples

## 🛠 Contributing
Add new helpers in a focused module; update `__init__.py` exports and list them here. Provide a short self-test in a `__main__` block without heavy runtime cost.
