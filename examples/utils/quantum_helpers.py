"""
Quantum Helpers
================
Utility quantum circuit/state preparation helpers used across examples.

Functions:
- create_bell_state() -> QuantumCircuit
- prepare_plus_state() -> QuantumCircuit
- apply_random_single_qubit_rotation(seed=None) -> QuantumCircuit
- measure_all(circuit) -> QuantumCircuit (copy with measurements)
- analyze_state(state) -> dict with probabilities & bloch vector (single qubit)

These helpers intentionally stay lightweight so they can be copied into
notebooks or extended without extra dependencies beyond Qiskit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
import random

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def create_bell_state() -> QuantumCircuit:
    """Return a circuit preparing the |Φ+> Bell state.
    Returns:
        QuantumCircuit with 2 qubits entangled in (|00> + |11>)/sqrt(2)
    """
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def prepare_plus_state() -> QuantumCircuit:
    """Return a 1-qubit circuit preparing |+> state."""
    qc = QuantumCircuit(1, name="plus")
    qc.h(0)
    return qc


def apply_random_single_qubit_rotation(seed: Optional[int] = None) -> QuantumCircuit:
    """Return a 1-qubit circuit with a random rotation (for experimentation).
    Args:
        seed: Optional random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, 2 * math.pi)
    lam = random.uniform(0, 2 * math.pi)
    qc = QuantumCircuit(1, name="random_u3")
    qc.u(theta, phi, lam, 0)
    return qc


def measure_all(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a shallow copy of circuit with measurements on all qubits."""
    qc = circuit.copy()
    qc.measure_all()
    return qc


def analyze_state(state) -> Dict[str, Any]:
    """Analyze a statevector or circuit producing a single-qubit state.
    Args:
        state: Statevector or QuantumCircuit (1 qubit) or object convertible.
    Returns:
        dict with keys: probabilities, bloch_vector (x,y,z)
    """
    if not isinstance(state, Statevector):
        state = Statevector.from_instruction(state)
    if int(math.log2(state.dim)) != 1:
        raise ValueError("analyze_state currently supports single-qubit states only")
    # Probabilities
    probs = [abs(a) ** 2 for a in state.data]
    # Bloch vector using expectations
    alpha, beta = state.data
    # <X> = 2 Re(alpha* beta*)? Actually <X> = 2 Re(alpha* beta_conj)
    bloch_x = 2 * (alpha.conjugate() * beta).real
    # <Y> = 2 Im(beta* alpha*)? Proper sign: <Y> = 2 * (alpha.conjugate()*beta).imag
    bloch_y = 2 * (alpha.conjugate() * beta).imag
    # <Z> = |alpha|^2 - |beta|^2
    bloch_z = abs(alpha) ** 2 - abs(beta) ** 2
    return {
        "probabilities": {"0": probs[0], "1": probs[1]},
        "bloch_vector": (bloch_x, bloch_y, bloch_z)
    }

if __name__ == "__main__":
    bell = create_bell_state()
    plus = prepare_plus_state()
    rand = apply_random_single_qubit_rotation(seed=0)
    analysis = analyze_state(plus)
    print("Bell depth:", bell.depth())
    print("Plus analysis:", analysis)
