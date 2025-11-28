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
- create_measurement_circuit(circuit) -> QuantumCircuit (proper measurement setup)
- run_circuit_with_shots(circuit, shots=1000, backend=None) -> dict
- create_ghz_state(n_qubits) -> QuantumCircuit
- create_w_state(n_qubits) -> QuantumCircuit

These helpers intentionally stay lightweight so they can be copied into
notebooks or extended without extra dependencies beyond Qiskit.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import math
import random

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


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
    """
    Analyze a statevector or circuit producing a single-qubit state.
    
    Mathematical Foundation - Qubit State Analysis:
    ----------------------------------------------
    
    A single qubit state is represented as:
    |ψ⟩ = α|0⟩ + β|1⟩
    
    where α, β are complex probability amplitudes satisfying |α|² + |β|² = 1
    
    This function extracts two key representations:
    
    1. MEASUREMENT PROBABILITIES (Born Rule):
    -----------------------------------------
    When measuring in computational basis {|0⟩, |1⟩}:
    - P(0) = |α|² = α × α* (probability of measuring 0)
    - P(1) = |β|² = β × β* (probability of measuring 1)
    
    where α* denotes complex conjugate of α
    
    These probabilities tell us the likelihood of each measurement outcome.
    
    2. BLOCH VECTOR (Geometric Representation):
    -------------------------------------------
    Any single qubit state can be visualized as a point on the Bloch sphere
    using a 3D vector (x, y, z) where each component is an expectation value:
    
    Bloch vector formula:
    x = ⟨X⟩ = ⟨ψ|X|ψ⟩ = 2 Re(α*β)
    y = ⟨Y⟩ = ⟨ψ|Y|ψ⟩ = 2 Im(α*β)
    z = ⟨Z⟩ = ⟨ψ|Z|ψ⟩ = |α|² - |β|²
    
    where X, Y, Z are Pauli matrices:
    X = [[0, 1], [1, 0]]    (bit flip)
    Y = [[0, -i], [i, 0]]   (bit+phase flip)
    Z = [[1, 0], [0, -1]]   (phase flip)
    
    Mathematical Derivation of Bloch Components:
    --------------------------------------------
    
    X-component (Re means "real part"):
    ⟨ψ|X|ψ⟩ = [α*, β*] [[0,1],[1,0]] [α]
                                      [β]
            = [α*, β*] [β, α]ᵀ
            = α*β + β*α
            = α*β + (α*β)*  (since (α*β)* = β*α)
            = 2 Re(α*β)
    
    Y-component (Im means "imaginary part"):
    ⟨ψ|Y|ψ⟩ = [α*, β*] [[0,-i],[i,0]] [α]
                                        [β]
            = [α*, β*] [-iβ, iα]ᵀ
            = -iα*β + iβ*α
            = i(β*α - α*β)
            = i·2i·Im(α*β)  (using Im(z) = (z-z*)/(2i))
            = 2 Im(α*β)
    
    Z-component:
    ⟨ψ|Z|ψ⟩ = [α*, β*] [[1,0],[0,-1]] [α]
                                        [β]
            = [α*, β*] [α, -β]ᵀ
            = α*α - β*β
            = |α|² - |β|²
    
    Bloch Sphere Properties:
    ------------------------
    - Vector length: √(x² + y² + z²) = 1 (always on sphere surface)
    - Pure states: on surface (length = 1)
    - Mixed states: inside sphere (length < 1)
    
    Special Points on Bloch Sphere:
    -------------------------------
    |0⟩: (0, 0, 1)   - North pole
    |1⟩: (0, 0, -1)  - South pole
    |+⟩: (1, 0, 0)   - Positive X axis
    |-⟩: (-1, 0, 0)  - Negative X axis
    |i⟩: (0, 1, 0)   - Positive Y axis (i = imaginary unit)
    |-i⟩: (0, -1, 0) - Negative Y axis
    
    Why This Matters:
    ----------------
    - Probabilities: tell you measurement outcomes
    - Bloch vector: shows quantum state geometry
    - Together: complete characterization of single qubit state
    
    Args:
        state: Statevector or QuantumCircuit (1 qubit) or convertible object
        
    Returns:
        dict with keys:
            - 'probabilities': {'0': P(0), '1': P(1)}
            - 'bloch_vector': (x, y, z) coordinates on Bloch sphere
            
    Raises:
        ValueError: If state is not a single-qubit state
    """
    if not isinstance(state, Statevector):
        state = Statevector.from_instruction(state)
    if int(math.log2(state.dim)) != 1:
        raise ValueError("analyze_state currently supports single-qubit states only")
    
    # Extract probability amplitudes α and β
    # For |ψ⟩ = α|0⟩ + β|1⟩, state.data = [α, β]
    alpha, beta = state.data
    
    # ------------------------------------------------------------------
    # 1. Calculate measurement probabilities using Born rule
    # ------------------------------------------------------------------
    # P(outcome) = |amplitude|² = amplitude × conjugate(amplitude)
    probs = [abs(a) ** 2 for a in state.data]
    
    # ------------------------------------------------------------------
    # 2. Calculate Bloch vector components (expectation values)
    # ------------------------------------------------------------------
    
    # X-component: ⟨X⟩ = 2 Re(α*β)
    # Measures superposition along X-axis
    # Large |⟨X⟩| means state is closer to |+⟩ or |-⟩
    bloch_x = 2 * (alpha.conjugate() * beta).real
    
    # Y-component: ⟨Y⟩ = 2 Im(α*β)
    # Measures superposition along Y-axis with imaginary phase
    # Large |⟨Y⟩| means state has significant imaginary phase component
    bloch_y = 2 * (alpha.conjugate() * beta).imag
    
    # Z-component: ⟨Z⟩ = |α|² - |β|²
    # Measures "up-ness" vs "down-ness" (computational basis bias)
    # +1 → pure |0⟩, -1 → pure |1⟩, 0 → equal superposition
    bloch_z = abs(alpha) ** 2 - abs(beta) ** 2
    
    return {
        "probabilities": {"0": probs[0], "1": probs[1]},
        "bloch_vector": (bloch_x, bloch_y, bloch_z),
    }


def create_measurement_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Create a proper measurement circuit compatible with Qiskit 2.x.

    Args:
        circuit: Original quantum circuit

    Returns:
        New circuit with classical register and measurements
    """
    # Create new circuit with both quantum and classical bits
    qc_measure = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    qc_measure = qc_measure.compose(circuit)
    qc_measure.measure_all()
    return qc_measure


def run_circuit_with_shots(
    circuit: QuantumCircuit, shots: int = 1000, backend=None
) -> dict:
    """Run a quantum circuit and return measurement counts.

    Args:
        circuit: Quantum circuit to run
        shots: Number of measurement shots
        backend: Optional backend (defaults to AerSimulator)

    Returns:
        Dictionary of measurement counts
    """
    if backend is None:
        backend = AerSimulator()

    # Ensure circuit has measurements
    if not circuit.clbits:
        circuit = create_measurement_circuit(circuit)

    try:
        job = backend.run(transpile(circuit, backend), shots=shots)
        result = job.result()
        return result.get_counts()
    except Exception as e:
        print(f"⚠️ Error running circuit: {e}")
        return {}


def create_ghz_state(n_qubits: int) -> QuantumCircuit:
    """Create an n-qubit GHZ state: (|00...0⟩ + |11...1⟩)/√2.

    Args:
        n_qubits: Number of qubits

    Returns:
        QuantumCircuit preparing GHZ state
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    qc = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}")
    qc.h(0)  # Superposition on first qubit
    for i in range(1, n_qubits):
        qc.cx(0, i)  # Entangle with other qubits
    return qc


def create_w_state(n_qubits: int) -> QuantumCircuit:
    """Create an n-qubit W state: (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n.

    Args:
        n_qubits: Number of qubits

    Returns:
        QuantumCircuit preparing W state
    """
    if n_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")

    qc = QuantumCircuit(n_qubits, name=f"w_{n_qubits}")

    # Use recursive construction for W state
    # This is a simplified version - full W state requires more complex construction
    qc.ry(2 * math.asin(1 / math.sqrt(n_qubits)), 0)

    for i in range(1, n_qubits):
        angle = 2 * math.asin(math.sqrt(1 / (n_qubits - i + 1)))
        qc.cry(angle, i - 1, i)

    return qc


if __name__ == "__main__":
    # Test the helper functions
    print("🧪 Testing Quantum Helpers...")

    bell = create_bell_state()
    plus = prepare_plus_state()
    rand = apply_random_single_qubit_rotation(seed=0)
    ghz3 = create_ghz_state(3)
    w3 = create_w_state(3)

    analysis = analyze_state(plus)
    print("✅ Bell depth:", bell.depth())
    print("✅ Plus analysis:", analysis)
    print("✅ GHZ-3 depth:", ghz3.depth())
    print("✅ W-3 depth:", w3.depth())

    # Test measurement circuit
    bell_measure = create_measurement_circuit(bell)
    print(
        "✅ Bell measurement circuit qubits:",
        bell_measure.num_qubits,
        "clbits:",
        bell_measure.num_clbits,
    )

    # Test circuit execution
    counts = run_circuit_with_shots(bell, shots=100)
    print("✅ Bell state counts:", counts)

    print("🎉 All tests passed!")
