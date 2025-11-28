#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 2
Quantum Gates and Circuits

This example demonstrates basic quantum gates and how to build quantum circuits,
showing the effects of different gates on qubit states.

Learning objectives:
- Understand basic quantum gates (X, Y, Z, H, S, T)
- Build quantum circuits step by step
- Visualize gate effects on qubit states
- Learn about single and multi-qubit gates

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_multivector, circuit_drawer
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def demonstrate_single_qubit_gates():
    """
    Demonstrate the effect of single-qubit gates.
    
    Mathematical Foundation - Quantum Gates:
    ----------------------------------------
    Quantum gates are unitary operators that transform qubit states.
    For a gate represented by matrix U, acting on state |ψ⟩:
    
    |ψ'⟩ = U|ψ⟩
    
    Unitarity Condition:
    -------------------
    A matrix U is unitary if: U†U = I (where † means conjugate transpose)
    This ensures:
    1. Probability conservation: ⟨ψ'|ψ'⟩ = 1
    2. Reversibility: U† reverses the operation (U†U = I)
    3. Information preservation (no-information loss)
    
    Common Single-Qubit Gates:
    ---------------------------
    
    1. IDENTITY (I):
       Matrix: I = [[1, 0],
                    [0, 1]]
       Effect: I|ψ⟩ = |ψ⟩ (no change)
    
    2. PAULI-X (Quantum NOT):
       Matrix: X = [[0, 1],
                    [1, 0]]
       Effect: X|0⟩ = |1⟩, X|1⟩ = |0⟩
       Bloch sphere: 180° rotation around X-axis
    
    3. PAULI-Y:
       Matrix: Y = [[0, -i],
                    [i,  0]]
       Effect: Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
       Bloch sphere: 180° rotation around Y-axis
    
    4. PAULI-Z (Phase flip):
       Matrix: Z = [[1,  0],
                    [0, -1]]
       Effect: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
       Bloch sphere: 180° rotation around Z-axis
    
    5. HADAMARD (H):
       Matrix: H = (1/√2)[[1,  1],
                          [1, -1]]
       Effect: H|0⟩ = |+⟩ = (|0⟩+|1⟩)/√2
               H|1⟩ = |-⟩ = (|0⟩-|1⟩)/√2
       Creates equal superposition from basis states
       Important: H² = I (self-inverse)
    
    6. PHASE (S) Gate:
       Matrix: S = [[1, 0],
                    [0, i]]
       Effect: S|0⟩ = |0⟩, S|1⟩ = i|1⟩
       Adds π/2 (90°) phase to |1⟩ component
       Note: S² = Z
    
    7. T Gate (π/8 gate):
       Matrix: T = [[1, 0],
                    [0, e^(iπ/4)]]
       Effect: T|0⟩ = |0⟩, T|1⟩ = e^(iπ/4)|1⟩
       Adds π/4 (45°) phase to |1⟩ component
       Note: T² = S, T⁴ = Z
    
    Why These Gates?
    ----------------
    - X, Y, Z are the Pauli matrices (fundamental in quantum mechanics)
    - H creates superposition (essential for quantum algorithms)
    - S, T are phase gates (important for quantum circuits)
    - Together they form a universal gate set (can approximate any single-qubit gate)
    
    Returns:
        dict: Dictionary of quantum circuits with different gates applied
    """
    print("=== SINGLE QUBIT GATES ===")
    print()

    # Define the gates to demonstrate
    # Each lambda function applies the corresponding gate to qubit 0
    gates = {
        "Identity (I)": lambda qc: None,  # Do nothing - identity operation
        "Pauli-X (NOT)": lambda qc: qc.x(0),  # Bit flip
        "Pauli-Y": lambda qc: qc.y(0),  # Bit + phase flip
        "Pauli-Z": lambda qc: qc.z(0),  # Phase flip only
        "Hadamard (H)": lambda qc: qc.h(0),  # Superposition creator
        "Phase (S)": lambda qc: qc.s(0),  # π/2 phase gate
        "T Gate": lambda qc: qc.t(0),  # π/4 phase gate
    }

    gate_descriptions = {
        "Identity (I)": "Does nothing - leaves qubit unchanged",
        "Pauli-X (NOT)": "Flips qubit: |0⟩ ↔ |1⟩ (quantum NOT gate)",
        "Pauli-Y": "Rotation around Y-axis (flips + phase)",
        "Pauli-Z": "Phase flip: |1⟩ → -|1⟩, |0⟩ unchanged",
        "Hadamard (H)": "Creates superposition: |0⟩ → (|0⟩+|1⟩)/√2",
        "Phase (S)": "Adds π/2 phase: |1⟩ → i|1⟩",
        "T Gate": "Adds π/4 phase: |1⟩ → e^(iπ/4)|1⟩",
    }

    circuits = {}

    for gate_name, gate_function in gates.items():
        # Start with |0⟩ state (default initial state)
        # |0⟩ = [1, 0]ᵀ in vector form
        qc = QuantumCircuit(1)
        if gate_function:
            gate_function(qc)
        circuits[gate_name] = qc

        print(f"{gate_name}:")
        print(f"  Description: {gate_descriptions[gate_name]}")
        print(f"  Circuit: {qc.data}")
        print()

    return circuits


def demonstrate_hadamard_sequence():
    """
    Demonstrate a sequence of Hadamard gates.
    
    Mathematical Foundation - Hadamard Gate Properties:
    ---------------------------------------------------
    The Hadamard gate has special mathematical properties:
    
    H = (1/√2)[[1,  1],
               [1, -1]]
    
    Key Property - Self-Inverse:
    ---------------------------
    H² = H × H = I (Identity)
    
    Mathematical Proof:
    H² = (1/√2)[[1,  1],    × (1/√2)[[1,  1],
                [1, -1]]              [1, -1]]
    
       = (1/2)[[1+1,   1-1],
               [1-1,   1+1]]
    
       = (1/2)[[2, 0],
               [0, 2]]
    
       = [[1, 0],
          [0, 1]] = I
    
    This means applying H twice returns to the original state!
    
    Sequence Effects:
    ----------------
    Starting from |0⟩:
    - 0 H gates: |0⟩ = [1, 0]ᵀ
    - 1 H gate:  |+⟩ = (|0⟩ + |1⟩)/√2 = [1/√2, 1/√2]ᵀ
    - 2 H gates: |0⟩ = [1, 0]ᵀ (back to start!)
    - 3 H gates: |+⟩ = [1/√2, 1/√2]ᵀ (same as 1 H gate)
    - 4 H gates: |0⟩ = [1, 0]ᵀ (back to start!)
    
    Pattern: H^n alternates between |0⟩ (even n) and |+⟩ (odd n)
    
    Physical Interpretation:
    ------------------------
    On the Bloch sphere, each H gate is a 180° rotation around the
    axis halfway between X and Z (the [1,0,1] direction).
    Two such rotations complete a full 360° cycle, returning to start.
    
    Returns:
        dict: Dictionary of circuits with different numbers of H gates
    """
    print("=== HADAMARD GATE SEQUENCE ===")
    print()

    circuits = {}

    # Apply multiple Hadamard gates to demonstrate periodicity
    # We'll see that H² = I (Hadamard is self-inverse)
    for i in range(4):
        qc = QuantumCircuit(1)
        # Apply H gate i times
        for _ in range(i):
            qc.h(0)
        circuits[f"{i} H gates"] = qc

        # Get the resulting quantum state
        state = Statevector.from_instruction(qc)
        print(f"After {i} Hadamard gate(s):")
        print(f"  State: {state}")
        # Calculate measurement probabilities using Born rule: P = |amplitude|²
        print(
            f"  Probabilities: |0⟩: {abs(state[0])**2:.3f}, |1⟩: {abs(state[1])**2:.3f}"
        )
        print()

    print("Notice: Two H gates return to original state (H² = I)")
    print("Pattern: Even number of H gates → |0⟩, Odd number → |+⟩")
    print()

    return circuits


def demonstrate_multi_qubit_gates():
    """
    Demonstrate multi-qubit gates.
    
    Mathematical Foundation - Multi-Qubit Gates:
    --------------------------------------------
    Multi-qubit gates act on systems of 2 or more qubits.
    For n qubits, the state space has dimension 2^n.
    
    State Vector for n qubits:
    |ψ⟩ = Σ α_i |i⟩ where i ranges over all 2^n basis states
    and Σ|α_i|² = 1 (normalization)
    
    1. CNOT (Controlled-X / CX) Gate:
    ----------------------------------
    2-qubit gate with one control and one target qubit.
    
    Operation:
    - If control = |0⟩: target unchanged
    - If control = |1⟩: target flipped (X gate applied)
    
    Matrix representation (4×4 for 2 qubits):
    CNOT = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]]
    
    Basis state action:
    |00⟩ → |00⟩  (control=0, target unchanged)
    |01⟩ → |01⟩  (control=0, target unchanged)
    |10⟩ → |11⟩  (control=1, target flipped)
    |11⟩ → |10⟩  (control=1, target flipped)
    
    Creating Entanglement:
    When control is in superposition, CNOT creates entanglement!
    Example: CNOT(H|0⟩ ⊗ |0⟩) = CNOT((|0⟩+|1⟩)/√2 ⊗ |0⟩)
                                = (|00⟩ + |11⟩)/√2  (Bell state!)
    
    2. Controlled-Z (CZ) Gate:
    ---------------------------
    Applies Z gate to target when control is |1⟩.
    
    Matrix representation:
    CZ = [[1, 0, 0,  0],
          [0, 1, 0,  0],
          [0, 0, 1,  0],
          [0, 0, 0, -1]]
    
    Basis state action:
    |00⟩ → |00⟩
    |01⟩ → |01⟩
    |10⟩ → |10⟩
    |11⟩ → -|11⟩  (only |11⟩ gets phase flip)
    
    Symmetry: CZ is symmetric - control and target are interchangeable!
    CZ(i,j) = CZ(j,i)
    
    3. Toffoli (CCX/CCNOT) Gate:
    -----------------------------
    3-qubit gate: two controls, one target.
    Applies X to target only when BOTH controls are |1⟩.
    
    Classical analog: AND gate followed by XOR
    
    Basis state action:
    |110⟩ → |111⟩  (both controls=1, target flipped)
    |111⟩ → |110⟩  (both controls=1, target flipped)
    All other states unchanged
    
    Universal Classical Computation:
    Toffoli + NOT gates are universal for classical computation!
    Can build any classical circuit using just these gates.
    
    State Space Dimensions:
    -----------------------
    - 1 qubit: 2 dimensions (2¹ = 2)
    - 2 qubits: 4 dimensions (2² = 4)
    - 3 qubits: 8 dimensions (2³ = 8)
    - n qubits: 2^n dimensions (exponential growth!)
    
    This exponential scaling is why quantum computers are powerful,
    but also why they're hard to simulate on classical computers.
    
    Returns:
        dict: Dictionary of circuits demonstrating multi-qubit gates
    """
    print("=== MULTI-QUBIT GATES ===")
    print()

    circuits = {}

    # CNOT gate (Controlled-X) - creates entanglement
    # Starting from |00⟩, we apply H to control, then CNOT
    # This creates a Bell state: (|00⟩ + |11⟩)/√2
    qc_cnot = QuantumCircuit(2)
    qc_cnot.h(0)  # Put control qubit in superposition: (|0⟩+|1⟩)/√2
    qc_cnot.cx(0, 1)  # Apply CNOT: entangles control and target
    circuits["CNOT Gate"] = qc_cnot

    # Controlled-Z gate - symmetric phase gate
    # Both qubits in superposition, then CZ adds phase to |11⟩ component
    qc_cz = QuantumCircuit(2)
    qc_cz.h(0)  # Put control qubit in superposition
    qc_cz.h(1)  # Put target qubit in superposition
    qc_cz.cz(0, 1)  # Apply CZ: adds -1 phase to |11⟩ component
    circuits["CZ Gate"] = qc_cz

    # Toffoli gate (CCX - Controlled-Controlled-X) - 3-qubit gate
    # Requires BOTH controls to be |1⟩ to flip target
    qc_ccx = QuantumCircuit(3)
    qc_ccx.h(0)  # Put first control in superposition
    qc_ccx.h(1)  # Put second control in superposition
    qc_ccx.ccx(0, 1, 2)  # Apply Toffoli: flips target only when both controls are |1⟩
    circuits["Toffoli (CCX)"] = qc_ccx

    for name, circuit in circuits.items():
        print(f"{name}:")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Gates: {len(circuit.data)}")
        state = Statevector.from_instruction(circuit)
        # State vector dimension = 2^n where n is number of qubits
        print(f"  Final state dimension: {len(state)} = 2^{circuit.num_qubits}")
        print()

    return circuits


def visualize_gate_effects(single_qubit_circuits):
    """Visualize the effects of single-qubit gates."""
    print("=== GATE EFFECTS VISUALIZATION ===")
    print()

    # Create individual Bloch sphere plots for each gate
    for i, (gate_name, circuit) in enumerate(single_qubit_circuits.items()):
        state = Statevector.from_instruction(circuit)

        print(f"{gate_name}:")
        print(f"  State vector: {state}")
        print(
            f"  Probabilities: |0⟩: {abs(state[0])**2:.3f}, |1⟩: {abs(state[1])**2:.3f}"
        )

        # Create individual Bloch sphere plots
        try:
            bloch_fig = plot_bloch_multivector(
                state, title=f"{gate_name} - Qubit State"
            )
            plt.savefig(f"module1_02_bloch_{i+1}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create Bloch sphere for {gate_name}: {e}")
            # Provide alternative visualization information
            print(
                f"  Alternative: State components - α=({state[0].real:.3f}+{state[0].imag:.3f}i), β=({state[1].real:.3f}+{state[1].imag:.3f}i)"
            )

        print()

    # Create a summary visualization with state information
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        gate_names = list(single_qubit_circuits.keys())
        prob_0 = []
        prob_1 = []

        for gate_name, circuit in single_qubit_circuits.items():
            state = Statevector.from_instruction(circuit)
            prob_0.append(abs(state[0]) ** 2)
            prob_1.append(abs(state[1]) ** 2)

        x = range(len(gate_names))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            prob_0,
            width,
            label="|0⟩ probability",
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            prob_1,
            width,
            label="|1⟩ probability",
            alpha=0.8,
        )

        ax.set_xlabel("Quantum Gates")
        ax.set_ylabel("Probability")
        ax.set_title("Gate Effects: Measurement Probabilities")
        ax.set_xticks(x)
        ax.set_xticklabels(gate_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("module1_02_gate_effects.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"⚠️ Could not create gate effects summary: {e}")


def create_quantum_circuit_examples():
    """Create example quantum circuits of increasing complexity."""
    print("=== QUANTUM CIRCUIT EXAMPLES ===")
    print()

    circuits = {}

    # Example 1: Simple circuit
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.z(0)
    qc1.h(0)
    circuits["Circuit 1: H-Z-H"] = qc1

    # Example 2: Multi-step circuit
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.h(0)
    qc2.h(1)
    circuits["Circuit 2: Bell + H"] = qc2

    # Example 3: Complex circuit
    qc3 = QuantumCircuit(3)
    qc3.h(0)
    qc3.cx(0, 1)
    qc3.cx(1, 2)
    qc3.h(2)
    qc3.cx(1, 2)
    qc3.cx(0, 1)
    qc3.h(0)
    circuits["Circuit 3: GHZ preparation"] = qc3

    # Display circuit information and create diagrams
    for i, (name, circuit) in enumerate(circuits.items()):
        print(f"{name}:")
        print(f"  Depth: {circuit.depth()}")
        print(f"  Gates: {circuit.count_ops()}")

        # Draw circuit - create individual figures to avoid ax parameter issues
        try:
            fig = circuit.draw(output="mpl", style={"backgroundcolor": "#EEEEEE"})
            fig.suptitle(f"{name} (Depth: {circuit.depth()})", fontsize=12)
            # Save individual circuit diagrams
            plt.figure(fig.number)
            plt.savefig(f"module1_02_circuit_{i+1}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not create circuit diagram: {e}")
            print(f"  Circuit structure: {circuit.data}")

        print()

    # Create combined figure with all circuits
    try:
        fig, axes = plt.subplots(len(circuits), 1, figsize=(12, 3 * len(circuits)))
        if len(circuits) == 1:
            axes = [axes]

        for i, (name, circuit) in enumerate(circuits.items()):
            # Use text representation instead of circuit_drawer with ax parameter
            axes[i].text(
                0.5,
                0.5,
                f"{name}\nDepth: {circuit.depth()}\nGates: {circuit.count_ops()}",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"{name} (Depth: {circuit.depth()})", fontsize=12, pad=20)

        plt.tight_layout()
        plt.savefig("module1_02_circuit_examples.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"⚠️ Could not create combined circuit diagram: {e}")

    return circuits


def demonstrate_gate_matrices():
    """
    Show the mathematical representation of quantum gates.
    
    Mathematical Foundation - Matrix Representation:
    ------------------------------------------------
    Quantum gates are represented as unitary matrices that transform
    state vectors through matrix multiplication.
    
    State Transformation:
    |ψ'⟩ = U|ψ⟩
    
    In matrix form, if |ψ⟩ = [α, β]ᵀ:
    [α']   [u₀₀ u₀₁] [α]
    [β'] = [u₁₀ u₁₁] [β]
    
    Unitarity Requirements:
    -----------------------
    A matrix U is unitary if U†U = I, where U† = (U*)ᵀ
    (conjugate transpose)
    
    This ensures:
    1. |det(U)| = 1 (determinant has unit magnitude)
    2. U preserves inner products: ⟨ψ|ψ⟩ = ⟨ψ'|ψ'⟩
    3. U is reversible: U† is also unitary and U†U = UU† = I
    
    Why Matrix Determinant Matters:
    -------------------------------
    For quantum gates, det(U) = e^(iφ) for some phase φ
    Common cases:
    - det(U) = 1: Special unitary (SU(2) group)
    - det(U) = -1: Includes global phase
    - |det(U)| = 1 always (unitarity requirement)
    
    Checking Unitarity:
    -------------------
    We verify U†U = I by computing:
    U† @ U = (U.conj().T) @ U
    
    If result equals identity matrix [[1,0],[0,1]], gate is unitary.
    
    Note on Complex Numbers:
    ------------------------
    - i = √(-1) is the imaginary unit
    - e^(iθ) = cos(θ) + i·sin(θ) (Euler's formula)
    - |e^(iθ)| = 1 (unit magnitude)
    - e^(iπ/4) = cos(π/4) + i·sin(π/4) = (1+i)/√2
    """
    print("=== GATE MATRICES ===")
    print()

    # Define gate matrices as numpy arrays
    # Each is a 2×2 complex matrix representing a unitary transformation
    
    # Identity - does nothing
    I = np.array([[1, 0], [0, 1]])
    
    # Pauli-X - bit flip (quantum NOT)
    X = np.array([[0, 1], [1, 0]])
    
    # Pauli-Y - bit flip with phase (combines X and Z)
    Y = np.array([[0, -1j], [1j, 0]])
    
    # Pauli-Z - phase flip
    Z = np.array([[1, 0], [0, -1]])
    
    # Hadamard - creates superposition (normalized rotation)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    
    # S gate - adds π/2 phase (√Z gate, since S² = Z)
    S = np.array([[1, 0], [0, 1j]])
    
    # T gate - adds π/4 phase (⁴√Z gate, since T⁴ = Z)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

    gates_matrices = {
        "Identity (I)": I,
        "Pauli-X": X,
        "Pauli-Y": Y,
        "Pauli-Z": Z,
        "Hadamard (H)": H,
        "Phase (S)": S,
        "T Gate": T,
    }

    for gate_name, matrix in gates_matrices.items():
        print(f"{gate_name}:")
        print(f"  Matrix:\n{matrix}")
        # Determinant should have magnitude 1 for unitary matrices
        print(f"  Determinant: {np.linalg.det(matrix):.3f}")
        # Check unitarity: U†U should equal identity matrix
        # matrix.conj().T is the conjugate transpose (Hermitian adjoint)
        print(f"  Unitary: {np.allclose(matrix @ matrix.conj().T, np.eye(2))}")
        print()

    print("Note: All quantum gates are unitary (reversible)")
    print("This means:")
    print("  • Information is preserved (no information loss)")
    print("  • Every gate has an inverse (U† = U⁻¹)")
    print("  • Probabilities are conserved (|det(U)| = 1)")
    print()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Quantum Gates and Circuits Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--show-matrices", action="store_true", help="Show gate matrix representations"
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 1, Example 2")
    print("Quantum Gates and Circuits")
    print("=" * 50)
    print()

    try:
        # Demonstrate single qubit gates
        single_qubit_circuits = demonstrate_single_qubit_gates()

        # Demonstrate Hadamard sequence
        hadamard_circuits = demonstrate_hadamard_sequence()

        # Demonstrate multi-qubit gates
        multi_qubit_circuits = demonstrate_multi_qubit_gates()

        # Visualize gate effects
        visualize_gate_effects(single_qubit_circuits)

        # Create circuit examples
        example_circuits = create_quantum_circuit_examples()

        # Show gate matrices if requested
        if args.show_matrices:
            demonstrate_gate_matrices()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module1_02_gate_effects.png - Gate effects on Bloch sphere")
        print("• module1_02_circuit_examples.png - Example quantum circuits")
        print()
        print("🎯 Key takeaways:")
        print("• Quantum gates are the building blocks of quantum circuits")
        print("• All quantum gates are reversible (unitary)")
        print("• Gates can create superposition and entanglement")
        print("• Circuit depth affects computational complexity")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
