#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 2, Example 2
Linear Algebra for Quantum Computing

This example demonstrates essential linear algebra concepts used in quantum computing,
including vector operations, matrix multiplication, and eigenvalue problems.

Learning objectives:
- Master vector operations in quantum state space
- Understand matrix representations of quantum gates
- Compute eigenvalues and eigenvectors of quantum operators
- Apply linear algebra to quantum state transformations

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from scipy.linalg import eig
import seaborn as sns


def demonstrate_vector_operations():
    """Demonstrate basic vector operations in quantum state space."""
    print("=== VECTOR OPERATIONS IN QUANTUM SPACE ===")
    print()

    # Define quantum state vectors
    state_0 = np.array([1, 0])  # |0⟩
    state_1 = np.array([0, 1])  # |1⟩
    state_plus = np.array([1, 1]) / np.sqrt(2)  # |+⟩
    state_minus = np.array([1, -1]) / np.sqrt(2)  # |-⟩

    states = {"|0⟩": state_0, "|1⟩": state_1, "|+⟩": state_plus, "|-⟩": state_minus}

    print("Basic quantum state vectors:")
    for label, state in states.items():
        norm = np.linalg.norm(state)
        print(f"{label}: {state} (norm: {norm:.3f})")
    print()

    # Vector addition and subtraction
    print("Vector operations:")
    sum_vector = state_0 + state_1
    diff_vector = state_0 - state_1

    print(f"|0⟩ + |1⟩ = {sum_vector}")
    print(f"|0⟩ - |1⟩ = {diff_vector}")
    print(f"Normalized (|0⟩ + |1⟩)/√2 = {sum_vector / np.linalg.norm(sum_vector)}")
    print(f"Normalized (|0⟩ - |1⟩)/√2 = {diff_vector / np.linalg.norm(diff_vector)}")
    print()

    # Dot products (inner products)
    print("Inner products ⟨ψ|φ⟩:")
    for label1, state1 in states.items():
        for label2, state2 in states.items():
            inner_product = np.vdot(state1, state2)
            print(f"⟨{label1}|{label2}⟩ = {inner_product:.3f}")
        print()

    return states


def demonstrate_matrix_operations():
    """Demonstrate matrix operations for quantum gates."""
    print("=== MATRIX OPERATIONS FOR QUANTUM GATES ===")
    print()

    # Define Pauli matrices and other important gates
    I = np.array([[1, 0], [0, 1]])  # Identity
    X = np.array([[0, 1], [1, 0]])  # Pauli-X
    Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y
    Z = np.array([[1, 0], [0, -1]])  # Pauli-Z
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard

    gates = {"I": I, "X": X, "Y": Y, "Z": Z, "H": H}

    print("Quantum gate matrices:")
    for name, matrix in gates.items():
        print(f"{name} gate:")
        print(matrix)
        print(f"Determinant: {np.linalg.det(matrix):.3f}")
        print(f"Trace: {np.trace(matrix):.3f}")
        print()

    # Matrix multiplication examples
    print("Matrix multiplication examples:")

    # HXH = Z (important identity)
    HXH = H @ X @ H
    print("H·X·H =")
    print(HXH)
    print(f"This equals Z: {np.allclose(HXH, Z)}")
    print()

    # Pauli matrices anticommutation
    print("Pauli matrices anticommutation {X,Y} = XY + YX:")
    anticommutator_XY = X @ Y + Y @ X
    print(f"XY + YX = \n{anticommutator_XY}")
    print(f"Equals zero matrix: {np.allclose(anticommutator_XY, np.zeros((2,2)))}")
    print()

    return gates


def demonstrate_eigenvalue_problems():
    """Demonstrate eigenvalue problems for quantum operators."""
    print("=== EIGENVALUE PROBLEMS ===")
    print()

    # Define matrices
    matrices = {
        "Pauli-X": np.array([[0, 1], [1, 0]]),
        "Pauli-Y": np.array([[0, -1j], [1j, 0]]),
        "Pauli-Z": np.array([[1, 0], [0, -1]]),
        "Hadamard": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    }

    eigendata = {}

    for name, matrix in matrices.items():
        eigenvalues, eigenvectors = eig(matrix)
        eigendata[name] = (eigenvalues, eigenvectors)

        print(f"{name} matrix eigenanalysis:")
        print(f"Matrix:\n{matrix}")
        print(f"Eigenvalues: {eigenvalues}")
        print("Eigenvectors:")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"  λ{i+1} = {val:.3f}, |v{i+1}⟩ = {vec}")
        print()

        # Verify eigenvalue equation: A|v⟩ = λ|v⟩
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            left_side = matrix @ vec
            right_side = val * vec
            print(
                f"Verification A|v{i+1}⟩ = λ{i+1}|v{i+1}⟩: {np.allclose(left_side, right_side)}"
            )
        print()

    return eigendata


def visualize_matrix_operations():
    """Visualize matrix operations and their effects."""
    print("=== MATRIX OPERATION VISUALIZATION ===")
    print()

    # Create a figure with subplots for different visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Define gates
    gates = {
        "I": np.array([[1, 0], [0, 1]]),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
        "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    }

    # Plot 1: Gate matrices as heatmaps
    ax = axes[0, 0]
    gate_names = list(gates.keys())
    gate_matrices = np.array([np.real(gates[name]) for name in gate_names])

    im = ax.imshow(gate_matrices.reshape(-1, 2), cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title("Real Parts of Gate Matrices")
    ax.set_yticks(range(0, len(gate_names) * 2, 2))
    ax.set_yticklabels(gate_names)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Col 0", "Col 1"])
    plt.colorbar(im, ax=ax)

    # Plot 2: Eigenvalues
    ax = axes[0, 1]
    for name, matrix in gates.items():
        eigenvals = np.linalg.eigvals(matrix)
        ax.scatter(np.real(eigenvals), np.imag(eigenvals), label=name, s=100, alpha=0.7)

    # Add unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Eigenvalues in Complex Plane")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Plot 3: State transformation examples
    ax = axes[0, 2]
    initial_state = np.array([1, 0])  # |0⟩

    transformed_states = {}
    for name, gate in gates.items():
        transformed = gate @ initial_state
        transformed_states[name] = transformed

        # Plot as arrows
        ax.arrow(
            0,
            0,
            np.real(transformed[0]),
            np.imag(transformed[0]),
            head_width=0.05,
            head_length=0.05,
            fc="red",
            alpha=0.7,
            length_includes_head=True,
        )
        ax.arrow(
            0,
            0,
            np.real(transformed[1]),
            np.imag(transformed[1]),
            head_width=0.05,
            head_length=0.05,
            fc="blue",
            alpha=0.7,
            length_includes_head=True,
        )

        ax.text(
            np.real(transformed[0]),
            np.imag(transformed[0]),
            f"{name}₀",
            fontsize=10,
            ha="center",
        )
        ax.text(
            np.real(transformed[1]),
            np.imag(transformed[1]),
            f"{name}₁",
            fontsize=10,
            ha="center",
        )

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("State Transformations of |0⟩")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Plot 4: Matrix commutation relationships
    ax = axes[1, 0]
    pauli_matrices = ["X", "Y", "Z"]
    commutation_matrix = np.zeros((3, 3))

    for i, gate1 in enumerate(pauli_matrices):
        for j, gate2 in enumerate(pauli_matrices):
            mat1 = gates[gate1]
            mat2 = gates[gate2]
            commutator = mat1 @ mat2 - mat2 @ mat1
            commutation_matrix[i, j] = np.linalg.norm(commutator)

    im = ax.imshow(commutation_matrix, cmap="Reds")
    ax.set_title("Pauli Matrix Commutation\n||[A,B]|| = ||AB - BA||")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(pauli_matrices)
    ax.set_yticklabels(pauli_matrices)
    plt.colorbar(im, ax=ax)

    # Add values to cells
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{commutation_matrix[i,j]:.1f}",
                ha="center",
                va="center",
                color="white" if commutation_matrix[i, j] > 1 else "black",
            )

    # Plot 5: Unitary property verification
    ax = axes[1, 1]
    gate_names = list(gates.keys())
    unitarity_errors = []

    for name, gate in gates.items():
        # Check if U†U = I
        unitary_product = gate.conj().T @ gate
        identity = np.eye(2)
        error = np.linalg.norm(unitary_product - identity)
        unitarity_errors.append(error)

    bars = ax.bar(gate_names, unitarity_errors, alpha=0.7, color="green")
    ax.set_ylabel("||U†U - I||")
    ax.set_title("Unitarity Verification")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add error values on bars
    for bar, error in zip(bars, unitarity_errors):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{error:.2e}",
            ha="center",
            va="bottom",
            rotation=90,
        )

    # Plot 6: Trace and determinant properties
    ax = axes[1, 2]
    traces = [np.trace(gates[name]) for name in gate_names]
    determinants = [np.linalg.det(gates[name]) for name in gate_names]

    x = np.arange(len(gate_names))
    width = 0.35

    ax.bar(x - width / 2, np.real(traces), width, label="Trace (real)", alpha=0.7)
    ax.bar(
        x + width / 2,
        np.real(determinants),
        width,
        label="Determinant (real)",
        alpha=0.7,
    )

    ax.set_xlabel("Gate")
    ax.set_ylabel("Value")
    ax.set_title("Trace and Determinant")
    ax.set_xticks(x)
    ax.set_xticklabels(gate_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("module2_02_matrix_operations.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_quantum_state_evolution():
    """Demonstrate how matrices evolve quantum states."""
    print("=== QUANTUM STATE EVOLUTION ===")
    print()

    # Start with |0⟩ state
    initial_state = np.array([1, 0])
    print(f"Initial state: |ψ₀⟩ = {initial_state}")
    print()

    # Define a sequence of operations
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    operations = [
        ("H", H, "Apply Hadamard: create superposition"),
        ("Z", Z, "Apply Z: add phase to |1⟩ component"),
        ("H", H, "Apply Hadamard: rotate back"),
        ("X", X, "Apply X: flip the qubit"),
    ]

    current_state = initial_state.copy()
    states_evolution = [current_state.copy()]

    print("State evolution through gate sequence:")
    for i, (gate_name, gate_matrix, description) in enumerate(operations):
        current_state = gate_matrix @ current_state
        states_evolution.append(current_state.copy())

        prob_0 = abs(current_state[0]) ** 2
        prob_1 = abs(current_state[1]) ** 2

        print(f"Step {i+1}: {description}")
        print(f"  Gate: {gate_name}")
        print(f"  State: {current_state}")
        print(f"  Probabilities: P(|0⟩) = {prob_0:.3f}, P(|1⟩) = {prob_1:.3f}")
        print()

    # Visualize evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot state amplitudes over time
    steps = range(len(states_evolution))
    amplitudes_0 = [state[0] for state in states_evolution]
    amplitudes_1 = [state[1] for state in states_evolution]

    ax1.plot(
        steps, np.real(amplitudes_0), "ro-", label="Real(α₀)", linewidth=2, markersize=8
    )
    ax1.plot(steps, np.imag(amplitudes_0), "r--", label="Imag(α₀)", linewidth=2)
    ax1.plot(
        steps, np.real(amplitudes_1), "bo-", label="Real(α₁)", linewidth=2, markersize=8
    )
    ax1.plot(steps, np.imag(amplitudes_1), "b--", label="Imag(α₁)", linewidth=2)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("State Amplitude Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Set x-axis labels
    step_labels = ["Initial"] + [f"{op[0]}" for op in operations]
    ax1.set_xticks(steps)
    ax1.set_xticklabels(step_labels, rotation=45)

    # Plot probabilities over time
    probabilities_0 = [abs(state[0]) ** 2 for state in states_evolution]
    probabilities_1 = [abs(state[1]) ** 2 for state in states_evolution]

    ax2.plot(steps, probabilities_0, "ro-", label="P(|0⟩)", linewidth=2, markersize=8)
    ax2.plot(steps, probabilities_1, "bo-", label="P(|1⟩)", linewidth=2, markersize=8)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Probability")
    ax2.set_title("Measurement Probability Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    ax2.set_xticks(steps)
    ax2.set_xticklabels(step_labels, rotation=45)

    plt.tight_layout()
    plt.savefig("module2_02_state_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()

    return states_evolution


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(
        description="Linear Algebra for Quantum Computing Demo"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--show-proofs",
        action="store_true",
        help="Show mathematical proofs and derivations",
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 2, Example 2")
    print("Linear Algebra for Quantum Computing")
    print("=" * 50)
    print()

    try:
        # Vector operations
        quantum_states = demonstrate_vector_operations()

        # Matrix operations
        quantum_gates = demonstrate_matrix_operations()

        # Eigenvalue problems
        eigendata = demonstrate_eigenvalue_problems()

        # Visualizations
        visualize_matrix_operations()

        # State evolution
        evolution = demonstrate_quantum_state_evolution()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module2_02_matrix_operations.png - Matrix operation analysis")
        print("• module2_02_state_evolution.png - Quantum state evolution")
        print()
        print("🎯 Key takeaways:")
        print("• Quantum states are vectors in complex vector space")
        print("• Quantum gates are unitary matrices preserving norm")
        print("• Eigenvalues/eigenvectors reveal gate properties")
        print("• Matrix multiplication represents sequential operations")
        print("• Linear algebra is the mathematical foundation of quantum computing")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy scipy seaborn")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
