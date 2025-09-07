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
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_multivector, circuit_drawer
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def demonstrate_single_qubit_gates():
    """Demonstrate the effect of single-qubit gates."""
    print("=== SINGLE QUBIT GATES ===")
    print()

    # Define the gates to demonstrate
    gates = {
        "Identity (I)": lambda qc: None,  # Do nothing
        "Pauli-X (NOT)": lambda qc: qc.x(0),
        "Pauli-Y": lambda qc: qc.y(0),
        "Pauli-Z": lambda qc: qc.z(0),
        "Hadamard (H)": lambda qc: qc.h(0),
        "Phase (S)": lambda qc: qc.s(0),
        "T Gate": lambda qc: qc.t(0),
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
        # Start with |0⟩ state
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
    """Demonstrate a sequence of Hadamard gates."""
    print("=== HADAMARD GATE SEQUENCE ===")
    print()

    circuits = {}

    # Apply multiple Hadamard gates
    for i in range(4):
        qc = QuantumCircuit(1)
        for _ in range(i):
            qc.h(0)
        circuits[f"{i} H gates"] = qc

        state = Statevector.from_instruction(qc)
        print(f"After {i} Hadamard gate(s):")
        print(f"  State: {state}")
        print(
            f"  Probabilities: |0⟩: {abs(state[0])**2:.3f}, |1⟩: {abs(state[1])**2:.3f}"
        )
        print()

    print("Notice: Two H gates return to original state (H² = I)")
    print()

    return circuits


def demonstrate_multi_qubit_gates():
    """Demonstrate multi-qubit gates."""
    print("=== MULTI-QUBIT GATES ===")
    print()

    circuits = {}

    # CNOT gate (Controlled-X)
    qc_cnot = QuantumCircuit(2)
    qc_cnot.h(0)  # Put control qubit in superposition
    qc_cnot.cx(0, 1)  # Apply CNOT
    circuits["CNOT Gate"] = qc_cnot

    # Controlled-Z gate
    qc_cz = QuantumCircuit(2)
    qc_cz.h(0)  # Put control qubit in superposition
    qc_cz.h(1)  # Put target qubit in superposition
    qc_cz.cz(0, 1)  # Apply CZ
    circuits["CZ Gate"] = qc_cz

    # Toffoli gate (CCX - Controlled-Controlled-X)
    qc_ccx = QuantumCircuit(3)
    qc_ccx.h(0)  # Put first control in superposition
    qc_ccx.h(1)  # Put second control in superposition
    qc_ccx.ccx(0, 1, 2)  # Apply Toffoli
    circuits["Toffoli (CCX)"] = qc_ccx

    for name, circuit in circuits.items():
        print(f"{name}:")
        print(f"  Qubits: {circuit.num_qubits}")
        print(f"  Gates: {len(circuit.data)}")
        state = Statevector.from_instruction(circuit)
        print(f"  Final state dimension: {len(state)}")
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
            plt.show()
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
        plt.show()

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
            plt.show()
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
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not create combined circuit diagram: {e}")

    return circuits


def demonstrate_gate_matrices():
    """Show the mathematical representation of quantum gates."""
    print("=== GATE MATRICES ===")
    print()

    # Define gate matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
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
        print(f"  Determinant: {np.linalg.det(matrix):.3f}")
        print(f"  Unitary: {np.allclose(matrix @ matrix.conj().T, np.eye(2))}")
        print()

    print("Note: All quantum gates are unitary (reversible)")
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
