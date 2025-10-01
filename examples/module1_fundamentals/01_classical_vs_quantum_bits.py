#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 1
Classical vs Quantum Bits

This example demonstrates the fundamental differences between classical bits
and quantum bits (qubits), including visualization of quantum states.

Learning objectives:
- Understand classical vs quantum information storage
- Visualize qubit states on the Bloch sphere
- Explore the concept of quantum superposition

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def demonstrate_classical_bits():
    """Demonstrate classical bit behavior and limitations."""
    print("=== CLASSICAL BITS ===")
    print()

    # Classical bit can only be 0 or 1
    classical_bit = 0
    print(f"Classical bit value: {classical_bit}")
    print("Possible states: 0 or 1")
    print("Properties:")
    print("- Deterministic: always gives the same value when read")
    print("- Binary: can only be in one of two states")
    print("- Independent: multiple bits don't influence each other")
    print()

    # Multiple classical bits
    classical_byte = [0, 1, 1, 0, 1, 0, 0, 1]
    print(f"Classical byte: {classical_byte}")
    print(
        f"As decimal: {sum(bit * 2**i for i, bit in enumerate(reversed(classical_byte)))}"
    )
    print()

    return classical_byte


def demonstrate_quantum_bits():
    """Demonstrate quantum bit (qubit) behavior and capabilities."""
    print("=== QUANTUM BITS (QUBITS) ===")
    print()

    # Create quantum circuits for different qubit states
    circuits = {}

    # |0⟩ state (classical-like)
    qc_0 = QuantumCircuit(1)
    circuits["|0⟩"] = qc_0

    # |1⟩ state (classical-like)
    qc_1 = QuantumCircuit(1)
    qc_1.x(0)  # Apply X gate to flip |0⟩ to |1⟩
    circuits["|1⟩"] = qc_1

    # |+⟩ state (superposition)
    qc_plus = QuantumCircuit(1)
    qc_plus.h(0)  # Apply Hadamard gate to create superposition
    circuits["|+⟩ = (|0⟩ + |1⟩)/√2"] = qc_plus

    # |-⟩ state (superposition)
    qc_minus = QuantumCircuit(1)
    qc_minus.x(0)
    qc_minus.h(0)
    circuits["|-⟩ = (|0⟩ - |1⟩)/√2"] = qc_minus

    # |i⟩ state (complex superposition)
    qc_i = QuantumCircuit(1)
    qc_i.h(0)
    qc_i.s(0)  # Apply S gate to add phase
    circuits["|i⟩ = (|0⟩ + i|1⟩)/√2"] = qc_i

    return circuits


def visualize_qubit_states(circuits, verbose=False):
    """Visualize qubit states on the Bloch sphere."""
    print("=== QUBIT STATE VISUALIZATION ===")
    print()

    states = {}
    bloch_figures = []

    for i, (label, circuit) in enumerate(circuits.items()):
        # Get the statevector
        state = Statevector.from_instruction(circuit)
        states[label] = state

        if verbose:
            print(f"State {label}:")
            print(f"  Statevector: {state}")
            print(
                f"  Probabilities: |0⟩: {abs(state[0])**2:.3f}, |1⟩: {abs(state[1])**2:.3f}"
            )
            print()

        # Plot individual Bloch sphere (Qiskit 2.x doesn't support ax parameter)
        try:
            bloch_fig = plot_bloch_multivector(state, title=f"Qubit State: {label}")
            bloch_figures.append(bloch_fig)

            # Save individual Bloch sphere
            filename = f"module1_01_qubit_state_{i:02d}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"💾 Saved: {filename}")

        except Exception as e:
            print(f"⚠️ Could not create Bloch sphere for {label}: {e}")

    if bloch_figures:
        plt.close()

    return states


def measure_qubits(circuits, shots=1000):
    """Demonstrate measurement of different qubit states."""
    print("=== MEASUREMENT RESULTS ===")
    print()

    simulator = AerSimulator()

    # Create figure for measurement histograms
    fig, axes = plt.subplots(1, len(circuits), figsize=(4 * len(circuits), 3))
    if len(circuits) == 1:
        axes = [axes]

    results = {}

    for i, (label, circuit) in enumerate(circuits.items()):
        # Create measurement circuit with proper classical register
        qc_measure = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
        qc_measure = qc_measure.compose(circuit)
        qc_measure.measure_all()

        # Run simulation
        try:
            job = simulator.run(transpile(qc_measure, simulator), shots=shots)
            result = job.result()
            counts = result.get_counts()
            results[label] = counts

            # Plot histogram (newer Qiskit may not support ax parameter)
            if i == 0:
                hist_fig = plot_histogram(counts, title=f"Measurements: {label}")
                # Save individual histogram
                plt.savefig(
                    f"module1_01_measurement_{i:02d}.png", dpi=300, bbox_inches="tight"
                )
                print(f"💾 Saved: module1_01_measurement_{i:02d}.png")

        except Exception as e:
            print(f"⚠️ Measurement error for {label}: {e}")
            results[label] = {}
            continue

        # Print results
        if label in results and results[label]:
            print(f"State {label} measured {shots} times:")
            for outcome, count in results[label].items():
                percentage = (count / shots) * 100
                print(f"  |{outcome}⟩: {count} times ({percentage:.1f}%)")
            print()

    plt.tight_layout()
    plt.savefig("module1_01_measurements.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results


def compare_classical_quantum():
    """Compare key differences between classical and quantum bits."""
    print("=== CLASSICAL vs QUANTUM COMPARISON ===")
    print()

    comparison = [
        ("Property", "Classical Bit", "Quantum Bit (Qubit)"),
        ("States", "0 or 1", "Superposition of 0 and 1"),
        ("Information", "1 bit", "Infinite precision (2 complex numbers)"),
        ("Measurement", "Always same result", "Probabilistic outcomes"),
        ("Copying", "Perfect copying", "No-cloning theorem"),
        ("Interaction", "Independent", "Can be entangled"),
        ("Gates", "AND, OR, NOT", "X, Y, Z, H, CNOT, etc."),
    ]

    # Print comparison table
    for row in comparison:
        print(f"{row[0]:<15} | {row[1]:<15} | {row[2]}")
        if row[0] == "Property":
            print("-" * 65)

    print()
    print("Key insights:")
    print("• Qubits can exist in superposition of 0 and 1 simultaneously")
    print("• Measurement collapses superposition to classical 0 or 1")
    print("• Quantum states carry more information than classical bits")
    print("• Quantum gates are reversible (unlike classical logic gates)")
    print()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Classical vs Quantum Bits Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Number of measurement shots (default: 1000)",
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 1, Example 1")
    print("Classical vs Quantum Bits")
    print("=" * 50)
    print()

    try:
        # Demonstrate classical bits
        classical_byte = demonstrate_classical_bits()

        # Demonstrate quantum bits
        quantum_circuits = demonstrate_quantum_bits()

        # Visualize qubit states
        states = visualize_qubit_states(quantum_circuits, args.verbose)

        # Measure qubits
        measurements = measure_qubits(quantum_circuits, args.shots)

        # Compare classical and quantum
        compare_classical_quantum()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module1_01_qubit_states.png - Bloch sphere visualizations")
        print("• module1_01_measurements.png - Measurement histograms")
        print()
        print("🎯 Key takeaways:")
        print("• Qubits can be in superposition (unlike classical bits)")
        print("• Measurement gives probabilistic results")
        print("• Quantum states contain much more information")
        print("• This is the foundation of quantum computing's power!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
