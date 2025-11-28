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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def demonstrate_classical_bits():
    """
    Demonstrate classical bit behavior and limitations.
    
    Mathematical Foundation:
    ------------------------
    A classical bit is the fundamental unit of classical information theory.
    It can exist in exactly one of two states at any given time:
    - State 0: represents "false" or "off"
    - State 1: represents "true" or "on"
    
    Mathematically, a classical bit can be represented as:
    - b ∈ {0, 1}
    
    For n classical bits, there are 2^n possible distinct states,
    but the system can only be in ONE of these states at a time.
    For example, 8 bits (1 byte) can represent 2^8 = 256 different values.
    
    Key Limitations:
    ---------------
    1. Binary nature: only two discrete states possible
    2. No superposition: cannot be "partially 0 and partially 1"
    3. Deterministic: reading the bit doesn't change its state
    4. No entanglement: bits are independent of each other
    
    Returns:
        list: A classical byte (8 bits) as a demonstration
    """
    print("=== CLASSICAL BITS ===")
    print()

    # Classical bit can only be 0 or 1 - this is deterministic
    # Unlike quantum bits, there is no probability involved
    classical_bit = 0
    print(f"Classical bit value: {classical_bit}")
    print("Possible states: 0 or 1")
    print("Properties:")
    print("- Deterministic: always gives the same value when read")
    print("- Binary: can only be in one of two states")
    print("- Independent: multiple bits don't influence each other")
    print()

    # Multiple classical bits store information in binary
    # Each bit position represents a power of 2: 2^0, 2^1, 2^2, ..., 2^7
    # Total value = Σ(bit_i × 2^i) for i from 0 to 7
    classical_byte = [0, 1, 1, 0, 1, 0, 0, 1]
    print(f"Classical byte: {classical_byte}")
    print(
        f"As decimal: {sum(bit * 2**i for i, bit in enumerate(reversed(classical_byte)))}"
    )
    print()

    return classical_byte


def demonstrate_quantum_bits():
    """
    Demonstrate quantum bit (qubit) behavior and capabilities.
    
    Mathematical Foundation:
    ------------------------
    A qubit is the fundamental unit of quantum information. Unlike classical bits,
    a qubit can exist in a superposition of states |0⟩ and |1⟩.
    
    General qubit state representation:
    |ψ⟩ = α|0⟩ + β|1⟩
    
    where:
    - |ψ⟩ (psi) is the quantum state vector (using Dirac "ket" notation)
    - α (alpha) is the complex probability amplitude for state |0⟩
    - β (beta) is the complex probability amplitude for state |1⟩
    - α, β ∈ ℂ (complex numbers)
    
    Normalization Constraint:
    -------------------------
    The total probability must equal 1, so:
    |α|² + |β|² = 1
    
    where |α|² is the probability of measuring 0, and |β|² is the probability of measuring 1.
    
    Important Quantum States:
    -------------------------
    1. |0⟩ = [1, 0]ᵀ - computational basis state (like classical 0)
    2. |1⟩ = [0, 1]ᵀ - computational basis state (like classical 1)
    3. |+⟩ = (|0⟩ + |1⟩)/√2 - equal superposition (50% chance of 0 or 1)
    4. |-⟩ = (|0⟩ - |1⟩)/√2 - equal superposition with negative phase
    5. |i⟩ = (|0⟩ + i|1⟩)/√2 - superposition with imaginary phase
    
    The factor 1/√2 ensures normalization: (1/√2)² + (1/√2)² = 1/2 + 1/2 = 1
    
    Returns:
        dict: Dictionary of quantum circuits demonstrating different qubit states
    """
    print("=== QUANTUM BITS (QUBITS) ===")
    print()

    # Create quantum circuits for different qubit states
    circuits = {}

    # |0⟩ state (computational basis state)
    # Statevector: [1, 0]ᵀ means 100% probability of measuring 0
    # This is equivalent to a classical bit with value 0
    qc_0 = QuantumCircuit(1)
    circuits["|0⟩"] = qc_0

    # |1⟩ state (computational basis state)
    # The X gate is a quantum NOT gate that flips |0⟩ to |1⟩
    # Matrix representation: X = [[0, 1], [1, 0]]
    # X|0⟩ = |1⟩, giving statevector [0, 1]ᵀ (100% probability of measuring 1)
    qc_1 = QuantumCircuit(1)
    qc_1.x(0)  # Apply X gate to flip |0⟩ to |1⟩
    circuits["|1⟩"] = qc_1

    # |+⟩ state (equal superposition - positive phase)
    # The Hadamard gate (H) creates equal superposition from |0⟩
    # H|0⟩ = (|0⟩ + |1⟩)/√2
    # Matrix: H = (1/√2)[[1, 1], [1, -1]]
    # Measurement gives 50% chance of 0 and 50% chance of 1
    qc_plus = QuantumCircuit(1)
    qc_plus.h(0)  # Apply Hadamard gate to create superposition
    circuits["|+⟩ = (|0⟩ + |1⟩)/√2"] = qc_plus

    # |-⟩ state (equal superposition - negative phase)
    # Created by X then H: H(X|0⟩) = H|1⟩ = (|0⟩ - |1⟩)/√2
    # The minus sign is a relative phase between |0⟩ and |1⟩
    # Measurement still gives 50/50, but phase matters for interference
    qc_minus = QuantumCircuit(1)
    qc_minus.x(0)
    qc_minus.h(0)
    circuits["|-⟩ = (|0⟩ - |1⟩)/√2"] = qc_minus

    # |i⟩ state (complex superposition with imaginary phase)
    # The S gate adds a 90° phase rotation
    # S = [[1, 0], [0, i]] where i = √(-1)
    # S(H|0⟩) = S((|0⟩ + |1⟩)/√2) = (|0⟩ + i|1⟩)/√2
    # The 'i' represents a complex phase - a purely quantum property!
    qc_i = QuantumCircuit(1)
    qc_i.h(0)
    qc_i.s(0)  # Apply S gate to add π/2 phase
    circuits["|i⟩ = (|0⟩ + i|1⟩)/√2"] = qc_i

    return circuits


def visualize_qubit_states(circuits, verbose=False):
    """
    Visualize qubit states on the Bloch sphere.
    
    Mathematical Foundation - The Bloch Sphere:
    -------------------------------------------
    The Bloch sphere is a geometrical representation of a single qubit state.
    Any pure qubit state can be represented as a point on the surface of a unit sphere.
    
    Parametric representation:
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩
    
    where:
    - θ (theta) ∈ [0, π] is the polar angle (latitude)
    - φ (phi) ∈ [0, 2π) is the azimuthal angle (longitude)
    
    Geometric Interpretation:
    -------------------------
    - North pole (θ=0): |0⟩ state
    - South pole (θ=π): |1⟩ state
    - Equator (θ=π/2): Equal superposition states like |+⟩, |-⟩, |i⟩, etc.
    - X-axis (θ=π/2, φ=0): |+⟩ = (|0⟩ + |1⟩)/√2
    - Y-axis (θ=π/2, φ=π/2): |i⟩ = (|0⟩ + i|1⟩)/√2
    - Z-axis: measures computational basis (|0⟩/|1⟩)
    
    Probability Calculation:
    ------------------------
    For a state |ψ⟩ = α|0⟩ + β|1⟩:
    - Probability of measuring |0⟩: P(0) = |α|² = (α × α*)
    - Probability of measuring |1⟩: P(1) = |β|² = (β × β*)
    where α* denotes the complex conjugate of α
    
    Note: P(0) + P(1) = 1 (normalization)
    
    Args:
        circuits (dict): Dictionary of quantum circuits to visualize
        verbose (bool): If True, print detailed state information
        
    Returns:
        dict: Dictionary of statevectors for each circuit
    """
    print("=== QUBIT STATE VISUALIZATION ===")
    print()

    states = {}
    bloch_figures = []

    for i, (label, circuit) in enumerate(circuits.items()):
        # Get the statevector representation of the quantum state
        # Statevector is a complex vector [α, β] where |ψ⟩ = α|0⟩ + β|1⟩
        state = Statevector.from_instruction(circuit)
        states[label] = state

        if verbose:
            print(f"State {label}:")
            print(f"  Statevector: {state}")
            # Calculate measurement probabilities using Born rule: P = |amplitude|²
            # This is computed as amplitude × complex_conjugate(amplitude)
            print(
                f"  Probabilities: |0⟩: {abs(state[0])**2:.3f}, |1⟩: {abs(state[1])**2:.3f}"
            )
            print()

        # Plot individual Bloch sphere (Qiskit 2.x doesn't support ax parameter)
        # The Bloch sphere visualization shows where this state lies on the unit sphere
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
    """
    Demonstrate measurement of different qubit states.
    
    Mathematical Foundation - Quantum Measurement (Born Rule):
    ----------------------------------------------------------
    When we measure a qubit in state |ψ⟩ = α|0⟩ + β|1⟩ in the computational basis:
    
    - Probability of outcome 0: P(0) = |α|²
    - Probability of outcome 1: P(1) = |β|²
    
    where |α|² means α × α* (amplitude times its complex conjugate).
    
    Key Properties of Quantum Measurement:
    --------------------------------------
    1. Probabilistic: Cannot predict individual outcome, only probabilities
    2. Irreversible: Measurement destroys the superposition (wave function collapse)
    3. Post-measurement state: After measuring outcome k, state becomes |k⟩
    4. Born Rule: Probability is the squared magnitude of the amplitude
    
    Why Multiple Shots?
    -------------------
    Since measurement is probabilistic, we need many repetitions (shots) to
    estimate the true probability distribution. With N shots:
    - Expected count for outcome k ≈ N × P(k)
    - Statistical error decreases as 1/√N
    
    Example: For |+⟩ = (|0⟩ + |1⟩)/√2:
    - |α|² = |(1/√2)|² = 1/2 = 50% chance of measuring 0
    - |β|² = |(1/√2)|² = 1/2 = 50% chance of measuring 1
    - With 1000 shots, expect ~500 zeros and ~500 ones
    
    Measurement Basis:
    ------------------
    This function measures in the computational (Z) basis {|0⟩, |1⟩}.
    Other bases are possible (X-basis, Y-basis) and give different results!
    
    Args:
        circuits (dict): Dictionary of quantum circuits to measure
        shots (int): Number of times to repeat each measurement
        
    Returns:
        dict: Dictionary of measurement counts for each circuit
    """
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
        # Classical register stores measurement outcomes (0 or 1)
        qc_measure = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
        qc_measure = qc_measure.compose(circuit)
        # measure_all() performs computational basis measurement on all qubits
        qc_measure.measure_all()

        # Run simulation
        # Each "shot" represents one complete experiment: prepare state → measure
        # We need multiple shots because quantum measurement is probabilistic
        try:
            job = simulator.run(transpile(qc_measure, simulator), shots=shots)
            result = job.result()
            # counts is a dictionary: {'0': count_0, '1': count_1}
            # The counts should follow the Born rule probabilities
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

        # Print results - compare empirical frequencies to theoretical probabilities
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
