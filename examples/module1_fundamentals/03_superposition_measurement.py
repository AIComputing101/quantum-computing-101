#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 3
Superposition and Measurement

This example explores quantum superposition in detail and demonstrates
how measurement affects quantum states.

Learning objectives:
- Create and analyze superposition states
- Understand measurement probabilities
- Explore the measurement collapse
- Compare different measurement bases

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter


def create_superposition_states():
    """Create various superposition states."""
    print("=== CREATING SUPERPOSITION STATES ===")
    print()
    
    circuits = {}
    
    # Equal superposition (Hadamard)
    qc_equal = QuantumCircuit(1)
    qc_equal.h(0)
    circuits['Equal: |+⟩ = (|0⟩ + |1⟩)/√2'] = qc_equal
    
    # Unequal superposition
    qc_unequal = QuantumCircuit(1)
    theta = np.pi / 3  # 60 degrees
    qc_unequal.ry(theta, 0)
    circuits[f'Unequal: cos({theta/2:.2f})|0⟩ + sin({theta/2:.2f})|1⟩'] = qc_unequal
    
    # Phase superposition
    qc_phase = QuantumCircuit(1)
    qc_phase.h(0)
    qc_phase.z(0)
    circuits['Phase: (|0⟩ - |1⟩)/√2'] = qc_phase
    
    # Complex superposition
    qc_complex = QuantumCircuit(1)
    qc_complex.h(0)
    qc_complex.s(0)
    circuits['Complex: (|0⟩ + i|1⟩)/√2'] = qc_complex
    
    # Analyze each state
    for label, circuit in circuits.items():
        state = Statevector.from_instruction(circuit)
        print(f"{label}:")
        print(f"  Statevector: {state}")
        print(f"  |0⟩ amplitude: {state[0]:.3f}")
        print(f"  |1⟩ amplitude: {state[1]:.3f}")
        print(f"  |0⟩ probability: {abs(state[0])**2:.3f}")
        print(f"  |1⟩ probability: {abs(state[1])**2:.3f}")
        print()
    
    return circuits


def demonstrate_measurement_collapse():
    """Demonstrate how measurement collapses superposition."""
    print("=== MEASUREMENT COLLAPSE ===")
    print()
    
    # Create a superposition state
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Create equal superposition
    
    print("Before measurement:")
    state_before = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    print(f"  State: {state_before}")
    print(f"  |0⟩ probability: {abs(state_before[0])**2:.3f}")
    print(f"  |1⟩ probability: {abs(state_before[1])**2:.3f}")
    print()
    
    # Simulate measurement multiple times
    simulator = AerSimulator()
    qc.measure(0, 0)
    
    shots = 1000
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    print(f"After {shots} measurements:")
    for outcome, count in counts.items():
        percentage = (count / shots) * 100
        print(f"  |{outcome}⟩: {count} times ({percentage:.1f}%)")
    print()
    
    print("Key insights:")
    print("• Before measurement: qubit is in superposition")
    print("• Each measurement gives definite result (0 or 1)")
    print("• Many measurements reveal the probabilities")
    print("• Each measurement 'collapses' the superposition")
    print()
    
    return counts


def explore_measurement_bases():
    """Explore measurement in different bases."""
    print("=== MEASUREMENT IN DIFFERENT BASES ===")
    print()
    
    # Start with |+⟩ state
    base_circuit = QuantumCircuit(1)
    base_circuit.h(0)
    
    print("Starting state: |+⟩ = (|0⟩ + |1⟩)/√2")
    print()
    
    measurements = {}
    
    # Z-basis measurement (computational basis)
    qc_z = base_circuit.copy()
    qc_z.add_register(base_circuit.cregs[0] if base_circuit.cregs else base_circuit.add_register('c', 1)[0])
    qc_z.measure_all()
    measurements['Z-basis (|0⟩, |1⟩)'] = qc_z
    
    # X-basis measurement
    qc_x = base_circuit.copy()
    qc_x.h(0)  # Rotate to X-basis
    qc_x.add_register(base_circuit.cregs[0] if base_circuit.cregs else base_circuit.add_register('c', 1)[0])
    qc_x.measure_all()
    measurements['X-basis (|+⟩, |-⟩)'] = qc_x
    
    # Y-basis measurement
    qc_y = base_circuit.copy()
    qc_y.sdg(0)  # S† gate
    qc_y.h(0)   # Rotate to Y-basis
    qc_y.add_register(base_circuit.cregs[0] if base_circuit.cregs else base_circuit.add_register('c', 1)[0])
    qc_y.measure_all()
    measurements['Y-basis (|+i⟩, |-i⟩)'] = qc_y
    
    # Simulate measurements
    simulator = AerSimulator()
    shots = 1000
    
    results = {}
    for basis, circuit in measurements.items():
        job = simulator.run(transpile(circuit, simulator), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results[basis] = counts
        
        print(f"{basis} measurement:")
        for outcome, count in counts.items():
            percentage = (count / shots) * 100
            print(f"  Outcome {outcome}: {count} times ({percentage:.1f}%)")
        print()
    
    # Visualize measurement results
    fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 3))
    if len(results) == 1:
        axes = [axes]
    
    for i, (basis, counts) in enumerate(results.items()):
        plot_histogram(counts, ax=axes[i])
        axes[i].set_title(f'{basis}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('module1_03_measurement_bases.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Key insights:")
    print("• Same quantum state gives different results in different bases")
    print("• |+⟩ is deterministic in X-basis but random in Z-basis")
    print("• Choice of measurement basis affects the information we extract")
    print()
    
    return results


def demonstrate_partial_measurements():
    """Demonstrate partial measurements on multi-qubit states."""
    print("=== PARTIAL MEASUREMENTS ===")
    print()
    
    # Create entangled state
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    
    print("Starting with Bell state: (|00⟩ + |11⟩)/√2")
    print()
    
    # Measure only first qubit
    qc_partial = qc.copy()
    qc_partial.measure(0, 0)
    
    simulator = AerSimulator()
    shots = 1000
    
    job = simulator.run(transpile(qc_partial, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    print(f"Measuring only first qubit ({shots} shots):")
    for outcome, count in counts.items():
        percentage = (count / shots) * 100
        print(f"  First qubit = {outcome[1]}: {count} times ({percentage:.1f}%)")
    print()
    
    print("What happens to the second qubit?")
    print("• If first qubit measures 0, second qubit is now in |0⟩")
    print("• If first qubit measures 1, second qubit is now in |1⟩")
    print("• The entanglement ensures perfect correlation")
    print()
    
    return counts


def analyze_superposition_parameters():
    """Analyze how rotation angle affects superposition."""
    print("=== SUPERPOSITION PARAMETERS ===")
    print()
    
    angles = np.linspace(0, np.pi, 9)  # 0 to 180 degrees
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    prob_0 = []
    prob_1 = []
    
    for theta in angles:
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        
        state = Statevector.from_instruction(qc)
        p0 = abs(state[0])**2
        p1 = abs(state[1])**2
        
        prob_0.append(p0)
        prob_1.append(p1)
        
        print(f"θ = {theta:.2f} rad ({np.degrees(theta):.0f}°):")
        print(f"  P(|0⟩) = {p0:.3f}, P(|1⟩) = {p1:.3f}")
    
    print()
    
    # Plot probabilities vs angle
    ax1.plot(np.degrees(angles), prob_0, 'bo-', label='P(|0⟩)', markersize=6)
    ax1.plot(np.degrees(angles), prob_1, 'ro-', label='P(|1⟩)', markersize=6)
    ax1.set_xlabel('Rotation Angle (degrees)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Measurement Probabilities vs Rotation Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Bloch sphere for selected angles
    selected_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    bloch_states = []
    
    for theta in selected_angles:
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        state = Statevector.from_instruction(qc)
        bloch_states.append(state)
    
    # Create a simple representation (can't easily subplot Bloch spheres)
    ax2.text(0.5, 0.5, 'Bloch Sphere Visualization\n\nθ = 0°: |0⟩ (North pole)\nθ = 45°: Superposition\nθ = 90°: Equal superposition\nθ = 135°: Superposition\nθ = 180°: |1⟩ (South pole)', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Qubit State Positions')
    
    plt.tight_layout()
    plt.savefig('module1_03_superposition_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Key insights:")
    print("• θ = 0: Pure |0⟩ state")
    print("• θ = π/2: Equal superposition")
    print("• θ = π: Pure |1⟩ state")
    print("• Probabilities follow cos²(θ/2) and sin²(θ/2)")
    print()


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description='Superposition and Measurement Demo')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--shots', type=int, default=1000,
                       help='Number of measurement shots (default: 1000)')
    args = parser.parse_args()
    
    print("🚀 Quantum Computing 101 - Module 1, Example 3")
    print("Superposition and Measurement")
    print("=" * 50)
    print()
    
    try:
        # Create superposition states
        superposition_circuits = create_superposition_states()
        
        # Demonstrate measurement collapse
        collapse_counts = demonstrate_measurement_collapse()
        
        # Explore different measurement bases
        basis_results = explore_measurement_bases()
        
        # Demonstrate partial measurements
        partial_counts = demonstrate_partial_measurements()
        
        # Analyze superposition parameters
        analyze_superposition_parameters()
        
        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module1_03_measurement_bases.png - Measurement in different bases")
        print("• module1_03_superposition_analysis.png - Superposition parameter analysis")
        print()
        print("🎯 Key takeaways:")
        print("• Superposition allows qubits to be in multiple states simultaneously")
        print("• Measurement collapses superposition to definite outcomes")
        print("• Measurement basis determines what information we extract")
        print("• Probabilities are fundamental to quantum mechanics")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit matplotlib numpy")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
