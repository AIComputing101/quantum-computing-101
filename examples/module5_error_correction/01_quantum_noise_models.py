#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5, Example 1
Quantum Noise and Error Models

This example explores different types of quantum noise and demonstrates
how errors affect quantum computations.

Learning objectives:
- Understand different quantum noise models
- Simulate realistic quantum errors
- Analyze error impact on quantum algorithms
- Characterize noise in quantum systems

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, process_fidelity, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
)
from qiskit.visualization import plot_histogram
import seaborn as sns


def demonstrate_basic_noise_types():
    """
    Demonstrate basic types of quantum noise.
    
    MATHEMATICAL BACKGROUND (For Beginners):
    ==========================================
    Quantum noise happens because quantum states are fragile. When a qubit
    interacts with its environment, it loses information. There are three main types:
    
    1. DEPOLARIZING NOISE - "Random Direction Errors"
       Mathematical Formula: ρ_noisy = (1-p)ρ + p·(I/2)
       Physical Meaning: With probability p, the qubit state gets replaced by
                        completely random noise (total chaos!)
       Example: Like randomly flipping a coin - you lose all information
    
    2. AMPLITUDE DAMPING - "Energy Loss" (T1 decay)
       Mathematical Formula: ρ_noisy = E₀·ρ·E₀† + E₁·ρ·E₁†
       where E₀ = [[1, 0], [0, √(1-γ)]], E₁ = [[0, √γ], [0, 0]]
       Physical Meaning: The excited state |1⟩ decays to ground state |0⟩
       Example: Like a battery slowly losing charge
    
    3. PHASE DAMPING - "Phase Information Loss" (T2 dephasing)  
       Mathematical Formula: ρ_noisy = (1-γ/2)ρ + (γ/2)Z·ρ·Z
       Physical Meaning: The relative phase between |0⟩ and |1⟩ gets randomized
       Example: Like two synchronized clocks slowly going out of sync
    
    KEY INSIGHT: All three noise types affect superposition states differently!
    """
    print("=== BASIC QUANTUM NOISE TYPES ===")
    print()

    # ==================================================================
    # STEP 1: Create a simple test circuit
    # ==================================================================
    # We'll create a superposition state |+⟩ = (|0⟩ + |1⟩)/√2
    # This is the MOST SENSITIVE state to noise (it has both amplitudes and phase)
    qc = QuantumCircuit(1)
    qc.h(0)  # Hadamard gate creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2

    initial_state = Statevector.from_instruction(qc)
    print(f"Initial state: {initial_state}")
    print(
        f"Initial probabilities: |0⟩: {abs(initial_state[0])**2:.3f}, |1⟩: {abs(initial_state[1])**2:.3f}"
    )
    print()

    # ==================================================================
    # STEP 2: Define different noise models
    # ==================================================================
    noise_models = {}

    # -------------------------------------------------------------------
    # 1. DEPOLARIZING NOISE - "Complete Randomization"
    # -------------------------------------------------------------------
    # MATH: With probability p, replace state with I/2 (maximally mixed state)
    #       ρ_out = (1-p)ρ + p·(I/2)
    # INTUITION: Imagine shaking a box with a coin - it becomes random!
    error_rate = 0.1  # 10% chance of noise per gate
    depol_error = depolarizing_error(error_rate, 1)  # 1 = single-qubit error
    noise_model_depol = NoiseModel()
    noise_model_depol.add_all_qubit_quantum_error(depol_error, ["h"])
    noise_models["Depolarizing"] = noise_model_depol

    # -------------------------------------------------------------------
    # 2. AMPLITUDE DAMPING - "Energy Decay" (Like a battery draining)
    # -------------------------------------------------------------------
    # MATH: Kraus operators E₀, E₁ describe energy loss
    #       E₀ preserves |0⟩, partially preserves |1⟩
    #       E₁ takes |1⟩ → |0⟩ (the decay process)
    # PHYSICAL MEANING: T1 = relaxation time (how long |1⟩ stays excited)
    # INTUITION: An atom in excited state falls back to ground state
    amp_damp_error = amplitude_damping_error(error_rate)
    noise_model_amp = NoiseModel()
    noise_model_amp.add_all_qubit_quantum_error(amp_damp_error, ["h"])
    noise_models["Amplitude Damping"] = noise_model_amp

    # -------------------------------------------------------------------
    # 3. PHASE DAMPING - "Clock Desynchronization"
    # -------------------------------------------------------------------
    # MATH: Randomly applies Z gate (phase flip): |0⟩ stays, |1⟩ → -|1⟩
    #       Superposition (|0⟩ + |1⟩) gradually loses phase coherence
    # PHYSICAL MEANING: T2 = dephasing time (how long phase info survives)
    # INTUITION: Two clocks ticking at slightly different rates
    # KEY: T2 ≤ 2T1 always (phase info more fragile than population)
    phase_damp_error = phase_damping_error(error_rate)
    noise_model_phase = NoiseModel()
    noise_model_phase.add_all_qubit_quantum_error(phase_damp_error, ["h"])
    noise_models["Phase Damping"] = noise_model_phase

    # ==================================================================
    # STEP 3: Test each noise model and measure the effects
    # ==================================================================
    # WHY MEASURE? To see how each noise type changes our superposition state
    # IDEAL RESULT: |+⟩ should give 50% |0⟩ and 50% |1⟩ when measured
    # NOISY RESULT: Different noise types will deviate differently!
    
    simulator = AerSimulator()
    results = {}

    for noise_name, noise_model in noise_models.items():
        # Add measurement to circuit
        test_circuit = qc.copy()
        test_circuit.measure_all()

        # Run with noise (1000 shots = 1000 repetitions)
        # STATISTICS: More shots = more accurate probability estimates
        # Formula: Standard error ∝ 1/√(shots)
        job = simulator.run(
            transpile(test_circuit, simulator), shots=1000, noise_model=noise_model
        )
        result = job.result()
        counts = result.get_counts()
        results[noise_name] = counts

        # Calculate probabilities from measurement counts
        # MATH: Probability = (# of outcomes) / (total shots)
        # This is Born's rule: P(i) = |⟨i|ψ⟩|²
        prob_0 = counts.get("0", 0) / 1000
        prob_1 = counts.get("1", 0) / 1000

        print(f"{noise_name} noise (error rate: {error_rate}):")
        print(f"  Measured probabilities: |0⟩: {prob_0:.3f}, |1⟩: {prob_1:.3f}")
        print(f"  Deviation from ideal: {abs(prob_0 - 0.5):.3f}")
        print(f"  INTERPRETATION: Larger deviation = more noise damage!")
        print()

    # Visualize results
    fig, axes = plt.subplots(1, len(noise_models), figsize=(4 * len(noise_models), 4))
    if len(noise_models) == 1:
        axes = [axes]

    for i, (noise_name, counts) in enumerate(results.items()):
        # plot_histogram no longer accepts ax parameter in Qiskit 2.x
        try:
            # Use matplotlib bar plot instead
            axes[i].bar(list(counts.keys()), list(counts.values()))
            axes[i].set_xlabel("Measurement Outcome")
            axes[i].set_ylabel("Counts")
        except Exception as e:
            print(f"⚠️ Could not create histogram: {e}")
        axes[i].set_title(f"{noise_name} Noise")
        axes[i].axhline(y=500, color="red", linestyle="--", alpha=0.7, label="Ideal")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("module5_01_noise_types.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results


def analyze_error_rates():
    """
    Analyze how different error rates affect quantum states.
    
    MATHEMATICAL CONCEPT (For Beginners):
    ======================================
    FIDELITY = How similar two quantum states are
    
    Mathematical Formula: F(ρ, σ) = Tr(√(√ρ σ √ρ))²
    For pure states: F = |⟨ψ|φ⟩|²
    
    Range: 0 ≤ F ≤ 1
    - F = 1.0 means states are identical (perfect!)
    - F = 0.5 means states are somewhat similar
    - F = 0.0 means states are completely different
    
    WHY IT MATTERS: Fidelity tells us how much damage noise has done.
    In quantum computing, we want F > 0.99 for useful computations!
    
    EXPERIMENT GOAL: See how fidelity degrades as error rate increases
    """
    print("=== ERROR RATE ANALYSIS ===")
    print()

    # ==================================================================
    # STEP 1: Create a test circuit - Bell state
    # ==================================================================
    # Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    # This is a MAXIMALLY ENTANGLED state - very sensitive to noise!
    # MATH: Starting from |00⟩, apply H to first qubit, then CNOT
    #       H|0⟩|0⟩ = (|0⟩+|1⟩)|0⟩/√2
    #       CNOT gives (|00⟩+|11⟩)/√2 ← Bell state!
    qc = QuantumCircuit(2)
    qc.h(0)      # Create superposition on first qubit
    qc.cx(0, 1)  # Entangle: if qubit 0 is |1⟩, flip qubit 1

    ideal_state = Statevector.from_instruction(qc)

    # ==================================================================
    # STEP 2: Test a range of error rates (from very small to large)
    # ==================================================================
    # We use logarithmic spacing: 0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%
    # WHY LOGARITHMIC? Error rates span multiple orders of magnitude!
    error_rates = np.logspace(-3, -1, 10)  # 10 points from 0.001 to 0.1

    # Store fidelity results for each noise type
    fidelities = {"Depolarizing": [], "Amplitude Damping": [], "Phase Damping": []}

    # Use density matrix method (needed for mixed states from noise)
    # MATH: Pure states → vectors |ψ⟩
    #       Mixed states → density matrices ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|
    simulator = AerSimulator(method="density_matrix")

    # ==================================================================
    # STEP 3: Loop through error rates and measure fidelity
    # ==================================================================
    for error_rate in error_rates:
        print(f"Testing error rate: {error_rate:.4f}")

        for noise_type in fidelities.keys():
            # =============================================================
            # Create noise model for this error rate
            # =============================================================
            # IMPORTANT: We need separate noise models for 1-qubit and 2-qubit gates
            # WHY? Gates have different durations and complexities
            # TYPICAL: 2-qubit gates have ~10× higher error rates than 1-qubit gates
            
            if noise_type == "Depolarizing":
                # DEPOLARIZING: Affects both single and two-qubit gates
                # MATH: ρ → (1-p)ρ + p·(I/d) where d = dimension (2 for qubits)
                error_1q = depolarizing_error(error_rate, 1)      # Single-qubit H gate
                error_2q = depolarizing_error(error_rate, 2)      # Two-qubit CNOT gate
            elif noise_type == "Amplitude Damping":
                # AMPLITUDE DAMPING: |1⟩ → |0⟩ energy decay
                # NOTE: Only defined for single qubits (it's a physical process)
                error_1q = amplitude_damping_error(error_rate)
                # For 2-qubit gates, approximate with depolarizing
                error_2q = depolarizing_error(error_rate, 2)
            else:  # Phase Damping
                # PHASE DAMPING: Phase coherence loss
                # MATH: Random Z rotations destroy phase relationships
                # NOTE: Also only for single qubits
                error_1q = phase_damping_error(error_rate)
                # For 2-qubit gates, approximate with depolarizing
                error_2q = depolarizing_error(error_rate, 2)

            # Build the complete noise model
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1q, ["h"])    # H gate gets error_1q
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])   # CNOT gets error_2q

            # =============================================================
            # Run noisy simulation
            # =============================================================
            # TECHNICAL: We save the density matrix (not just measurements)
            # WHY? Density matrix ρ contains full info about mixed states
            # PURE STATE: ρ = |ψ⟩⟨ψ| (rank-1 matrix)
            # NOISY STATE: ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ| (rank > 1, mixed state)
            qc_copy = qc.copy()
            qc_copy.save_density_matrix()  # Tell simulator to save ρ
            
            job = simulator.run(transpile(qc_copy, simulator), noise_model=noise_model)
            result = job.result()
            noisy_state = result.data()['density_matrix']

            # =============================================================
            # Calculate fidelity (how similar ideal vs noisy states are)
            # =============================================================
            # FORMULA: F = Tr(√(√ρ₁ ρ₂ √ρ₁))²
            # INTERPRETATION:
            # - F ≈ 1.0: Very little noise damage (excellent!)
            # - F ≈ 0.8-0.9: Moderate noise (usable)
            # - F < 0.7: High noise (problematic)
            fidelity = state_fidelity(ideal_state, noisy_state)
            fidelities[noise_type].append(fidelity)

    print()

    # Plot fidelity vs error rate
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "red", "green"]
    for i, (noise_type, fidelity_list) in enumerate(fidelities.items()):
        ax.semilogx(
            error_rates,
            fidelity_list,
            "o-",
            color=colors[i],
            label=noise_type,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Error Rate")
    ax.set_ylabel("State Fidelity")
    ax.set_title("State Fidelity vs Error Rate for Different Noise Types")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("module5_01_error_rate_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    return error_rates, fidelities


def demonstrate_algorithm_degradation():
    """Show how noise affects quantum algorithm performance."""
    print("=== ALGORITHM DEGRADATION UNDER NOISE ===")
    print()

    # Use Deutsch-Jozsa algorithm as example
    def create_dj_circuit(n_qubits, function_type="constant"):
        """Create Deutsch-Jozsa circuit."""
        qc = QuantumCircuit(n_qubits + 1, n_qubits)

        # Initialize
        qc.x(n_qubits)
        for i in range(n_qubits + 1):
            qc.h(i)

        # Oracle (simplified)
        if function_type == "balanced":
            qc.cx(0, n_qubits)

        # Final Hadamards
        for i in range(n_qubits):
            qc.h(i)

        qc.measure(range(n_qubits), range(n_qubits))
        return qc

    n_qubits = 2
    error_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    results = {"constant": {}, "balanced": {}}

    simulator = AerSimulator()

    for function_type in ["constant", "balanced"]:
        print(f"Testing {function_type} function:")

        for error_rate in error_rates:
            # Create circuit
            qc = create_dj_circuit(n_qubits, function_type)

            if error_rate > 0:
                # Add noise
                error_1q = depolarizing_error(error_rate, 1)
                error_2q = depolarizing_error(error_rate, 2)
                noise_model = NoiseModel()
                noise_model.add_all_qubit_quantum_error(error_1q, ["h"])
                noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
            else:
                noise_model = None

            # Run simulation
            job = simulator.run(
                transpile(qc, simulator), shots=1000, noise_model=noise_model
            )
            result = job.result()
            counts = result.get_counts()

            # Analyze success rate
            zero_string = "0" * n_qubits
            zero_count = counts.get(zero_string, 0)

            if function_type == "constant":
                success_rate = zero_count / 1000  # Should be ~1 for constant
            else:
                success_rate = (1000 - zero_count) / 1000  # Should be ~1 for balanced

            results[function_type][error_rate] = success_rate
            print(f"  Error rate {error_rate:.3f}: Success rate {success_rate:.3f}")

        print()

    # Plot algorithm performance degradation
    fig, ax = plt.subplots(figsize=(10, 6))

    for function_type, data in results.items():
        error_rates_list = list(data.keys())
        success_rates = list(data.values())

        ax.plot(
            error_rates_list,
            success_rates,
            "o-",
            label=f"{function_type.capitalize()} Function",
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Error Rate")
    ax.set_ylabel("Algorithm Success Rate")
    ax.set_title("Deutsch-Jozsa Algorithm Performance vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("module5_01_algorithm_degradation.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results


def characterize_realistic_noise():
    """Characterize more realistic noise models."""
    print("=== REALISTIC NOISE CHARACTERIZATION ===")
    print()

    # Create a more complex noise model with gate-dependent errors
    def create_realistic_noise_model():
        """Create a realistic noise model."""
        noise_model = NoiseModel()

        # Single-qubit gate errors
        single_qubit_error = 0.001
        single_qubit_gates = ["h", "x", "y", "z", "s", "t"]

        for gate in single_qubit_gates:
            error = depolarizing_error(single_qubit_error, 1)
            noise_model.add_all_qubit_quantum_error(error, gate)

        # Two-qubit gate errors (higher error rate)
        two_qubit_error = 0.01
        error = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error, "cx")

        # Measurement errors
        readout_error = [[0.99, 0.01], [0.02, 0.98]]  # Readout error matrix
        noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

    realistic_noise = create_realistic_noise_model()

    # Test various quantum circuits
    test_circuits = {}

    # Simple circuit
    qc1 = QuantumCircuit(1, 1)
    qc1.h(0)
    qc1.measure(0, 0)
    test_circuits["Single H gate"] = qc1

    # Multiple gates
    qc2 = QuantumCircuit(2, 2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.h(0)
    qc2.h(1)
    qc2.measure_all()
    test_circuits["Bell + H gates"] = qc2

    # Deep circuit
    qc3 = QuantumCircuit(3, 3)
    for layer in range(5):
        for qubit in range(3):
            qc3.h(qubit)
        for qubit in range(2):
            qc3.cx(qubit, qubit + 1)
    qc3.measure_all()
    test_circuits["Deep circuit (5 layers)"] = qc3

    # Compare ideal vs noisy results
    simulator = AerSimulator()
    comparison_results = {}

    for circuit_name, circuit in test_circuits.items():
        print(f"Testing {circuit_name}:")

        # Ideal simulation
        job_ideal = simulator.run(transpile(circuit, simulator), shots=1000)
        ideal_counts = job_ideal.result().get_counts()

        # Noisy simulation
        job_noisy = simulator.run(
            transpile(circuit, simulator), shots=1000, noise_model=realistic_noise
        )
        noisy_counts = job_noisy.result().get_counts()

        comparison_results[circuit_name] = {
            "ideal": ideal_counts,
            "noisy": noisy_counts,
        }

        # Calculate total variation distance
        all_outcomes = set(ideal_counts.keys()) | set(noisy_counts.keys())
        tv_distance = 0
        for outcome in all_outcomes:
            p_ideal = ideal_counts.get(outcome, 0) / 1000
            p_noisy = noisy_counts.get(outcome, 0) / 1000
            tv_distance += abs(p_ideal - p_noisy)
        tv_distance /= 2

        print(f"  Total variation distance: {tv_distance:.3f}")
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Gate count: {sum(circuit.count_ops().values())}")
        print()

    # Visualize comparison
    fig, axes = plt.subplots(
        len(test_circuits), 2, figsize=(12, 4 * len(test_circuits))
    )

    for i, (circuit_name, results) in enumerate(comparison_results.items()):
        # Ideal results
        # plot_histogram no longer accepts ax parameter in Qiskit 2.x
        try:
            # Use matplotlib bar plot instead
            axes[i, 0].bar(
                list(results["ideal"].keys()), list(results["ideal"].values())
            )
            axes[i, 0].set_xlabel("Measurement Outcome")
            axes[i, 0].set_ylabel("Counts")
        except Exception as e:
            print(f"⚠️ Could not create histogram: {e}")
        axes[i, 0].set_title(f"{circuit_name} - Ideal")

        # Noisy results
        # plot_histogram no longer accepts ax parameter in Qiskit 2.x
        try:
            # Use matplotlib bar plot instead
            axes[i, 1].bar(
                list(results["noisy"].keys()), list(results["noisy"].values())
            )
            axes[i, 1].set_xlabel("Measurement Outcome")
            axes[i, 1].set_ylabel("Counts")
        except Exception as e:
            print(f"⚠️ Could not create histogram: {e}")
        axes[i, 1].set_title(f"{circuit_name} - Noisy")

    plt.tight_layout()
    plt.savefig("module5_01_realistic_noise.png", dpi=300, bbox_inches="tight")
    plt.close()

    return comparison_results


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Quantum Noise and Error Models Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.01,
        help="Base error rate for demonstrations (default: 0.01)",
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 5, Example 1")
    print("Quantum Noise and Error Models")
    print("=" * 50)
    print()

    try:
        # Demonstrate basic noise types
        noise_results = demonstrate_basic_noise_types()

        # Analyze error rates
        error_rates, fidelities = analyze_error_rates()

        # Show algorithm degradation
        algorithm_results = demonstrate_algorithm_degradation()

        # Characterize realistic noise
        realistic_results = characterize_realistic_noise()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module5_01_noise_types.png - Different noise type effects")
        print("• module5_01_error_rate_analysis.png - Fidelity vs error rate")
        print(
            "• module5_01_algorithm_degradation.png - Algorithm performance under noise"
        )
        print("• module5_01_realistic_noise.png - Realistic noise model comparison")
        print()
        print("🎯 Key takeaways:")
        print("• Quantum systems are inherently noisy and fragile")
        print("• Different noise types affect quantum states differently")
        print("• Error rates have exponential impact on algorithm performance")
        print("• Realistic noise models include gate-dependent and readout errors")
        print(
            "• Error correction and mitigation are essential for practical quantum computing"
        )

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy seaborn")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
