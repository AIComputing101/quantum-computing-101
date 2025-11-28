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
    """
    Show how noise affects quantum algorithm performance.
    
    MATHEMATICAL CONCEPT (For Beginners):
    ======================================
    REAL-WORLD IMPACT OF NOISE:
    We've learned about noise types (depolarizing, damping, etc.)
    But how does this affect ACTUAL quantum algorithms?
    
    TEST CASE: Deutsch-Jozsa Algorithm
    ==================================
    PROBLEM: Determine if a function f:{0,1}ⁿ → {0,1} is:
    - CONSTANT: f(x) = 0 for all x, OR f(x) = 1 for all x
    - BALANCED: f(x) = 0 for half of x, f(x) = 1 for other half
    
    CLASSICAL SOLUTION: Need to test 2ⁿ⁻¹ + 1 inputs (exponential!)
    QUANTUM SOLUTION: Need only 1 query! (exponential speedup)
    
    ALGORITHM STEPS:
    1. Prepare superposition: H^⊗n|0⟩ⁿ = Σₓ|x⟩/√(2ⁿ)
    2. Query oracle: Apply Uƒ (encodes function f)
    3. Interfere: Apply H^⊗n again
    4. Measure: |00...0⟩ → constant, anything else → balanced
    
    MATHEMATICAL GUARANTEE (Ideal):
    - Constant function → Measure |00...0⟩ with probability 1.0
    - Balanced function → Measure non-zero with probability 1.0
    
    WITH NOISE:
    - Errors corrupt superposition
    - Interference becomes imperfect
    - Wrong answer becomes possible!
    
    SUCCESS RATE DEGRADATION:
    Error rate p → Success rate ≈ (1-p)^d
    where d = circuit depth
    
    EXAMPLE:
    p = 0.01 (1% error), d = 10 gates
    Success ≈ (0.99)^10 ≈ 0.904 (90.4%)
    
    10% failure rate from just 1% per-gate errors!
    
    KEY INSIGHT: Quantum algorithms are EXPONENTIALLY sensitive to noise
    This is why error correction is essential!
    """
    print("=== ALGORITHM DEGRADATION UNDER NOISE ===")
    print()

    # ==================================================================
    # Deutsch-Jozsa Algorithm Implementation
    # ==================================================================
    def create_dj_circuit(n_qubits, function_type="constant"):
        """
        Create Deutsch-Jozsa circuit.
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        CIRCUIT STRUCTURE:
        
        Input qubits: n qubits in |0⟩
        Ancilla: 1 qubit in |1⟩ (for phase kickback)
        
        STEP 1: Initialize ancilla in |-⟩ state
        |1⟩ --H--> |-⟩ = (|0⟩ - |1⟩)/√2
        
        STEP 2: Create uniform superposition
        |0⟩^⊗n --H^⊗n--> Σₓ|x⟩/√(2ⁿ)
        
        STEP 3: Apply oracle Uƒ (phase kickback)
        For constant: Does nothing (or global phase)
        For balanced: Applies (-1)^f(x) phase to half of states
        
        STEP 4: Interference (Second Hadamard layer)
        For constant: All paths interfere constructively at |0⟩^⊗n
        For balanced: Paths interfere destructively, avoiding |0⟩^⊗n
        
        MATHEMATICAL FORMULA:
        Final state: H^⊗n · Uƒ · H^⊗n|0⟩^⊗n
        
        Constant: → |0⟩^⊗n (all amplitude at zero state)
        Balanced: → Σₓ≠₀ αₓ|x⟩ (amplitude distributed, zero at |0⟩^⊗n)
        """
        # n input qubits + 1 ancilla qubit
        qc = QuantumCircuit(n_qubits + 1, n_qubits)

        # ==============================================================
        # STEP 1: Initialize ancilla qubit to |-⟩ state
        # ==============================================================
        # MATH: |0⟩ --X--> |1⟩ --H--> |-⟩ = (|0⟩ - |1⟩)/√2
        # PURPOSE: Enable phase kickback from oracle
        qc.x(n_qubits)  # Flip ancilla to |1⟩
        
        # ==============================================================
        # STEP 2: Create uniform superposition on all qubits
        # ==============================================================
        # MATH: |0⟩ --H--> |+⟩ = (|0⟩ + |1⟩)/√2
        # For n qubits: |0⟩^⊗n --H^⊗n--> Σₓ|x⟩/√(2ⁿ)
        # This creates superposition over ALL possible n-bit strings
        for i in range(n_qubits + 1):
            qc.h(i)

        # ==============================================================
        # STEP 3: Oracle Uƒ (Simplified implementation)
        # ==============================================================
        # CONSTANT FUNCTION: Oracle does nothing (identity)
        #   Result: Global phase only, doesn't affect measurement
        #
        # BALANCED FUNCTION: Oracle applies CNOT
        #   MATH: CNOT flips ancilla based on input qubit
        #   Effect: Half of superposition terms get phase flip
        #
        # GENERAL ORACLE: Should apply (-1)^f(x) phase to each |x⟩
        # SIMPLIFIED: We use single CNOT to demonstrate balanced case
        if function_type == "balanced":
            qc.cx(0, n_qubits)  # Simplified balanced oracle

        # ==============================================================
        # STEP 4: Interference (Final Hadamard layer)
        # ==============================================================
        # MATH: H^⊗n applies Hadamard to each input qubit
        # PURPOSE: Create interference that reveals function type
        #
        # CONSTANT: Constructive interference → All amplitude at |0⟩^⊗n
        # BALANCED: Destructive interference → No amplitude at |0⟩^⊗n
        for i in range(n_qubits):
            qc.h(i)

        # ==============================================================
        # STEP 5: Measure input qubits
        # ==============================================================
        # EXPECTED OUTCOME:
        # Constant: Measure |00...0⟩ (all zeros)
        # Balanced: Measure anything EXCEPT |00...0⟩
        qc.measure(range(n_qubits), range(n_qubits))
        return qc

    # ==================================================================
    # EXPERIMENT SETUP
    # ==================================================================
    n_qubits = 2  # Test with 2-qubit Deutsch-Jozsa (4 possible inputs)
    
    # Test a range of error rates from perfect (0%) to very noisy (5%)
    # MATHEMATICAL EXPECTATION: Success rate should decay with error rate
    # Formula: Success ≈ (1-p)^depth where p = error rate, depth = # gates
    error_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    # Store results for both function types
    results = {"constant": {}, "balanced": {}}

    simulator = AerSimulator()

    # ==================================================================
    # RUN EXPERIMENTS: Test both constant and balanced functions
    # ==================================================================
    for function_type in ["constant", "balanced"]:
        print(f"Testing {function_type} function:")

        for error_rate in error_rates:
            # -------------------------------------------------------------
            # Create Deutsch-Jozsa circuit for this function type
            # -------------------------------------------------------------
            qc = create_dj_circuit(n_qubits, function_type)

            # -------------------------------------------------------------
            # Add noise model (if error_rate > 0)
            # -------------------------------------------------------------
            if error_rate > 0:
                # NOISE MODEL: Depolarizing errors on gates
                # WHY? Most common error type, affects both coherence and gates
                # MATH: Each gate has probability p of complete randomization
                error_1q = depolarizing_error(error_rate, 1)  # Single-qubit gates (H)
                error_2q = depolarizing_error(error_rate, 2)  # Two-qubit gates (CNOT)
                
                noise_model = NoiseModel()
                noise_model.add_all_qubit_quantum_error(error_1q, ["h"])
                noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
            else:
                noise_model = None  # Perfect execution (ideal case)

            # -------------------------------------------------------------
            # Run simulation with 1000 shots
            # -------------------------------------------------------------
            # STATISTICS: 1000 shots gives ~3% statistical error
            # Formula: σ ≈ √(p(1-p)/n) ≈ 0.03 for p≈0.5, n=1000
            job = simulator.run(
                transpile(qc, simulator), shots=1000, noise_model=noise_model
            )
            result = job.result()
            counts = result.get_counts()

            # -------------------------------------------------------------
            # Analyze success rate
            # -------------------------------------------------------------
            # SUCCESS CRITERION depends on function type:
            zero_string = "0" * n_qubits  # The all-zeros measurement outcome
            zero_count = counts.get(zero_string, 0)

            if function_type == "constant":
                # CONSTANT: Success = measuring |00...0⟩
                # IDEAL: Should get 100% |00...0⟩
                # NOISY: Errors cause wrong measurements
                success_rate = zero_count / 1000
            else:
                # BALANCED: Success = measuring anything EXCEPT |00...0⟩
                # IDEAL: Should NEVER get |00...0⟩ (0% probability)
                # NOISY: Errors might accidentally give |00...0⟩
                success_rate = (1000 - zero_count) / 1000

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
    """
    Characterize more realistic noise models based on real hardware.
    
    MATHEMATICAL CONCEPT (For Beginners):
    ======================================
    REALISTIC NOISE MODELING:
    Previous examples used simple, uniform noise.
    Real quantum computers have COMPLEX, GATE-DEPENDENT noise!
    
    KEY CHARACTERISTICS OF REAL HARDWARE NOISE:
    ===========================================
    
    1. GATE-DEPENDENT ERROR RATES:
       Single-qubit gates: ~0.1% error (fast, simple)
       Two-qubit gates: ~1% error (slow, complex) [10× worse!]
       
    2. GATE-TYPE VARIATIONS:
       Native gates (H, X): Lower error
       Non-native gates (Y, arbitrary rotations): Higher error
       
    3. READOUT ERRORS:
       Asymmetric: |0⟩ → |1⟩ (1-2%) vs |1⟩ → |0⟩ (5-15%)
       WHY? Excited state |1⟩ can decay during readout
       
    4. COHERENCE TIMES:
       T1 (relaxation): 50-200 μs
       T2 (dephasing): 20-100 μs (always T2 ≤ 2T1)
       
    5. CROSSTALK:
       Gates on nearby qubits can interfere
       (Not modeled in this simple example)
    
    MATHEMATICAL MODEL:
    ===================
    Total error for circuit:
    
    E_total = Σᵢ E_gate(i) + Σⱼ E_idle(j) + E_readout
    
    where:
    - E_gate(i) = error from gate i (depends on gate type)
    - E_idle(j) = decoherence during idle time j
    - E_readout = measurement error
    
    TYPICAL ERROR BUDGET:
    Single-qubit: 0.1% × n_1q gates
    Two-qubit: 1% × n_2q gates
    Readout: 2% per qubit
    
    EXAMPLE CALCULATION:
    Circuit with 10 single-qubit gates, 5 CNOTs, 2-qubit measurement:
    E ≈ 10×0.001 + 5×0.01 + 2×0.02 = 0.01 + 0.05 + 0.04 = 0.1 (10% error!)
    
    KEY INSIGHT: Two-qubit gates DOMINATE error budget!
    Circuit optimization: Minimize CNOT count first!
    """
    print("=== REALISTIC NOISE CHARACTERIZATION ===")
    print()

    # ==================================================================
    # Create realistic noise model based on actual quantum hardware
    # ==================================================================
    def create_realistic_noise_model():
        """
        Create a realistic noise model mimicking real quantum hardware.
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        NOISE MODEL COMPONENTS:
        
        1. GATE ERRORS (Depolarizing model):
           ρ → (1-p)ρ + p·I/d
           
           For qubits (d=2):
           ρ → (1-p)ρ + p·I/2
           
           MEANING: With probability p, gate completely randomizes state
           
        2. READOUT ERROR MATRIX M:
           M = [[P(measure 0|prepared 0), P(measure 0|prepared 1)],
                [P(measure 1|prepared 0), P(measure 1|prepared 1)]]
           
           ASYMMETRIC in real hardware:
           M ≈ [[0.99, 0.02],  ← More likely to incorrectly measure |0⟩ when state is |1⟩
                [0.01, 0.98]]  ← Less likely to incorrectly measure |1⟩ when state is |0⟩
           
           WHY ASYMMETRIC? |1⟩ can decay to |0⟩ during measurement!
        
        ERROR RATE HIERARCHY (Realistic values):
        =========================================
        Single-qubit gates: 0.1% (0.001)
          - Fast execution (~20-50 ns)
          - Simple pulses
          - Well-calibrated
          
        Two-qubit gates: 1% (0.01) [10× WORSE!]
          - Slow execution (~200-500 ns)
          - Complex entangling operations
          - Sensitive to calibration
          - Dominates error budget
          
        Readout: 1-2% (0.01-0.02)
          - Occurs once at circuit end
          - Fixed cost per measurement
          
        PRACTICAL IMPLICATION:
        Minimize two-qubit gates at all costs!
        1 CNOT ≈ 10 single-qubit gates in terms of error
        """
        noise_model = NoiseModel()

        # ==============================================================
        # COMPONENT 1: Single-qubit gate errors
        # ==============================================================
        # ERROR RATE: 0.1% per gate (typical for modern hardware)
        # GATES AFFECTED: H, X, Y, Z, S, T (common single-qubit gates)
        #
        # MATHEMATICAL MODEL: Depolarizing error
        # Each gate has 0.1% probability of completely randomizing qubit
        single_qubit_error = 0.001  # 0.1%
        single_qubit_gates = ["h", "x", "y", "z", "s", "t"]

        for gate in single_qubit_gates:
            # Apply same error rate to all single-qubit gates
            # ASSUMPTION: All single-qubit gates have similar fidelity
            # REALITY: Some variations exist, but this is good approximation
            error = depolarizing_error(single_qubit_error, 1)
            noise_model.add_all_qubit_quantum_error(error, gate)

        # ==============================================================
        # COMPONENT 2: Two-qubit gate errors (CNOT)
        # ==============================================================
        # ERROR RATE: 1% per CNOT (typical for modern hardware)
        # WHY 10× WORSE? 
        # - Longer gate time (more decoherence)
        # - Complex two-qubit interaction
        # - Harder to calibrate
        # - Crosstalk from neighboring qubits
        #
        # MATHEMATICAL MODEL: Two-qubit depolarizing
        # ρ → (1-p)ρ + p·I/4 (for 2-qubit system, d=4)
        two_qubit_error = 0.01  # 1%
        error = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error, "cx")

        # ==============================================================
        # COMPONENT 3: Readout (measurement) errors
        # ==============================================================
        # CONFUSION MATRIX: M[i,j] = P(measure i | prepared j)
        #
        # M = [[0.99, 0.02],  ← Columns: prepared state
        #      [0.01, 0.98]]  ← Rows: measured outcome
        #
        # READING THE MATRIX:
        # M[0,0] = 0.99: If prepared |0⟩, measure |0⟩ with 99% probability
        # M[1,0] = 0.01: If prepared |0⟩, measure |1⟩ with 1% probability
        # M[0,1] = 0.02: If prepared |1⟩, measure |0⟩ with 2% probability (DECAY!)
        # M[1,1] = 0.98: If prepared |1⟩, measure |1⟩ with 98% probability
        #
        # ASYMMETRY: 2% |1⟩→|0⟩ vs 1% |0⟩→|1⟩
        # PHYSICAL REASON: Excited state |1⟩ relaxes to ground |0⟩
        readout_error = [[0.99, 0.02],  # Row 0: Probability of measuring |0⟩
                        [0.01, 0.98]]   # Row 1: Probability of measuring |1⟩
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
