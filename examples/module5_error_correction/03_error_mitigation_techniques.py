#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5: Error Correction
Example 3: Error Mitigation Techniques

Implementation of various quantum error mitigation techniques.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
import warnings

warnings.filterwarnings("ignore")


class ErrorMitigation:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def create_noise_model(self, p_depolar=0.01, p_damping=0.005):
        """Create noise model for simulation."""
        noise_model = NoiseModel()

        # Depolarizing error on single-qubit gates
        error_1q = depolarizing_error(p_depolar, 1)
        noise_model.add_all_qubit_quantum_error(
            error_1q, ["h", "x", "y", "z", "ry", "rx", "rz"]
        )

        # Depolarizing error on two-qubit gates
        error_2q = depolarizing_error(p_depolar * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cy", "cz"])

        # Amplitude damping
        error_damping = amplitude_damping_error(p_damping)
        noise_model.add_all_qubit_quantum_error(error_damping, ["id"])

        return noise_model

    def zero_noise_extrapolation(self, circuit, noise_factors=[1, 2, 3], shots=1024):
        """
        Zero-Noise Extrapolation (ZNE) - A Clever Error Mitigation Trick!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE BIG IDEA: If we can't eliminate noise, let's add MORE noise
        intentionally, then mathematically extrapolate back to zero noise!
        
        HOW IT WORKS:
        1. Run circuit with normal noise → Get result at noise level λ₁
        2. Run circuit with 2× noise → Get result at noise level λ₂  
        3. Run circuit with 3× noise → Get result at noise level λ₃
        4. Fit a curve through these points
        5. Extrapolate the curve back to λ=0 (zero noise!)
        
        MATHEMATICAL FORMULA:
        Assume: E(λ) = a + b·λ + c·λ² (expectation value vs noise)
        Measure: E(λ₁), E(λ₂), E(λ₃)
        Extrapolate: E(0) = a ← This is our estimate of the ideal result!
        
        ANALOGY: Like drawing a line through noisy data points and reading
                 where it crosses the y-axis (zero noise)
        
        LIMITATION: Works best when noise scales smoothly (linear or polynomial)
        OVERHEAD: Need to run circuit multiple times (2-5× more shots)
        """
        results = []

        # =============================================================
        # Loop through different noise scaling factors
        # =============================================================
        for factor in noise_factors:
            # NOISE SCALING: We artificially increase noise by factor
            # HOW? Add extra gates that cancel out (U·U†) but add noise
            # MATH: If original noise = λ, scaled noise ≈ factor × λ
            scaled_circuit = self.scale_circuit_noise(circuit, factor)

            # Simulate with noise
            noise_model = self.create_noise_model()
            simulator = AerSimulator(noise_model=noise_model)

            job = simulator.run(scaled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Calculate expectation value (assuming measurement of Z on first qubit)
            # MATH: ⟨Z⟩ = P(0) - P(1) = probability of |0⟩ minus probability of |1⟩
            # Range: -1 (all |1⟩) to +1 (all |0⟩)
            expectation = self.calculate_expectation_z(counts)
            results.append((factor, expectation))

        # =============================================================
        # Extrapolate to zero noise (the magic step!)
        # =============================================================
        factors = [r[0] for r in results]        # Noise scaling factors [1, 2, 3]
        expectations = [r[1] for r in results]   # Measured ⟨Z⟩ values

        # Linear extrapolation: E(λ) = a + b·λ
        # MATH: Fit line through points, then evaluate at λ=0
        # np.polyfit returns [slope, intercept] for degree 1 polynomial
        coeffs = np.polyfit(factors, expectations, 1)
        zero_noise_estimate = coeffs[1]  # y-intercept = E(0) ← Zero noise estimate!
        
        # INTERPRETATION: zero_noise_estimate is our best guess at the ideal result
        # without noise, computed by extrapolating from noisy measurements!

        return {
            "measurements": results,
            "zero_noise_estimate": zero_noise_estimate,
            "extrapolation_coeffs": coeffs,
        }

    def scale_circuit_noise(self, circuit, factor):
        """Scale circuit noise by repetition."""
        if factor == 1:
            return circuit.copy()

        # Create new circuit by copying the original (preserves registers)
        scaled_circuit = circuit.copy()
        
        # Clear the circuit data but keep the registers
        scaled_circuit.data.clear()

        for instruction in circuit.data:
            # Add original instruction
            scaled_circuit.append(instruction)

            # Add repetitions for noise scaling (simplified)
            if instruction.operation.name in ["h", "x", "y", "z"]:
                for _ in range(factor - 1):
                    # Add pairs of operations that cancel out but add noise
                    scaled_circuit.append(instruction)
                    scaled_circuit.append(instruction)

        return scaled_circuit

    def readout_error_mitigation(self, circuit, shots=1024):
        """
        Readout error mitigation using calibration - Fix measurement mistakes!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE PROBLEM: Measurement errors corrupt results
        When we measure |0⟩, might get |1⟩ (and vice versa)
        
        THE SOLUTION: Calibration + Matrix Inversion
        
        STEP-BY-STEP PROCESS:
        =====================
        
        STEP 1: CALIBRATION
        Prepare EVERY possible basis state, measure it
        Build "confusion matrix" M that describes measurement errors
        
        STEP 2: INVERSION
        Compute M^(-1) (inverse of confusion matrix)
        
        STEP 3: CORRECTION
        For noisy measurement results p_noisy:
        Corrected results: p_ideal = M^(-1) · p_noisy
        
        MATHEMATICAL FORMULA:
        =====================
        Measurement process: p_measured = M · p_true
        
        where:
        M[i,j] = P(measure i | prepared j)
        p_true = true probability distribution
        p_measured = what we actually observe
        
        INVERSION:
        p_true = M^(-1) · p_measured
        
        EXAMPLE (2 qubits):
        Prepare |00⟩, |01⟩, |10⟩, |11⟩
        Measure each 1000 times → Build 4×4 matrix M
        
        M = [[P(00|00), P(00|01), P(00|10), P(00|11)],
             [P(01|00), P(01|01), P(01|10), P(01|11)],
             [P(10|00), P(10|01), P(10|10), P(10|11)],
             [P(11|00), P(11|01), P(11|10), P(11|11)]]
        
        IDEAL (No errors): M = I (identity matrix)
        REALISTIC: M has off-diagonal elements (errors!)
        
        WHY THIS WORKS:
        Measurement errors are CLASSICAL (not quantum)
        Can be described by stochastic matrix
        Linear algebra: Invertible if errors not too large
        
        LIMITATION:
        Works only if M is invertible (determinant ≠ 0)
        Requires: errors < 50% (otherwise information lost)
        
        OVERHEAD:
        Need 2^n calibration circuits for n qubits
        2 qubits → 4 circuits
        3 qubits → 8 circuits
        4 qubits → 16 circuits (feasible for small n)
        """
        # ==============================================================
        # STEP 1: CALIBRATION - Prepare all basis states
        # ==============================================================
        # GOAL: Learn what the measurement errors are
        # METHOD: Prepare known states, measure them, build confusion matrix
        
        cal_circuits = []
        n_qubits = circuit.num_qubits
        n_states = 2**n_qubits  # Total number of basis states
        # EXAMPLE: 2 qubits → 4 states |00⟩, |01⟩, |10⟩, |11⟩

        # Build calibration circuit for each basis state
        # LOOP: state_idx from 0 to 2^n - 1
        for state_idx in range(n_states):
            cal_circuit = QuantumCircuit(n_qubits)
            
            # Prepare basis state by applying X gates
            # BIT MANIPULATION TRICK:
            # state_idx = 0 (binary: 00) → No X gates → |00⟩
            # state_idx = 1 (binary: 01) → X on qubit 0 → |01⟩  
            # state_idx = 2 (binary: 10) → X on qubit 1 → |10⟩
            # state_idx = 3 (binary: 11) → X on both → |11⟩
            for qubit_idx in range(n_qubits):
                # Check if bit qubit_idx is set in state_idx
                # (state_idx >> qubit_idx) shifts bits right
                # & 1 checks if least significant bit is 1
                if (state_idx >> qubit_idx) & 1:
                    cal_circuit.x(qubit_idx)  # Flip this qubit to |1⟩
                    
            cal_circuit.measure_all()
            cal_circuits.append(cal_circuit)

        # ==============================================================
        # STEP 2: Run calibration circuits with noise
        # ==============================================================
        # MEASURE: What do we actually observe when we prepare each state?
        # This tells us the confusion matrix M
        noise_model = self.create_noise_model()
        simulator = AerSimulator(noise_model=noise_model)

        cal_results = []
        for cal_circuit in cal_circuits:
            job = simulator.run(cal_circuit, shots=shots)
            result = job.result()
            cal_results.append(result.get_counts())
            # Each result contains measurement distribution for one basis state

        # ==============================================================
        # STEP 3: Build calibration (confusion) matrix M
        # ==============================================================
        # MATHEMATICAL CONSTRUCTION: M[i,j] from calibration data
        # Column j: Results when we prepared state j
        # Row i: Probability of measuring state i
        cal_matrix = self.build_calibration_matrix(cal_results, circuit.num_qubits)

        # ==============================================================
        # STEP 4: Run actual circuit with noise
        # ==============================================================
        # This is the circuit whose results we want to correct
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        noisy_counts = result.get_counts()

        # ==============================================================
        # STEP 5: Apply mitigation (matrix inversion)
        # ==============================================================
        # MATHEMATICAL OPERATION: p_ideal = M^(-1) · p_noisy
        # This "undoes" the measurement errors!
        mitigated_counts = self.apply_readout_mitigation(noisy_counts, cal_matrix)

        return {
            "noisy_counts": noisy_counts,
            "mitigated_counts": mitigated_counts,
            "calibration_matrix": cal_matrix,
        }

    def build_calibration_matrix(self, cal_results, n_qubits):
        """
        Build calibration matrix from calibration results.
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        CONFUSION MATRIX (Calibration Matrix) M:
        Describes how measurement errors transform probabilities
        
        MATRIX STRUCTURE (n qubits → 2^n × 2^n matrix):
        M[i,j] = P(measure state i | prepared state j)
        
        EXAMPLE (2 qubits, 4×4 matrix):
                Prepared: |00⟩  |01⟩  |10⟩  |11⟩
        Measured |00⟩: [ M₀₀   M₀₁   M₀₂   M₀₃ ]
                |01⟩: [ M₁₀   M₁₁   M₁₂   M₁₃ ]
                |10⟩: [ M₂₀   M₂₁   M₂₂   M₂₃ ]
                |11⟩: [ M₃₀   M₃₁   M₃₂   M₃₃ ]
        
        READING THE MATRIX:
        Column j: Distribution we measure when preparing state j
        Row i: Probability of observing outcome i
        
        IDEAL (No errors): M = I (identity)
        REALISTIC: M has off-diagonal elements
        
        CONSTRUCTION ALGORITHM:
        =======================
        FOR each prepared state j = 0 to 2^n-1:
            1. Prepared state j
            2. Measure 1000 times
            3. Count how often we get each outcome i
            4. M[i,j] = count(i) / 1000
        
        MATHEMATICAL PROPERTY:
        Each column sums to 1 (it's a probability distribution)
        Σᵢ M[i,j] = 1 for all j
        
        Matrix is STOCHASTIC (non-negative, columns sum to 1)
        
        WHY WE NEED THIS:
        Once we know M, we can invert it to correct future measurements!
        """
        # ==============================================================
        # Initialize calibration matrix
        # ==============================================================
        # SIZE: 2^n × 2^n for n qubits
        # EXAMPLE: 2 qubits → 4×4 matrix, 3 qubits → 8×8 matrix
        n_states = 2**n_qubits
        cal_matrix = np.zeros((n_states, n_states))

        # ==============================================================
        # Fill matrix from calibration data
        # ==============================================================
        # i = prepared state index (which calibration circuit)
        # counts = measurement results for that prepared state
        for i, counts in enumerate(cal_results):
            total_shots = sum(counts.values())
            
            # Loop through all measured outcomes
            for state, count in counts.items():
                # CLEAN STATE STRING: Remove spaces (Qiskit formatting)
                # EXAMPLE: "0 1" → "01"
                state_clean = state.replace(" ", "")
                
                # CONVERT BINARY STRING TO INTEGER
                # EXAMPLE: "01" → 1, "10" → 2, "11" → 3
                # MATH: Binary to decimal conversion
                state_int = int(state_clean, 2)
                
                # FILL MATRIX ENTRY M[measured_state, prepared_state]
                # PROBABILITY: count / total_shots
                # EXAMPLE: Prepared |00⟩ (i=0), measured |01⟩ 50 times out of 1000
                #          → M[1, 0] = 50/1000 = 0.05 (5% error)
                cal_matrix[state_int, i] = count / total_shots

        # ==============================================================
        # RESULT: Calibration matrix M fully populated
        # ==============================================================
        # Each column j: Probability distribution when preparing state j
        # Diagonal elements: Probability of correct measurement (should be high!)
        # Off-diagonal: Probability of incorrect measurement (errors)
        
        return cal_matrix

    def apply_readout_mitigation(self, counts, cal_matrix):
        """
        Apply readout error mitigation - The correction step!
        
        MATHEMATICAL CONCEPT (For Beginners):
        ======================================
        THE CORRECTION FORMULA:
        p_ideal = M^(-1) · p_noisy
        
        WHERE:
        - M = calibration (confusion) matrix
        - M^(-1) = inverse matrix (undoes the errors!)
        - p_noisy = what we measured (with errors)
        - p_ideal = what we SHOULD have measured (without errors)
        
        STEP-BY-STEP PROCESS:
        =====================
        
        STEP 1: Convert measurement counts to probability vector
        Counts: {'00': 450, '01': 300, '10': 200, '11': 50}
        Total: 1000 shots
        Probabilities: [0.45, 0.30, 0.20, 0.05]
        
        STEP 2: Compute inverse calibration matrix
        M^(-1) = inv(M)
        
        WHY IT WORKS: If measurement transforms as p_noisy = M·p_ideal,
        then p_ideal = M^(-1)·p_noisy (multiply both sides by M^(-1))
        
        MATHEMATICAL VERIFICATION:
        M^(-1) · M = I (identity)
        M^(-1) · (M · p_ideal) = M^(-1) · p_noisy
        (M^(-1) · M) · p_ideal = M^(-1) · p_noisy
        I · p_ideal = M^(-1) · p_noisy
        p_ideal = M^(-1) · p_noisy ✓
        
        STEP 3: Matrix multiplication
        p_ideal = M^(-1) @ p_noisy
        
        EXAMPLE (2 qubits):
        p_noisy = [0.45, 0.30, 0.20, 0.05]  (what we measured)
        M^(-1) ≈ [[1.02, -0.01, -0.01, 0],
                  [-0.01, 1.03, 0, -0.02],
                  [-0.01, 0, 1.03, -0.02],
                  [0, -0.02, -0.02, 1.05]]
        p_ideal ≈ M^(-1) @ p_noisy (corrected probabilities!)
        
        STEP 4: Convert back to counts
        mitigated_counts = p_ideal × total_shots
        
        IMPORTANT NOTES:
        ================
        1. Mitigated probabilities can be NEGATIVE (quasi-probabilities)
           This is OK! It's a mathematical artifact of error correction
           Negative values typically small (< 5%)
           
        2. Matrix must be INVERTIBLE
           If errors too large (> 50%), M becomes singular (can't invert)
           
        3. Statistical noise amplification
           Inversion amplifies statistical fluctuations
           Need more shots for mitigated results to converge
        
        PRACTICAL BENEFIT:
        Typical improvement: 2-5× reduction in measurement errors
        Cost: Just classical post-processing (essentially free!)
        """
        # ==============================================================
        # STEP 1: Convert counts dictionary to probability vector
        # ==============================================================
        # VECTOR FORMAT: [P(|00⟩), P(|01⟩), P(|10⟩), P(|11⟩), ...]
        # SIZE: 2^n elements for n qubits
        n_states = cal_matrix.shape[0]
        total_shots = sum(counts.values())

        prob_vector = np.zeros(n_states)
        for state, count in counts.items():
            # Clean state string (remove Qiskit formatting spaces)
            state_clean = state.replace(" ", "")
            # Convert binary string to integer index
            # EXAMPLE: "10" → int("10", 2) = 2
            state_int = int(state_clean, 2)
            # Calculate probability for this state
            prob_vector[state_int] = count / total_shots

        # ==============================================================
        # STEP 2: Invert calibration matrix and apply correction
        # ==============================================================
        try:
            # COMPUTE INVERSE: M^(-1)
            # LINEAR ALGEBRA: Uses LU decomposition or similar
            # COMPLEXITY: O(n³) for n×n matrix
            # EXAMPLE: 4×4 matrix → ~64 operations
            inv_cal_matrix = np.linalg.inv(cal_matrix)
            
            # APPLY CORRECTION: Matrix-vector multiplication
            # MATHEMATICAL OPERATION: p_ideal = M^(-1) @ p_noisy
            # @ = matrix multiplication operator in Python/NumPy
            # RESULT: Corrected probability distribution
            mitigated_probs = inv_cal_matrix @ prob_vector

            # ==============================================================
            # STEP 3: Convert corrected probabilities back to counts
            # ==============================================================
            mitigated_counts = {}
            for i, prob in enumerate(mitigated_probs):
                # QUASI-PROBABILITIES: Can be negative due to inversion
                # Filter: Only include non-negative (prob > 0)
                # MATHEMATICAL NOTE: Small negative values are OK and expected
                # They arise from statistical fluctuations + inversion
                if prob > 0:
                    # Convert state index to binary string
                    # EXAMPLE: i=2, n=2 qubits → "10"
                    state_str = format(i, f"0{int(np.log2(n_states))}b")
                    # Scale back to counts (multiply by total shots)
                    mitigated_counts[state_str] = int(prob * total_shots)

            return mitigated_counts

        except np.linalg.LinAlgError:
            # ==============================================================
            # ERROR HANDLING: Matrix inversion failed
            # ==============================================================
            # REASONS:
            # 1. Matrix is singular (determinant = 0)
            # 2. Errors too large (> 50%) → Matrix not invertible
            # 3. Numerical instability
            #
            # FALLBACK: Return original noisy counts (no mitigation)
            print("⚠️  Warning: Calibration matrix not invertible!")
            print("    Errors may be too large for mitigation")
            return counts

    def symmetry_verification(self, circuit, symmetry_circuits, shots=1024):
        """Symmetry verification for error detection."""
        # Run original circuit
        noise_model = self.create_noise_model()
        simulator = AerSimulator(noise_model=noise_model)

        job = simulator.run(circuit, shots=shots)
        result = job.result()
        original_counts = result.get_counts()

        # Run symmetry verification circuits
        symmetry_results = []
        for sym_circuit in symmetry_circuits:
            job = simulator.run(sym_circuit, shots=shots)
            result = job.result()
            sym_counts = result.get_counts()

            # Check symmetry violation
            violation = self.calculate_symmetry_violation(original_counts, sym_counts)
            symmetry_results.append(
                {"circuit": sym_circuit, "counts": sym_counts, "violation": violation}
            )

        return {
            "original_counts": original_counts,
            "symmetry_results": symmetry_results,
            "average_violation": np.mean([r["violation"] for r in symmetry_results]),
        }

    def calculate_symmetry_violation(self, counts1, counts2):
        """Calculate symmetry violation metric."""
        # Simple metric: difference in expectation values
        exp1 = self.calculate_expectation_z(counts1)
        exp2 = self.calculate_expectation_z(counts2)
        return abs(exp1 - exp2)

    def calculate_expectation_z(self, counts):
        """Calculate expectation value of Z measurement on first qubit."""
        total_shots = sum(counts.values())
        expectation = 0

        for state, count in counts.items():
            # Remove spaces and get first bit
            state_clean = state.replace(" ", "")
            # Z eigenvalue is +1 for |0⟩, -1 for |1⟩ on first qubit
            if state_clean[0] == "0":
                expectation += count / total_shots
            else:
                expectation -= count / total_shots

        return expectation

    def compare_mitigation_methods(self, test_circuit, shots=1024):
        """Compare different mitigation methods."""
        results = {}

        # No mitigation (with noise)
        noise_model = self.create_noise_model()
        simulator = AerSimulator(noise_model=noise_model)
        job = simulator.run(test_circuit, shots=shots)
        result = job.result()
        results["noisy"] = {
            "counts": result.get_counts(),
            "expectation": self.calculate_expectation_z(result.get_counts()),
        }

        # Ideal (no noise)
        ideal_simulator = AerSimulator()
        job = ideal_simulator.run(test_circuit, shots=shots)
        result = job.result()
        results["ideal"] = {
            "counts": result.get_counts(),
            "expectation": self.calculate_expectation_z(result.get_counts()),
        }

        # Zero noise extrapolation
        zne_result = self.zero_noise_extrapolation(test_circuit, shots=shots)
        results["zne"] = {
            "expectation": zne_result["zero_noise_estimate"],
            "measurements": zne_result["measurements"],
        }

        # Readout error mitigation
        rem_result = self.readout_error_mitigation(test_circuit, shots=shots)
        results["rem"] = {
            "noisy_counts": rem_result["noisy_counts"],
            "mitigated_counts": rem_result["mitigated_counts"],
            "expectation": self.calculate_expectation_z(rem_result["mitigated_counts"]),
        }

        return results

    def visualize_mitigation_results(self, comparison_results):
        """Visualize mitigation method comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Expectation value comparison
        methods = ["Ideal", "Noisy", "ZNE", "REM"]
        expectations = [
            comparison_results["ideal"]["expectation"],
            comparison_results["noisy"]["expectation"],
            comparison_results["zne"]["expectation"],
            comparison_results["rem"]["expectation"],
        ]
        colors = ["green", "red", "blue", "orange"]

        bars = ax1.bar(methods, expectations, alpha=0.7, color=colors)
        ax1.set_title("Expectation Value Comparison")
        ax1.set_ylabel("⟨Z⟩")
        ax1.axhline(
            y=comparison_results["ideal"]["expectation"],
            color="green",
            linestyle="--",
            alpha=0.5,
            label="Ideal",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Error analysis
        ideal_exp = comparison_results["ideal"]["expectation"]
        errors = [
            0,  # Ideal
            abs(comparison_results["noisy"]["expectation"] - ideal_exp),
            abs(comparison_results["zne"]["expectation"] - ideal_exp),
            abs(comparison_results["rem"]["expectation"] - ideal_exp),
        ]

        ax2.bar(methods, errors, alpha=0.7, color=colors)
        ax2.set_title("Absolute Error from Ideal")
        ax2.set_ylabel("|Error|")
        ax2.grid(True, alpha=0.3)

        # Zero noise extrapolation details
        if "measurements" in comparison_results["zne"]:
            zne_data = comparison_results["zne"]["measurements"]
            factors = [d[0] for d in zne_data]
            measured_exps = [d[1] for d in zne_data]

            ax3.scatter(factors, measured_exps, color="blue", s=50, alpha=0.7)

            # Fit line
            coeffs = np.polyfit(factors, measured_exps, 1)
            x_fit = np.linspace(0, max(factors), 100)
            y_fit = coeffs[0] * x_fit + coeffs[1]
            ax3.plot(x_fit, y_fit, "b--", alpha=0.7)

            ax3.axhline(
                y=coeffs[1],
                color="red",
                linestyle=":",
                label=f"ZNE Estimate: {coeffs[1]:.3f}",
            )
            ax3.set_title("Zero Noise Extrapolation")
            ax3.set_xlabel("Noise Factor")
            ax3.set_ylabel("⟨Z⟩")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Improvement metrics
        noisy_error = abs(comparison_results["noisy"]["expectation"] - ideal_exp)
        zne_error = abs(comparison_results["zne"]["expectation"] - ideal_exp)
        rem_error = abs(comparison_results["rem"]["expectation"] - ideal_exp)

        improvements = []
        if noisy_error > 0:
            zne_improvement = (noisy_error - zne_error) / noisy_error * 100
            rem_improvement = (noisy_error - rem_error) / noisy_error * 100
            improvements = [zne_improvement, rem_improvement]

            ax4.bar(["ZNE", "REM"], improvements, alpha=0.7, color=["blue", "orange"])
            ax4.set_title("Error Reduction (%)")
            ax4.set_ylabel("Improvement (%)")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum Error Mitigation Techniques")
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Noise level")
    parser.add_argument(
        "--method",
        choices=["zne", "rem", "sv", "all"],
        default="all",
        help="Mitigation method",
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 5: Error Correction")
    print("Example 3: Error Mitigation Techniques")
    print("=" * 42)

    mitigator = ErrorMitigation(verbose=args.verbose)

    try:
        # Create test circuit (don't pre-create classical bits, measure_all will add them)
        test_circuit = QuantumCircuit(2)
        test_circuit.h(0)
        test_circuit.cx(0, 1)
        test_circuit.measure_all()

        print(f"\n🧪 Test Circuit:")
        print(f"   Qubits: {test_circuit.num_qubits}")
        print(f"   Depth: {test_circuit.depth()}")
        print(f"   Gates: {test_circuit.size()}")
        print(f"   Noise level: {args.noise_level}")

        if args.method == "all":
            # Compare all methods
            print(f"\n🔄 Comparing mitigation methods...")
            results = mitigator.compare_mitigation_methods(test_circuit, args.shots)

            print(f"\n📊 Results Summary:")
            print(f"   Ideal:     ⟨Z⟩ = {results['ideal']['expectation']:.4f}")
            print(f"   Noisy:     ⟨Z⟩ = {results['noisy']['expectation']:.4f}")
            print(f"   ZNE:       ⟨Z⟩ = {results['zne']['expectation']:.4f}")
            print(f"   REM:       ⟨Z⟩ = {results['rem']['expectation']:.4f}")

            # Calculate improvements
            ideal_exp = results["ideal"]["expectation"]
            noisy_error = abs(results["noisy"]["expectation"] - ideal_exp)
            zne_error = abs(results["zne"]["expectation"] - ideal_exp)
            rem_error = abs(results["rem"]["expectation"] - ideal_exp)

            if noisy_error > 0:
                zne_improvement = (noisy_error - zne_error) / noisy_error * 100
                rem_improvement = (noisy_error - rem_error) / noisy_error * 100

                print(f"\n📈 Error Reduction:")
                print(f"   ZNE: {zne_improvement:.1f}%")
                print(f"   REM: {rem_improvement:.1f}%")

            if args.show_visualization:
                mitigator.visualize_mitigation_results(results)

        else:
            # Run specific method
            if args.method == "zne":
                print(f"\n🎯 Zero Noise Extrapolation...")
                result = mitigator.zero_noise_extrapolation(
                    test_circuit, shots=args.shots
                )
                print(f"   Estimate: {result['zero_noise_estimate']:.4f}")

            elif args.method == "rem":
                print(f"\n🎯 Readout Error Mitigation...")
                result = mitigator.readout_error_mitigation(
                    test_circuit, shots=args.shots
                )
                noisy_exp = mitigator.calculate_expectation_z(result["noisy_counts"])
                mitigated_exp = mitigator.calculate_expectation_z(
                    result["mitigated_counts"]
                )
                print(f"   Noisy: {noisy_exp:.4f}")
                print(f"   Mitigated: {mitigated_exp:.4f}")

        print(f"\n📚 Key Concepts:")
        print(f"   • ZNE: Extrapolates to zero noise using noise scaling")
        print(f"   • REM: Corrects readout errors using calibration")
        print(f"   • SV: Detects errors using symmetry properties")
        print(f"   • Mitigation ≠ Correction: improves but doesn't eliminate errors")

        print(f"\n✅ Error mitigation analysis completed!")

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
