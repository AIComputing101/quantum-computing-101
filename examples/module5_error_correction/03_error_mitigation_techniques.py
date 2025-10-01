#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 5: Error Correction
Example 3: Error Mitigation Techniques

Implementation of various quantum error mitigation techniques.
"""

import numpy as np
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
        """Zero noise extrapolation technique."""
        results = []

        for factor in noise_factors:
            # Scale noise by repeating circuit elements
            scaled_circuit = self.scale_circuit_noise(circuit, factor)

            # Simulate with noise
            noise_model = self.create_noise_model()
            simulator = AerSimulator(noise_model=noise_model)

            job = simulator.run(scaled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Calculate expectation value (assuming measurement of Z on first qubit)
            expectation = self.calculate_expectation_z(counts)
            results.append((factor, expectation))

        # Extrapolate to zero noise
        factors = [r[0] for r in results]
        expectations = [r[1] for r in results]

        # Linear extrapolation
        coeffs = np.polyfit(factors, expectations, 1)
        zero_noise_estimate = coeffs[1]  # y-intercept

        return {
            "measurements": results,
            "zero_noise_estimate": zero_noise_estimate,
            "extrapolation_coeffs": coeffs,
        }

    def scale_circuit_noise(self, circuit, factor):
        """Scale circuit noise by repetition."""
        if factor == 1:
            return circuit.copy()

        # Create new circuit with repeated operations
        scaled_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

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
        """Readout error mitigation using calibration."""
        # Calibration: measure |0⟩ and |1⟩ states
        cal_circuits = []

        # Calibration circuit for |0⟩
        cal_0 = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        cal_0.measure_all()
        cal_circuits.append(cal_0)

        # Calibration circuit for |1⟩
        cal_1 = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        for i in range(circuit.num_qubits):
            cal_1.x(i)
        cal_1.measure_all()
        cal_circuits.append(cal_1)

        # Run calibration
        noise_model = self.create_noise_model()
        simulator = AerSimulator(noise_model=noise_model)

        cal_results = []
        for cal_circuit in cal_circuits:
            job = simulator.run(cal_circuit, shots=shots)
            result = job.result()
            cal_results.append(result.get_counts())

        # Build calibration matrix
        cal_matrix = self.build_calibration_matrix(cal_results, circuit.num_qubits)

        # Run main circuit with noise
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        noisy_counts = result.get_counts()

        # Apply readout error mitigation
        mitigated_counts = self.apply_readout_mitigation(noisy_counts, cal_matrix)

        return {
            "noisy_counts": noisy_counts,
            "mitigated_counts": mitigated_counts,
            "calibration_matrix": cal_matrix,
        }

    def build_calibration_matrix(self, cal_results, n_qubits):
        """Build calibration matrix from calibration results."""
        n_states = 2**n_qubits
        cal_matrix = np.zeros((n_states, n_states))

        for i, counts in enumerate(cal_results):
            total_shots = sum(counts.values())
            for state, count in counts.items():
                state_int = int(state, 2)
                cal_matrix[state_int, i] = count / total_shots

        return cal_matrix

    def apply_readout_mitigation(self, counts, cal_matrix):
        """Apply readout error mitigation."""
        n_states = cal_matrix.shape[0]
        total_shots = sum(counts.values())

        # Convert counts to probability vector
        prob_vector = np.zeros(n_states)
        for state, count in counts.items():
            state_int = int(state, 2)
            prob_vector[state_int] = count / total_shots

        # Invert calibration matrix and apply
        try:
            inv_cal_matrix = np.linalg.inv(cal_matrix)
            mitigated_probs = inv_cal_matrix @ prob_vector

            # Convert back to counts
            mitigated_counts = {}
            for i, prob in enumerate(mitigated_probs):
                if prob > 0:  # Only include positive probabilities
                    state_str = format(i, f"0{int(np.log2(n_states))}b")
                    mitigated_counts[state_str] = int(prob * total_shots)

            return mitigated_counts

        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
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
            # Z eigenvalue is +1 for |0⟩, -1 for |1⟩ on first qubit
            if state[0] == "0":
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
        # Create test circuit
        test_circuit = QuantumCircuit(2, 2)
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
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
