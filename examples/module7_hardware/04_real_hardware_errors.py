#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms
Example 4: Real Hardware Error Analysis

Implementation of hardware noise characterization and error analysis techniques.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import (
    QuantumCircuit,
    ClassicalRegister,
    transpile,
    QuantumRegister,
    ClassicalRegister,
)
from qiskit_aer import AerSimulator

# Handle different Qiskit Aer versions for noise models
try:
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
        readout_error,
    )
except ImportError:
    try:
        from qiskit_aer.noise import (
            NoiseModel,
            depolarizing_error,
            thermal_relaxation_error,
        )
        from qiskit_aer.noise import ReadoutError as readout_error
    except ImportError:
        try:
            from qiskit_aer.noise import (
                NoiseModel,
                depolarizing_error,
                thermal_relaxation_error,
            )

            # Create a simple readout error class for compatibility
            def readout_error(probabilities):
                from qiskit_aer.noise import ReadoutError

                return ReadoutError(probabilities)

        except ImportError:
            print("ℹ️  Noise models not fully available, using simplified simulation")
from qiskit.quantum_info import state_fidelity, process_fidelity, Statevector
from qiskit.result import marginal_counts
from qiskit.circuit.library import XGate, HGate, CXGate
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")


class HardwareNoiseCharacterizer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.noise_models = {}
        self.error_rates = {}

    def create_realistic_noise_model(self, backend_name="ibm_like"):
        """Create realistic noise model based on hardware characteristics."""
        noise_model = NoiseModel()

        if backend_name == "ibm_like":
            # IBM-like superconducting parameters
            gate_params = {
                "single_qubit_gate_time": 50e-9,  # 50 ns
                "two_qubit_gate_time": 300e-9,  # 300 ns
                "single_qubit_error_rate": 0.001,
                "two_qubit_error_rate": 0.01,
                "T1": 100e-6,  # 100 μs
                "T2": 70e-6,  # 70 μs
                "readout_error_rate": 0.02,
            }
        elif backend_name == "ionq_like":
            # IonQ-like trapped ion parameters
            gate_params = {
                "single_qubit_gate_time": 10e-6,  # 10 μs
                "two_qubit_gate_time": 200e-6,  # 200 μs
                "single_qubit_error_rate": 0.0005,
                "two_qubit_error_rate": 0.005,
                "T1": 10,  # 10 s
                "T2": 1,  # 1 s
                "readout_error_rate": 0.01,
            }
        elif backend_name == "rigetti_like":
            # Rigetti-like superconducting parameters
            gate_params = {
                "single_qubit_gate_time": 20e-9,  # 20 ns
                "two_qubit_gate_time": 150e-9,  # 150 ns
                "single_qubit_error_rate": 0.002,
                "two_qubit_error_rate": 0.015,
                "T1": 80e-6,  # 80 μs
                "T2": 50e-6,  # 50 μs
                "readout_error_rate": 0.025,
            }
        else:
            # Default parameters
            gate_params = {
                "single_qubit_gate_time": 50e-9,
                "two_qubit_gate_time": 300e-9,
                "single_qubit_error_rate": 0.001,
                "two_qubit_error_rate": 0.01,
                "T1": 100e-6,
                "T2": 70e-6,
                "readout_error_rate": 0.02,
            }

        # Add depolarizing errors
        single_qubit_error = depolarizing_error(
            gate_params["single_qubit_error_rate"], 1
        )
        two_qubit_error = depolarizing_error(gate_params["two_qubit_error_rate"], 2)

        # Add to single-qubit gates
        noise_model.add_all_qubit_quantum_error(
            single_qubit_error, ["x", "y", "z", "h", "sx", "rz", "ry"]
        )

        # Add to two-qubit gates
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx", "cz"])

        # Add thermal relaxation errors
        for qubit in range(10):  # Assume 10 qubits
            thermal_error_1q = thermal_relaxation_error(
                gate_params["T1"],
                gate_params["T2"],
                gate_params["single_qubit_gate_time"],
            )
            thermal_error_2q = thermal_relaxation_error(
                gate_params["T1"], gate_params["T2"], gate_params["two_qubit_gate_time"]
            ).expand(
                thermal_relaxation_error(
                    gate_params["T1"],
                    gate_params["T2"],
                    gate_params["two_qubit_gate_time"],
                )
            )

            noise_model.add_quantum_error(
                thermal_error_1q, ["x", "y", "z", "h", "sx", "rz", "ry"], [qubit]
            )

            # Add two-qubit thermal errors
            for other_qubit in range(qubit + 1, 10):
                noise_model.add_quantum_error(
                    thermal_error_2q, ["cx", "cz"], [qubit, other_qubit]
                )

        # Add readout errors
        readout_prob = gate_params["readout_error_rate"]
        readout_error_matrix = [
            [1 - readout_prob, readout_prob],
            [readout_prob, 1 - readout_prob],
        ]

        for qubit in range(10):
            noise_model.add_readout_error(readout_error(readout_error_matrix), [qubit])

        self.noise_models[backend_name] = {
            "noise_model": noise_model,
            "parameters": gate_params,
        }

        return noise_model, gate_params

    def measure_gate_fidelities(self, noise_model, n_trials=1000):
        """Measure single and two-qubit gate fidelities."""
        simulator = AerSimulator(noise_model=noise_model)
        fidelities = {}

        # Single-qubit gate fidelities
        single_qubit_gates = ["x", "y", "z", "h", "sx"]

        for gate_name in single_qubit_gates:
            if self.verbose:
                print(f"   Measuring {gate_name} gate fidelity...")

            fidelity_measurements = []

            for trial in range(min(n_trials, 100)):  # Limit for performance
                # Create circuit with random initial state
                qc = QuantumCircuit(1)

                # Random initial state preparation
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                qc.ry(theta, 0)
                qc.rz(phi, 0)

                # Apply gate
                if gate_name == "x":
                    qc.x(0)
                elif gate_name == "y":
                    qc.y(0)
                elif gate_name == "z":
                    qc.z(0)
                elif gate_name == "h":
                    qc.h(0)
                elif gate_name == "sx":
                    qc.sx(0)

                # Get ideal and noisy states
                ideal_state = Statevector.from_instruction(qc)

                # Run with noise
                job = simulator.run(qc, shots=1000)
                result = job.result()

                # Estimate state from measurements
                counts = result.get_counts()
                total_shots = sum(counts.values())

                if total_shots > 0:
                    prob_0 = counts.get("0", 0) / total_shots
                    prob_1 = counts.get("1", 0) / total_shots

                    # Reconstruct density matrix (simplified)
                    measured_probs = [prob_0, prob_1]
                    ideal_probs = np.abs(ideal_state.data) ** 2

                    # Fidelity approximation
                    fidelity = (
                        np.sqrt(np.sum(np.sqrt(measured_probs * ideal_probs))) ** 2
                    )
                    fidelity_measurements.append(fidelity)

            if fidelity_measurements:
                avg_fidelity = np.mean(fidelity_measurements)
                std_fidelity = np.std(fidelity_measurements)
                fidelities[gate_name] = {
                    "mean": avg_fidelity,
                    "std": std_fidelity,
                    "measurements": fidelity_measurements,
                }

        # Two-qubit gate fidelities
        two_qubit_gates = ["cx"]

        for gate_name in two_qubit_gates:
            if self.verbose:
                print(f"   Measuring {gate_name} gate fidelity...")

            fidelity_measurements = []

            for trial in range(min(n_trials, 50)):  # Fewer trials for two-qubit gates
                # Create circuit with random initial state
                qc = QuantumCircuit(2)

                # Random initial state preparation
                for qubit in range(2):
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(0, 2 * np.pi)
                    qc.ry(theta, qubit)
                    qc.rz(phi, qubit)

                # Apply gate
                if gate_name == "cx":
                    qc.cx(0, 1)

                qc.measure_all()

                # Run with and without noise for comparison
                job_noisy = simulator.run(qc, shots=1000)
                result_noisy = job_noisy.result()
                counts_noisy = result_noisy.get_counts()

                # Approximate fidelity from distribution similarity
                # This is a simplified approach
                total_shots = sum(counts_noisy.values())
                if total_shots > 0:
                    # Estimate fidelity based on expected vs measured distribution
                    # For demonstration purposes, use a simplified metric
                    entropy = 0
                    for count in counts_noisy.values():
                        if count > 0:
                            p = count / total_shots
                            entropy -= p * np.log2(p)

                    # Higher entropy suggests more noise
                    # Convert to rough fidelity estimate
                    max_entropy = 2  # For 2 qubits
                    fidelity = 1 - (entropy / max_entropy) * 0.5  # Rough approximation
                    fidelity_measurements.append(max(0, min(1, fidelity)))

            if fidelity_measurements:
                avg_fidelity = np.mean(fidelity_measurements)
                std_fidelity = np.std(fidelity_measurements)
                fidelities[gate_name] = {
                    "mean": avg_fidelity,
                    "std": std_fidelity,
                    "measurements": fidelity_measurements,
                }

        return fidelities

    def characterize_coherence_times(self, noise_model, max_time=500e-6, n_points=20):
        """Characterize T1 and T2 coherence times."""
        simulator = AerSimulator(noise_model=noise_model)

        time_points = np.linspace(0, max_time, n_points)

        # T1 measurement (amplitude damping)
        t1_data = []

        if self.verbose:
            print("   Measuring T1 (relaxation time)...")

        for t in time_points:
            # Create circuit for T1 measurement
            qc = QuantumCircuit(1, 1)
            qc.x(0)  # Start in |1⟩ state

            # Add delay (simulated with identity gates)
            n_delays = int(t / 50e-9)  # Assume 50ns per gate
            for _ in range(n_delays):
                qc.id(0)

            qc.measure(0, 0)

            # Run measurement
            job = simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Probability of staying in |1⟩
            prob_1 = counts.get("1", 0) / sum(counts.values())
            t1_data.append(prob_1)

        # T2 measurement (dephasing)
        t2_data = []

        if self.verbose:
            print("   Measuring T2 (dephasing time)...")

        for t in time_points:
            # Create Ramsey sequence
            qc = QuantumCircuit(1, 1)
            qc.h(0)  # Create superposition

            # Add delay
            n_delays = int(t / 50e-9)
            for _ in range(n_delays):
                qc.id(0)

            qc.h(0)  # Second pulse
            qc.measure(0, 0)

            # Run measurement
            job = simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Visibility of interference fringes
            prob_0 = counts.get("0", 0) / sum(counts.values())
            visibility = abs(prob_0 - 0.5) * 2  # Convert to visibility
            t2_data.append(visibility)

        # Fit exponential decay
        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        # Fit T1
        try:
            popt_t1, _ = curve_fit(
                exp_decay, time_points, t1_data, p0=[1.0, 100e-6, 0.0], maxfev=1000
            )
            t1_fitted = popt_t1[1]
        except:
            t1_fitted = None
            popt_t1 = None

        # Fit T2
        try:
            popt_t2, _ = curve_fit(
                exp_decay, time_points, t2_data, p0=[1.0, 50e-6, 0.0], maxfev=1000
            )
            t2_fitted = popt_t2[1]
        except:
            t2_fitted = None
            popt_t2 = None

        coherence_data = {
            "time_points": time_points,
            "t1_data": t1_data,
            "t2_data": t2_data,
            "t1_fitted": t1_fitted,
            "t2_fitted": t2_fitted,
            "t1_params": popt_t1,
            "t2_params": popt_t2,
        }

        return coherence_data

    def measure_readout_errors(self, noise_model, n_qubits=5):
        """Measure readout error rates for each qubit."""
        simulator = AerSimulator(noise_model=noise_model)
        readout_errors = {}

        for qubit in range(n_qubits):
            if self.verbose:
                print(f"   Measuring readout error for qubit {qubit}...")

            # Measure |0⟩ state
            qc_0 = QuantumCircuit(n_qubits, n_qubits)
            qc_0.measure(qubit, qubit)

            job_0 = simulator.run(qc_0, shots=1000)
            result_0 = job_0.result()
            counts_0 = result_0.get_counts()

            # Extract single qubit measurement
            total_0 = 0
            error_0_to_1 = 0

            for state, count in counts_0.items():
                if len(state) > qubit:
                    bit_value = state[-(qubit + 1)]  # Reverse indexing
                    total_0 += count
                    if bit_value == "1":
                        error_0_to_1 += count

            p_0_to_1 = error_0_to_1 / total_0 if total_0 > 0 else 0

            # Measure |1⟩ state
            qc_1 = QuantumCircuit(n_qubits, n_qubits)
            qc_1.x(qubit)
            qc_1.measure(qubit, qubit)

            job_1 = simulator.run(qc_1, shots=1000)
            result_1 = job_1.result()
            counts_1 = result_1.get_counts()

            # Extract single qubit measurement
            total_1 = 0
            error_1_to_0 = 0

            for state, count in counts_1.items():
                if len(state) > qubit:
                    bit_value = state[-(qubit + 1)]
                    total_1 += count
                    if bit_value == "0":
                        error_1_to_0 += count

            p_1_to_0 = error_1_to_0 / total_1 if total_1 > 0 else 0

            readout_errors[qubit] = {
                "p_0_to_1": p_0_to_1,
                "p_1_to_0": p_1_to_0,
                "average_error": (p_0_to_1 + p_1_to_0) / 2,
            }

        return readout_errors

    def analyze_circuit_noise_sensitivity(self, circuit, noise_models):
        """Analyze how different noise models affect circuit performance."""
        results = {}

        # Get ideal result first
        ideal_simulator = AerSimulator()
        ideal_job = ideal_simulator.run(circuit, shots=1000)
        ideal_result = ideal_job.result()
        ideal_counts = ideal_result.get_counts()

        for model_name, model_info in noise_models.items():
            if self.verbose:
                print(f"   Analyzing with {model_name} noise model...")

            noise_model = model_info["noise_model"]
            simulator = AerSimulator(noise_model=noise_model)

            # Run with noise
            noisy_job = simulator.run(circuit, shots=1000)
            noisy_result = noisy_job.result()
            noisy_counts = noisy_result.get_counts()

            # Calculate fidelity between ideal and noisy distributions
            fidelity = self.calculate_distribution_fidelity(ideal_counts, noisy_counts)

            # Calculate total variation distance
            tvd = self.calculate_tvd(ideal_counts, noisy_counts)

            # Calculate effective error rate
            effective_error = 1 - fidelity

            results[model_name] = {
                "ideal_counts": ideal_counts,
                "noisy_counts": noisy_counts,
                "fidelity": fidelity,
                "tvd": tvd,
                "effective_error": effective_error,
                "parameters": model_info["parameters"],
            }

        return results

    def calculate_distribution_fidelity(self, counts1, counts2):
        """Calculate fidelity between two probability distributions."""
        # Get all states
        all_states = set(counts1.keys()) | set(counts2.keys())

        # Normalize counts
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        if total1 == 0 or total2 == 0:
            return 0

        # Calculate fidelity
        fidelity = 0
        for state in all_states:
            p1 = counts1.get(state, 0) / total1
            p2 = counts2.get(state, 0) / total2
            fidelity += np.sqrt(p1 * p2)

        return fidelity**2

    def calculate_tvd(self, counts1, counts2):
        """Calculate total variation distance between two distributions."""
        all_states = set(counts1.keys()) | set(counts2.keys())

        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        if total1 == 0 or total2 == 0:
            return 1

        tvd = 0
        for state in all_states:
            p1 = counts1.get(state, 0) / total1
            p2 = counts2.get(state, 0) / total2
            tvd += abs(p1 - p2)

        return tvd / 2


class ErrorMitigationAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def zero_noise_extrapolation(self, circuit, noise_levels, shots=1000):
        """Implement zero-noise extrapolation error mitigation."""
        results = []

        for noise_scale in noise_levels:
            # Scale noise by repeating circuit elements
            scaled_circuit = self.scale_circuit_noise(circuit, noise_scale)

            # Run simulation
            simulator = AerSimulator()
            job = simulator.run(scaled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Calculate expectation value (simplified for demonstration)
            expectation = self.calculate_expectation_value(counts)
            results.append(expectation)

        # Extrapolate to zero noise
        if len(results) >= 2:
            # Linear extrapolation
            x = np.array(noise_levels)
            y = np.array(results)

            # Fit line and extrapolate to x=0
            coeffs = np.polyfit(x, y, 1)
            zero_noise_value = coeffs[1]  # y-intercept
        else:
            zero_noise_value = results[0] if results else 0

        return {
            "noise_levels": noise_levels,
            "measured_values": results,
            "extrapolated_value": zero_noise_value,
        }

    def scale_circuit_noise(self, circuit, scale_factor):
        """Scale circuit noise by unitary folding."""
        if scale_factor <= 1:
            return circuit

        # For simplicity, add identity gates to increase noise
        scaled_circuit = circuit.copy()
        n_extra_gates = int((scale_factor - 1) * circuit.depth())

        for _ in range(n_extra_gates):
            qubit = np.random.randint(circuit.num_qubits)
            scaled_circuit.id(qubit)

        return scaled_circuit

    def calculate_expectation_value(self, counts):
        """Calculate expectation value from measurement counts."""
        total = sum(counts.values())
        if total == 0:
            return 0

        # Simple parity expectation
        expectation = 0
        for state, count in counts.items():
            parity = sum(int(bit) for bit in state) % 2
            sign = 1 if parity == 0 else -1
            expectation += sign * count / total

        return expectation

    def readout_error_mitigation(self, counts, readout_matrix):
        """Apply readout error mitigation."""
        # This is a simplified version
        # In practice, would invert the full readout calibration matrix

        mitigated_counts = {}
        total = sum(counts.values())

        for state, count in counts.items():
            # Apply inverse of readout error
            # Simplified for demonstration
            mitigated_counts[state] = count * 1.05  # Rough correction

        return mitigated_counts


def visualize_error_analysis(
    characterizer, gate_fidelities, coherence_data, readout_errors, noise_analysis
):
    """Visualize hardware error analysis results."""
    fig = plt.figure(figsize=(16, 12))

    # Gate fidelities
    ax1 = plt.subplot(2, 3, 1)

    gates = []
    fidelities = []
    errors = []

    for gate, data in gate_fidelities.items():
        gates.append(gate.upper())
        fidelities.append(data["mean"])
        errors.append(data["std"])

    if gates:
        bars = ax1.bar(
            gates, fidelities, yerr=errors, capsize=5, alpha=0.7, color="skyblue"
        )
        ax1.set_ylabel("Gate Fidelity")
        ax1.set_title("Gate Fidelity Measurements")
        ax1.set_ylim(0, 1)

        # Add value labels
        for bar, fidelity in zip(bars, fidelities):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{fidelity:.3f}",
                ha="center",
                va="bottom",
            )

    # Coherence times
    ax2 = plt.subplot(2, 3, 2)

    if coherence_data["t1_fitted"] and coherence_data["t2_fitted"]:
        times = ["T1", "T2"]
        values = [
            coherence_data["t1_fitted"] * 1e6,
            coherence_data["t2_fitted"] * 1e6,
        ]  # Convert to μs

        bars = ax2.bar(times, values, alpha=0.7, color=["orange", "green"])
        ax2.set_ylabel("Coherence Time (μs)")
        ax2.set_title("Coherence Time Measurements")

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

    # Coherence decay curves
    ax3 = plt.subplot(2, 3, 3)

    time_us = coherence_data["time_points"] * 1e6  # Convert to μs

    ax3.plot(
        time_us, coherence_data["t1_data"], "o-", label="T1 (Relaxation)", alpha=0.7
    )
    ax3.plot(
        time_us, coherence_data["t2_data"], "s-", label="T2 (Dephasing)", alpha=0.7
    )

    # Add fitted curves if available
    if coherence_data["t1_params"] is not None:
        t1_fit = (
            coherence_data["t1_params"][0]
            * np.exp(-coherence_data["time_points"] / coherence_data["t1_params"][1])
            + coherence_data["t1_params"][2]
        )
        ax3.plot(time_us, t1_fit, "--", color="orange", alpha=0.8)

    if coherence_data["t2_params"] is not None:
        t2_fit = (
            coherence_data["t2_params"][0]
            * np.exp(-coherence_data["time_points"] / coherence_data["t2_params"][1])
            + coherence_data["t2_params"][2]
        )
        ax3.plot(time_us, t2_fit, "--", color="green", alpha=0.8)

    ax3.set_xlabel("Time (μs)")
    ax3.set_ylabel("Signal")
    ax3.set_title("Coherence Decay Curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Readout errors
    ax4 = plt.subplot(2, 3, 4)

    qubits = list(readout_errors.keys())
    error_rates = [readout_errors[q]["average_error"] for q in qubits]

    if qubits:
        bars = ax4.bar([f"Q{q}" for q in qubits], error_rates, alpha=0.7, color="red")
        ax4.set_ylabel("Readout Error Rate")
        ax4.set_title("Readout Error by Qubit")

        # Add value labels
        for bar, error in zip(bars, error_rates):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{error:.3f}",
                ha="center",
                va="bottom",
            )

    # Noise model comparison
    ax5 = plt.subplot(2, 3, 5)

    models = list(noise_analysis.keys())
    fidelities = [noise_analysis[model]["fidelity"] for model in models]

    if models:
        bars = ax5.bar(models, fidelities, alpha=0.7, color="purple")
        ax5.set_ylabel("Circuit Fidelity")
        ax5.set_title("Noise Model Comparison")
        ax5.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, fidelity in zip(bars, fidelities):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{fidelity:.3f}",
                ha="center",
                va="bottom",
            )

    # Summary and insights
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = "Hardware Error Analysis Summary:\n\n"

    if gate_fidelities:
        best_gate = max(gate_fidelities.items(), key=lambda x: x[1]["mean"])
        worst_gate = min(gate_fidelities.items(), key=lambda x: x[1]["mean"])

        summary_text += (
            f"Best Gate: {best_gate[0].upper()} ({best_gate[1]['mean']:.3f})\n"
        )
        summary_text += (
            f"Worst Gate: {worst_gate[0].upper()} ({worst_gate[1]['mean']:.3f})\n\n"
        )

    if coherence_data["t1_fitted"] and coherence_data["t2_fitted"]:
        summary_text += f"T1: {coherence_data['t1_fitted']*1e6:.1f} μs\n"
        summary_text += f"T2: {coherence_data['t2_fitted']*1e6:.1f} μs\n\n"

    if readout_errors:
        avg_readout_error = np.mean(
            [e["average_error"] for e in readout_errors.values()]
        )
        summary_text += f"Avg Readout Error: {avg_readout_error:.3f}\n\n"

    summary_text += "Error Sources:\n\n"
    summary_text += "Gate Errors:\n"
    summary_text += "• Calibration drift\n"
    summary_text += "• Control electronics noise\n"
    summary_text += "• Crosstalk between qubits\n\n"

    summary_text += "Coherence Errors:\n"
    summary_text += "• Energy relaxation (T1)\n"
    summary_text += "• Pure dephasing (T2*)\n"
    summary_text += "• Environmental coupling\n\n"

    summary_text += "Mitigation Strategies:\n"
    summary_text += "• Dynamical decoupling\n"
    summary_text += "• Error correction codes\n"
    summary_text += "• Zero-noise extrapolation\n"
    summary_text += "• Readout error mitigation"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.7),
    )

    plt.tight_layout()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Real Hardware Error Analysis")
    parser.add_argument(
        "--backend-type",
        choices=["ibm_like", "ionq_like", "rigetti_like"],
        default="ibm_like",
        help="Hardware backend type to simulate",
    )
    parser.add_argument(
        "--circuit",
        choices=["bell", "ghz", "qft", "random"],
        default="bell",
        help="Test circuit type",
    )
    parser.add_argument(
        "--measure-fidelities", action="store_true", help="Measure gate fidelities"
    )
    parser.add_argument(
        "--measure-coherence", action="store_true", help="Measure coherence times"
    )
    parser.add_argument(
        "--measure-readout", action="store_true", help="Measure readout errors"
    )
    parser.add_argument(
        "--compare-noise-models",
        action="store_true",
        help="Compare different noise models",
    )
    parser.add_argument(
        "--error-mitigation",
        action="store_true",
        help="Test error mitigation techniques",
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms")
    print("Example 4: Real Hardware Error Analysis")
    print("=" * 51)

    try:
        # Initialize characterizer
        characterizer = HardwareNoiseCharacterizer(verbose=args.verbose)

        # Create noise model
        print(f"\n🔧 Creating {args.backend_type} noise model...")
        noise_model, params = characterizer.create_realistic_noise_model(
            args.backend_type
        )

        print(f"   Hardware parameters:")
        for param, value in params.items():
            if "time" in param.lower():
                if value < 1e-3:
                    print(f"     {param}: {value*1e6:.1f} μs")
                else:
                    print(f"     {param}: {value:.3f} s")
            elif "rate" in param.lower():
                print(f"     {param}: {value:.4f}")
            else:
                print(f"     {param}: {value}")

        # Create test circuit
        if args.circuit == "bell":
            test_circuit = QuantumCircuit(2, 2)
            test_circuit.h(0)
            test_circuit.cx(0, 1)
            test_circuit.measure_all()
        elif args.circuit == "ghz":
            test_circuit = QuantumCircuit(3, 3)
            test_circuit.h(0)
            test_circuit.cx(0, 1)
            test_circuit.cx(0, 2)
            test_circuit.measure_all()
        elif args.circuit == "qft":
            from qiskit.circuit.library import QFT

            qft = QFT(3)
            test_circuit = qft.decompose()
            test_circuit.measure_all()
        else:  # random
            test_circuit = QuantumCircuit(3, 3)
            for _ in range(10):
                gate = np.random.choice(["h", "x", "y", "z", "rx", "ry", "rz", "cx"])
                if gate in ["h", "x", "y", "z"]:
                    qubit = np.random.randint(3)
                    getattr(test_circuit, gate)(qubit)
                elif gate in ["rx", "ry", "rz"]:
                    qubit = np.random.randint(3)
                    angle = np.random.uniform(0, 2 * np.pi)
                    getattr(test_circuit, gate)(angle, qubit)
                elif gate == "cx":
                    control = np.random.randint(3)
                    target = np.random.randint(3)
                    if control != target:
                        test_circuit.cx(control, target)
            test_circuit.measure_all()

        print(f"\n🎯 Test circuit: {args.circuit}")
        print(f"   Qubits: {test_circuit.num_qubits}")
        print(f"   Depth: {test_circuit.depth()}")
        print(f"   Gates: {sum(dict(test_circuit.count_ops()).values())}")

        # Gate fidelity measurements
        gate_fidelities = {}
        if args.measure_fidelities:
            print(f"\n📊 Measuring gate fidelities...")
            gate_fidelities = characterizer.measure_gate_fidelities(
                noise_model, n_trials=50
            )

            print(f"   Results:")
            for gate, data in gate_fidelities.items():
                print(f"     {gate.upper()}: {data['mean']:.3f} ± {data['std']:.3f}")

        # Coherence measurements
        coherence_data = {}
        if args.measure_coherence:
            print(f"\n⏱️  Measuring coherence times...")
            coherence_data = characterizer.characterize_coherence_times(noise_model)

            print(f"   Results:")
            if coherence_data["t1_fitted"]:
                print(f"     T1: {coherence_data['t1_fitted']*1e6:.1f} μs")
            if coherence_data["t2_fitted"]:
                print(f"     T2: {coherence_data['t2_fitted']*1e6:.1f} μs")

        # Readout error measurements
        readout_errors = {}
        if args.measure_readout:
            print(f"\n📖 Measuring readout errors...")
            readout_errors = characterizer.measure_readout_errors(
                noise_model, n_qubits=3
            )

            print(f"   Results:")
            for qubit, errors in readout_errors.items():
                print(f"     Qubit {qubit}: {errors['average_error']:.3f}")
                print(f"       P(0→1): {errors['p_0_to_1']:.3f}")
                print(f"       P(1→0): {errors['p_1_to_0']:.3f}")

        # Noise model comparison
        noise_analysis = {}
        if args.compare_noise_models:
            print(f"\n🔄 Comparing noise models...")

            # Create multiple noise models
            noise_models = {}
            for backend_type in ["ibm_like", "ionq_like", "rigetti_like"]:
                nm, params = characterizer.create_realistic_noise_model(backend_type)
                noise_models[backend_type] = {"noise_model": nm, "parameters": params}

            noise_analysis = characterizer.analyze_circuit_noise_sensitivity(
                test_circuit, noise_models
            )

            print(f"   Results:")
            for model_name, results in noise_analysis.items():
                print(f"     {model_name}:")
                print(f"       Fidelity: {results['fidelity']:.3f}")
                print(f"       TVD: {results['tvd']:.3f}")
                print(f"       Effective error: {results['effective_error']:.3f}")

        # Error mitigation
        if args.error_mitigation:
            print(f"\n🛡️  Testing error mitigation...")

            mitigator = ErrorMitigationAnalyzer(verbose=args.verbose)

            # Zero-noise extrapolation
            noise_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
            zne_result = mitigator.zero_noise_extrapolation(test_circuit, noise_levels)

            print(f"   Zero-noise extrapolation:")
            print(f"     Extrapolated value: {zne_result['extrapolated_value']:.3f}")
            print(
                f"     Measured values: {[f'{v:.3f}' for v in zne_result['measured_values']]}"
            )

        # Run complete analysis if no specific measurements requested
        if not any(
            [
                args.measure_fidelities,
                args.measure_coherence,
                args.measure_readout,
                args.compare_noise_models,
            ]
        ):
            print(f"\n🔬 Running complete error analysis...")

            # Quick measurements
            gate_fidelities = characterizer.measure_gate_fidelities(
                noise_model, n_trials=20
            )
            coherence_data = characterizer.characterize_coherence_times(
                noise_model, n_points=10
            )
            readout_errors = characterizer.measure_readout_errors(
                noise_model, n_qubits=3
            )

            # Quick noise comparison
            noise_models = {
                args.backend_type: {"noise_model": noise_model, "parameters": params}
            }
            noise_analysis = characterizer.analyze_circuit_noise_sensitivity(
                test_circuit, noise_models
            )

            print(f"\n📊 Summary Results:")
            print(f"   Gate fidelities: {len(gate_fidelities)} measured")
            print(
                f"   Coherence times: T1={coherence_data.get('t1_fitted', 0)*1e6:.1f}μs, T2={coherence_data.get('t2_fitted', 0)*1e6:.1f}μs"
            )
            print(f"   Readout errors: {len(readout_errors)} qubits")
            print(
                f"   Circuit fidelity: {list(noise_analysis.values())[0]['fidelity']:.3f}"
            )

        # Visualization
        if args.show_visualization and any(
            [gate_fidelities, coherence_data, readout_errors, noise_analysis]
        ):
            visualize_error_analysis(
                characterizer,
                gate_fidelities,
                coherence_data,
                readout_errors,
                noise_analysis,
            )

        print(f"\n📚 Key Insights:")
        print(f"   • Hardware noise significantly impacts quantum computation")
        print(f"   • Different platforms have distinct error characteristics")
        print(f"   • Gate fidelities vary significantly between gate types")
        print(f"   • Coherence times limit maximum circuit depth")
        print(f"   • Readout errors affect measurement accuracy")

        print(f"\n🎯 Error Mitigation Strategies:")
        print(f"   • Use error correction codes for fault-tolerant computation")
        print(f"   • Apply dynamical decoupling to extend coherence")
        print(f"   • Implement zero-noise extrapolation for NISQ algorithms")
        print(f"   • Calibrate and correct readout errors")
        print(f"   • Optimize circuits for hardware constraints")

        print(f"\n✅ Hardware error analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
