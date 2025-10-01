#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 4: Quantum Algorithms
Example 3: Quantum Fourier Transform

Implementation and analysis of the Quantum Fourier Transform algorithm.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import warnings

warnings.filterwarnings("ignore")


class QuantumFourierTransform:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def build_qft_circuit(self, n_qubits, inverse=False):
        """Build QFT circuit."""
        qft = QuantumCircuit(n_qubits, name=f'{"IQFT" if inverse else "QFT"}')

        for i in range(n_qubits):
            qft.h(i)
            for j in range(i + 1, n_qubits):
                angle = np.pi / (2 ** (j - i))
                if inverse:
                    angle *= -1
                qft.cp(angle, j, i)

        # Swap qubits for correct ordering
        for i in range(n_qubits // 2):
            qft.swap(i, n_qubits - 1 - i)

        return qft

    def analyze_qft(self, n_qubits, input_state=None):
        """Analyze QFT transformation."""
        circuit = QuantumCircuit(n_qubits)

        # Prepare input state
        if input_state is not None:
            circuit.initialize(input_state, range(n_qubits))
        else:
            # Default: |001...⟩ state
            circuit.x(0)

        # Get input state
        input_sv = Statevector.from_instruction(circuit)

        # Apply QFT
        qft = self.build_qft_circuit(n_qubits)
        circuit.compose(qft, inplace=True)

        # Get output state
        output_sv = Statevector.from_instruction(circuit)

        return {
            "input_state": input_sv.data,
            "output_state": output_sv.data,
            "circuit": circuit,
            "qft_circuit": qft,
        }

    def demonstrate_period_finding(self, n_qubits, period):
        """Demonstrate QFT in period finding."""
        # Create periodic function
        circuit = QuantumCircuit(n_qubits, n_qubits)

        # Initialize superposition
        for i in range(n_qubits):
            circuit.h(i)

        # Simulate periodic function (simplified)
        for i in range(0, n_qubits, period):
            if i < n_qubits:
                circuit.z(i)

        # Apply inverse QFT
        iqft = self.build_qft_circuit(n_qubits, inverse=True)
        circuit.compose(iqft, inplace=True)

        circuit.measure_all()

        # Execute
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        return {"circuit": circuit, "counts": counts, "period": period}

    def visualize_qft_results(self, analysis_results):
        """Visualize QFT analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        input_state = analysis_results["input_state"]
        output_state = analysis_results["output_state"]

        # Input state amplitudes
        ax1.bar(range(len(input_state)), np.abs(input_state), alpha=0.7, color="blue")
        ax1.set_title("Input State Amplitudes")
        ax1.set_xlabel("Basis State")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        # Output state amplitudes
        ax2.bar(range(len(output_state)), np.abs(output_state), alpha=0.7, color="red")
        ax2.set_title("QFT Output Amplitudes")
        ax2.set_xlabel("Basis State")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)

        # Phase comparison
        input_phases = np.angle(input_state)
        output_phases = np.angle(output_state)

        ax3.scatter(
            range(len(input_phases)),
            input_phases,
            alpha=0.7,
            label="Input",
            color="blue",
        )
        ax3.scatter(
            range(len(output_phases)),
            output_phases,
            alpha=0.7,
            label="Output",
            color="red",
        )
        ax3.set_title("Phase Evolution")
        ax3.set_xlabel("Basis State")
        ax3.set_ylabel("Phase (radians)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Circuit properties
        circuit = analysis_results["qft_circuit"]
        metrics = ["Depth", "Gates", "CX Gates"]
        values = [circuit.depth(), circuit.size(), circuit.count_ops().get("cp", 0)]

        ax4.bar(metrics, values, alpha=0.7, color="green")
        ax4.set_title("QFT Circuit Metrics")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum Fourier Transform Analysis")
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits")
    parser.add_argument(
        "--period", type=int, default=2, help="Period for period finding demo"
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 4: Quantum Algorithms")
    print("Example 3: Quantum Fourier Transform")
    print("=" * 42)

    qft_analyzer = QuantumFourierTransform(verbose=args.verbose)

    try:
        # Analyze QFT
        print(f"\nAnalyzing {args.qubits}-qubit QFT...")
        analysis = qft_analyzer.analyze_qft(args.qubits)

        print(f"Input state norm: {np.linalg.norm(analysis['input_state']):.6f}")
        print(f"Output state norm: {np.linalg.norm(analysis['output_state']):.6f}")
        print(f"QFT circuit depth: {analysis['qft_circuit'].depth()}")
        print(f"QFT gate count: {analysis['qft_circuit'].size()}")

        # Period finding demonstration
        print(f"\nPeriod finding with period {args.period}...")
        period_result = qft_analyzer.demonstrate_period_finding(
            args.qubits, args.period
        )

        # Find peaks in measurement results
        counts = period_result["counts"]
        max_count = max(counts.values())
        peaks = [k for k, v in counts.items() if v > max_count * 0.5]

        print(f"Detected peaks: {peaks}")
        print(
            f"Expected period signature: multiples of {2**args.qubits // args.period}"
        )

        if args.show_visualization:
            qft_analyzer.visualize_qft_results(analysis)

        print(f"\n✅ QFT analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
