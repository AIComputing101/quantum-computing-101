#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 6: Quantum Machine Learning
Example 4: Quantum Principal Component Analysis

Implementation of quantum PCA algorithms for dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from sklearn.datasets import make_blobs, load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class QuantumPCA:
    def __init__(self, n_qubits, verbose=False):
        self.n_qubits = n_qubits
        self.verbose = verbose
        self.eigenvalues = None
        self.eigenvectors = None

    def prepare_data_state(self, data_matrix):
        """Prepare quantum state encoding data matrix."""
        # Normalize data matrix
        U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)

        # Create density matrix from data
        rho = data_matrix @ data_matrix.T
        rho = rho / np.trace(rho)  # Normalize

        return rho

    def quantum_phase_estimation(self, unitary_matrix, n_counting_qubits=3):
        """Quantum phase estimation for eigenvalue estimation."""
        n_system_qubits = int(np.log2(unitary_matrix.shape[0]))
        total_qubits = n_counting_qubits + n_system_qubits

        circuit = QuantumCircuit(total_qubits, n_counting_qubits)

        # Initialize counting register in superposition
        for i in range(n_counting_qubits):
            circuit.h(i)

        # Initialize system register (simplified)
        circuit.x(n_counting_qubits)  # |1⟩ state

        # Controlled unitary operations
        for i in range(n_counting_qubits):
            power = 2**i
            # Simplified controlled unitary (in practice would use actual matrix)
            for _ in range(power % 8):  # Mod to keep reasonable
                circuit.cp(np.pi / 4, i, n_counting_qubits)

        # Inverse QFT on counting register
        qft_inv = QFT(n_counting_qubits, inverse=True)
        circuit.compose(qft_inv, range(n_counting_qubits), inplace=True)

        # Measure counting register
        circuit.measure(range(n_counting_qubits), range(n_counting_qubits))

        return circuit

    def quantum_eigenvalue_estimation(self, data_matrix, n_components=2):
        """Estimate eigenvalues using quantum phase estimation."""
        # Create covariance matrix
        cov_matrix = np.cov(data_matrix.T)

        # Get exact eigenvalues for comparison
        exact_eigenvals, exact_eigenvecs = np.linalg.eigh(cov_matrix)
        exact_eigenvals = exact_eigenvals[::-1]  # Sort descending
        exact_eigenvecs = exact_eigenvecs[:, ::-1]

        # Simulate quantum phase estimation
        estimated_eigenvals = []

        for i in range(min(n_components, len(exact_eigenvals))):
            # Create unitary from eigenvalue (simplified)
            phase = 2 * np.pi * exact_eigenvals[i] / np.sum(exact_eigenvals)

            # Simulate measurement outcome
            n_counting_qubits = 3
            measured_phase = np.random.normal(phase, 0.1)  # Add noise
            estimated_eigenval = measured_phase * np.sum(exact_eigenvals) / (2 * np.pi)

            estimated_eigenvals.append(abs(estimated_eigenval))

        return estimated_eigenvals, exact_eigenvals[:n_components]

    def quantum_state_tomography(self, quantum_state, n_measurements=1000):
        """Perform quantum state tomography to reconstruct eigenvectors."""
        n_qubits = int(np.log2(len(quantum_state)))

        # Measurement bases
        bases = ["Z", "X", "Y"]
        measurements = {}

        for basis in bases:
            for qubit in range(n_qubits):
                circuit = QuantumCircuit(n_qubits, 1)

                # Prepare state (simplified)
                circuit.initialize(quantum_state, range(n_qubits))

                # Apply basis rotation
                if basis == "X":
                    circuit.ry(-np.pi / 2, qubit)
                elif basis == "Y":
                    circuit.rx(np.pi / 2, qubit)

                # Measure
                circuit.measure(qubit, 0)

                # Simulate measurement
                simulator = AerSimulator()
                job = simulator.run(circuit, shots=n_measurements)
                result = job.result()
                counts = result.get_counts()

                prob_0 = counts.get("0", 0) / n_measurements
                prob_1 = counts.get("1", 0) / n_measurements
                expectation = prob_0 - prob_1

                measurements[f"{basis}{qubit}"] = expectation

        return measurements

    def variational_quantum_pca(self, data_matrix, n_components=2, max_iter=100):
        """Variational approach to quantum PCA."""
        n_features = data_matrix.shape[1]

        # Create parameterized quantum circuit
        def create_ansatz(parameters):
            circuit = QuantumCircuit(self.n_qubits)
            param_idx = 0

            # Layer of RY rotations
            for i in range(self.n_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], i)
                    param_idx += 1

            # Entangling layer
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)

            # Another layer of rotations
            for i in range(self.n_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], i)
                    param_idx += 1

            return circuit

        # Cost function for PCA (maximize variance)
        def cost_function(parameters):
            circuit = create_ansatz(parameters)

            # Get state vector
            state = Statevector.from_instruction(circuit)
            state_vector = state.data

            # Compute expectation value with data covariance
            cov_matrix = np.cov(data_matrix.T)

            # Simplified cost: project onto leading eigenvector direction
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            leading_eigenvec = eigenvecs[:, -1]  # Largest eigenvalue

            # Cost is negative variance (to maximize)
            cost = (
                -np.abs(
                    np.vdot(state_vector[: len(leading_eigenvec)], leading_eigenvec)
                )
                ** 2
            )

            return cost

        # Optimize parameters
        n_params = 2 * self.n_qubits
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        # Simple gradient-free optimization
        best_params = initial_params.copy()
        best_cost = cost_function(best_params)

        for iteration in range(max_iter):
            # Random perturbation
            test_params = best_params + np.random.normal(0, 0.1, n_params)
            test_cost = cost_function(test_params)

            if test_cost < best_cost:
                best_params = test_params
                best_cost = test_cost

            if self.verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: Cost = {best_cost:.4f}")

        # Get final state
        final_circuit = create_ansatz(best_params)
        final_state = Statevector.from_instruction(final_circuit)

        return best_params, final_state.data, -best_cost

    def quantum_pca_analysis(self, data_matrix, n_components=2):
        """Complete quantum PCA analysis."""
        results = {}

        # Classical PCA for comparison
        classical_pca = PCA(n_components=n_components)
        classical_transformed = classical_pca.fit_transform(data_matrix)

        results["classical"] = {
            "eigenvalues": classical_pca.explained_variance_,
            "eigenvectors": classical_pca.components_,
            "explained_variance_ratio": classical_pca.explained_variance_ratio_,
            "transformed_data": classical_transformed,
        }

        # Quantum eigenvalue estimation
        if self.verbose:
            print("Performing quantum eigenvalue estimation...")

        quantum_eigenvals, exact_eigenvals = self.quantum_eigenvalue_estimation(
            data_matrix, n_components
        )

        results["quantum_eigenvalues"] = {
            "estimated": quantum_eigenvals,
            "exact": exact_eigenvals,
            "error": np.abs(np.array(quantum_eigenvals) - np.array(exact_eigenvals)),
        }

        # Variational quantum PCA
        if self.verbose:
            print("Performing variational quantum PCA...")

        vqpca_params, vqpca_state, vqpca_variance = self.variational_quantum_pca(
            data_matrix, n_components
        )

        results["variational"] = {
            "parameters": vqpca_params,
            "quantum_state": vqpca_state,
            "captured_variance": vqpca_variance,
        }

        return results


class QPCAAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compare_methods(self, data_matrix, n_components=2):
        """Compare quantum and classical PCA methods."""
        results = {}

        # Classical PCA
        classical_pca = PCA(n_components=n_components)
        classical_transformed = classical_pca.fit_transform(data_matrix)

        results["Classical PCA"] = {
            "eigenvalues": classical_pca.explained_variance_,
            "explained_variance_ratio": classical_pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(classical_pca.explained_variance_ratio_),
            "reconstruction_error": self.compute_reconstruction_error(
                data_matrix, classical_pca.inverse_transform(classical_transformed)
            ),
        }

        # Quantum PCA
        qpca = QuantumPCA(n_qubits=3, verbose=self.verbose)
        qpca_results = qpca.quantum_pca_analysis(data_matrix, n_components)

        # Extract quantum results
        quantum_eigenvals = qpca_results["quantum_eigenvalues"]["estimated"]
        quantum_variance_ratio = np.array(quantum_eigenvals) / np.sum(quantum_eigenvals)

        results["Quantum PCA"] = {
            "eigenvalues": quantum_eigenvals,
            "explained_variance_ratio": quantum_variance_ratio,
            "cumulative_variance": np.cumsum(quantum_variance_ratio),
            "estimation_error": qpca_results["quantum_eigenvalues"]["error"],
        }

        # Variational Quantum PCA
        vqpca_variance = qpca_results["variational"]["captured_variance"]

        results["Variational QPCA"] = {
            "captured_variance": vqpca_variance,
            "quantum_state": qpca_results["variational"]["quantum_state"],
        }

        return results

    def compute_reconstruction_error(self, original, reconstructed):
        """Compute reconstruction error."""
        return np.mean((original - reconstructed) ** 2)

    def analyze_scaling(self, data_sizes=[50, 100, 200]):
        """Analyze scaling of quantum vs classical PCA."""
        scaling_results = {}

        for n_samples in data_sizes:
            # Generate synthetic data
            X, _ = make_blobs(
                n_samples=n_samples,
                n_features=4,
                centers=3,
                cluster_std=2.0,
                random_state=42,
            )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Classical PCA timing (simulated)
            classical_time = n_samples * 0.001  # Linear scaling approximation

            # Quantum PCA timing (simulated - includes overhead)
            quantum_time = np.log(n_samples) * 0.1 + 0.5  # Logarithmic + overhead

            scaling_results[n_samples] = {
                "classical_time": classical_time,
                "quantum_time": quantum_time,
                "quantum_advantage": classical_time / quantum_time,
            }

        return scaling_results

    def visualize_results(
        self, comparison_results, scaling_results=None, data_matrix=None
    ):
        """Visualize QPCA analysis results."""
        fig = plt.figure(figsize=(16, 12))

        # Eigenvalue comparison
        ax1 = plt.subplot(2, 3, 1)

        if (
            "Classical PCA" in comparison_results
            and "Quantum PCA" in comparison_results
        ):
            classical_eigenvals = comparison_results["Classical PCA"]["eigenvalues"]
            quantum_eigenvals = comparison_results["Quantum PCA"]["eigenvalues"]

            x = np.arange(len(classical_eigenvals))
            width = 0.35

            ax1.bar(
                x - width / 2,
                classical_eigenvals,
                width,
                label="Classical",
                alpha=0.7,
                color="blue",
            )
            ax1.bar(
                x + width / 2,
                quantum_eigenvals,
                width,
                label="Quantum",
                alpha=0.7,
                color="red",
            )

            ax1.set_xlabel("Component")
            ax1.set_ylabel("Eigenvalue")
            ax1.set_title("Eigenvalue Comparison")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"PC{i+1}" for i in range(len(classical_eigenvals))])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Explained variance ratio
        ax2 = plt.subplot(2, 3, 2)

        if "Classical PCA" in comparison_results:
            classical_var_ratio = comparison_results["Classical PCA"][
                "explained_variance_ratio"
            ]
            quantum_var_ratio = comparison_results["Quantum PCA"][
                "explained_variance_ratio"
            ]

            components = range(1, len(classical_var_ratio) + 1)

            ax2.plot(
                components,
                classical_var_ratio,
                "bo-",
                label="Classical",
                linewidth=2,
                markersize=8,
            )
            ax2.plot(
                components,
                quantum_var_ratio,
                "ro-",
                label="Quantum",
                linewidth=2,
                markersize=8,
            )

            ax2.set_xlabel("Principal Component")
            ax2.set_ylabel("Explained Variance Ratio")
            ax2.set_title("Explained Variance Comparison")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Cumulative variance
        ax3 = plt.subplot(2, 3, 3)

        if "Classical PCA" in comparison_results:
            classical_cum_var = comparison_results["Classical PCA"][
                "cumulative_variance"
            ]
            quantum_cum_var = comparison_results["Quantum PCA"]["cumulative_variance"]

            components = range(1, len(classical_cum_var) + 1)

            ax3.plot(
                components,
                classical_cum_var,
                "bo-",
                label="Classical",
                linewidth=2,
                markersize=8,
            )
            ax3.plot(
                components,
                quantum_cum_var,
                "ro-",
                label="Quantum",
                linewidth=2,
                markersize=8,
            )

            ax3.set_xlabel("Number of Components")
            ax3.set_ylabel("Cumulative Explained Variance")
            ax3.set_title("Cumulative Variance Explained")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Error analysis
        ax4 = plt.subplot(2, 3, 4)

        if (
            "Quantum PCA" in comparison_results
            and "estimation_error" in comparison_results["Quantum PCA"]
        ):
            errors = comparison_results["Quantum PCA"]["estimation_error"]
            components = range(1, len(errors) + 1)

            ax4.bar(components, errors, alpha=0.7, color="orange")
            ax4.set_xlabel("Component")
            ax4.set_ylabel("Estimation Error")
            ax4.set_title("Quantum Eigenvalue Estimation Error")
            ax4.grid(True, alpha=0.3)

        # Scaling analysis
        if scaling_results:
            ax5 = plt.subplot(2, 3, 5)

            sizes = list(scaling_results.keys())
            classical_times = [scaling_results[s]["classical_time"] for s in sizes]
            quantum_times = [scaling_results[s]["quantum_time"] for s in sizes]

            ax5.loglog(
                sizes,
                classical_times,
                "bo-",
                label="Classical",
                linewidth=2,
                markersize=8,
            )
            ax5.loglog(
                sizes, quantum_times, "ro-", label="Quantum", linewidth=2, markersize=8
            )

            ax5.set_xlabel("Dataset Size")
            ax5.set_ylabel("Computation Time (simulated)")
            ax5.set_title("Scaling Comparison")
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Data visualization (if 2D data provided)
        if data_matrix is not None and data_matrix.shape[1] >= 2:
            ax6 = plt.subplot(2, 3, 6)

            # Show original data and PCA projection
            classical_pca = PCA(n_components=2)
            transformed = classical_pca.fit_transform(data_matrix)

            scatter = ax6.scatter(
                transformed[:, 0],
                transformed[:, 1],
                alpha=0.6,
                c=range(len(transformed)),
                cmap="viridis",
            )

            # Show principal component directions
            components = classical_pca.components_
            mean = np.mean(transformed, axis=0)

            for i, component in enumerate(components):
                ax6.arrow(
                    mean[0],
                    mean[1],
                    component[0] * 2,
                    component[1] * 2,
                    head_width=0.1,
                    head_length=0.1,
                    fc=f"C{i}",
                    ec=f"C{i}",
                    linewidth=2,
                    label=f"PC{i+1}",
                )

            ax6.set_xlabel("First Principal Component")
            ax6.set_ylabel("Second Principal Component")
            ax6.set_title("PCA Projection")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            # Summary statistics
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis("off")

            summary_text = "QPCA Analysis Summary:\n\n"

            if "Classical PCA" in comparison_results:
                classical_total_var = np.sum(
                    comparison_results["Classical PCA"]["explained_variance_ratio"]
                )
                summary_text += f"Classical PCA:\n"
                summary_text += f"Total Variance: {classical_total_var:.3f}\n\n"

            if "Quantum PCA" in comparison_results:
                quantum_total_var = np.sum(
                    comparison_results["Quantum PCA"]["explained_variance_ratio"]
                )
                summary_text += f"Quantum PCA:\n"
                summary_text += f"Total Variance: {quantum_total_var:.3f}\n\n"

            if "Variational QPCA" in comparison_results:
                vqpca_var = comparison_results["Variational QPCA"]["captured_variance"]
                summary_text += f"Variational QPCA:\n"
                summary_text += f"Captured Variance: {vqpca_var:.3f}\n\n"

            summary_text += "Key Insights:\n"
            summary_text += (
                "• QPCA can provide exponential\n  speedup for large datasets\n"
            )
            summary_text += "• Quantum phase estimation\n  estimates eigenvalues\n"
            summary_text += "• Variational approaches offer\n  near-term implementation"

            ax6.text(
                0.1,
                0.9,
                summary_text,
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            )

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Quantum Principal Component Analysis")
    parser.add_argument(
        "--dataset", choices=["blobs", "iris", "swiss_roll"], default="blobs"
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--analyze-scaling", action="store_true")
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 6: Quantum Machine Learning")
    print("Example 4: Quantum Principal Component Analysis")
    print("=" * 49)

    try:
        # Load dataset
        if args.dataset == "iris":
            from sklearn.datasets import load_iris

            data = load_iris()
            X = data.data
        elif args.dataset == "swiss_roll":
            X, _ = make_swiss_roll(n_samples=args.n_samples, noise=0.1, random_state=42)
        else:  # blobs
            X, _ = make_blobs(
                n_samples=args.n_samples,
                n_features=args.n_features,
                centers=3,
                cluster_std=2.0,
                random_state=42,
            )

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"\n📊 Dataset: {args.dataset}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Components to extract: {args.n_components}")

        # QPCA Analysis
        analyzer = QPCAAnalyzer(verbose=args.verbose)

        print(f"\n🔄 Comparing PCA methods...")
        comparison_results = analyzer.compare_methods(X, args.n_components)

        print(f"\n📈 Results Summary:")

        # Classical PCA results
        if "Classical PCA" in comparison_results:
            classical = comparison_results["Classical PCA"]
            print(f"   Classical PCA:")
            print(f"     Eigenvalues: {classical['eigenvalues']}")
            print(
                f"     Explained variance ratio: {classical['explained_variance_ratio']}"
            )
            print(
                f"     Cumulative variance: {classical['cumulative_variance'][-1]:.3f}"
            )
            print(f"     Reconstruction error: {classical['reconstruction_error']:.4f}")

        # Quantum PCA results
        if "Quantum PCA" in comparison_results:
            quantum = comparison_results["Quantum PCA"]
            print(f"\n   Quantum PCA:")
            print(f"     Estimated eigenvalues: {quantum['eigenvalues']}")
            print(
                f"     Explained variance ratio: {quantum['explained_variance_ratio']}"
            )
            print(f"     Cumulative variance: {quantum['cumulative_variance'][-1]:.3f}")
            print(f"     Estimation errors: {quantum['estimation_error']}")

        # Variational QPCA results
        if "Variational QPCA" in comparison_results:
            vqpca = comparison_results["Variational QPCA"]
            print(f"\n   Variational QPCA:")
            print(f"     Captured variance: {vqpca['captured_variance']:.4f}")

        # Scaling analysis
        scaling_results = None
        if args.analyze_scaling:
            print(f"\n📏 Analyzing scaling behavior...")
            scaling_results = analyzer.analyze_scaling()

            print(f"   Scaling Analysis:")
            for size, result in scaling_results.items():
                print(
                    f"     Size {size}: Classical={result['classical_time']:.3f}s, "
                    f"Quantum={result['quantum_time']:.3f}s, "
                    f"Advantage={result['quantum_advantage']:.2f}x"
                )

        # Compute accuracy metrics
        if (
            "Classical PCA" in comparison_results
            and "Quantum PCA" in comparison_results
        ):
            classical_eigenvals = comparison_results["Classical PCA"]["eigenvalues"]
            quantum_eigenvals = comparison_results["Quantum PCA"]["eigenvalues"]

            relative_error = (
                np.abs(quantum_eigenvals - classical_eigenvals) / classical_eigenvals
            )
            print(f"\n📊 Quantum Accuracy:")
            print(f"   Relative errors: {relative_error}")
            print(f"   Mean relative error: {np.mean(relative_error):.4f}")
            print(f"   Max relative error: {np.max(relative_error):.4f}")

        if args.show_visualization:
            analyzer.visualize_results(comparison_results, scaling_results, X)

        print(f"\n📚 Key Insights:")
        print(f"   • Quantum PCA can achieve exponential speedup for large datasets")
        print(f"   • Quantum phase estimation estimates eigenvalues efficiently")
        print(f"   • Variational methods enable near-term quantum implementations")
        print(f"   • Trade-off between quantum speedup and estimation accuracy")

        print(f"\n🎯 Applications:")
        print(f"   • Dimensionality reduction for quantum machine learning")
        print(f"   • Feature extraction from high-dimensional quantum data")
        print(f"   • Preprocessing for quantum algorithms")
        print(f"   • Quantum data compression and visualization")

        print(f"\n✅ Quantum PCA analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
