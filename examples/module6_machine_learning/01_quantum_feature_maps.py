#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 6: Quantum Machine Learning
Example 1: Quantum Feature Maps

Implementation of quantum feature maps for encoding classical data into quantum states.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from sklearn.datasets import make_classification, load_iris, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")


class QuantumFeatureMaps:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def create_angle_encoding_map(self, n_features, data_point):
        """Create angle encoding feature map."""
        circuit = QuantumCircuit(n_features, name="Angle_Encoding")

        for i, feature in enumerate(data_point):
            # Encode feature as rotation angle
            circuit.ry(feature * np.pi, i)

        return circuit

    def create_amplitude_encoding_map(self, data_point):
        """Create amplitude encoding feature map."""
        # Normalize data for amplitude encoding
        normalized_data = data_point / np.linalg.norm(data_point)

        # Pad to power of 2 if necessary
        n_qubits = int(np.ceil(np.log2(len(normalized_data))))
        n_amplitudes = 2**n_qubits

        if len(normalized_data) < n_amplitudes:
            padded_data = np.zeros(n_amplitudes)
            padded_data[: len(normalized_data)] = normalized_data
            normalized_data = padded_data

        circuit = QuantumCircuit(n_qubits, name="Amplitude_Encoding")
        circuit.initialize(normalized_data, range(n_qubits))

        return circuit

    def create_iqp_feature_map(self, n_features, data_point, depth=2):
        """Create IQP (Instantaneous Quantum Polynomial) feature map."""
        circuit = QuantumCircuit(n_features, name="IQP_FeatureMap")

        for layer in range(depth):
            # Hadamard layer
            for i in range(n_features):
                circuit.h(i)

            # Entangling layer with data encoding
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Two-qubit interaction with data encoding
                    angle = data_point[i] * data_point[j] * np.pi
                    circuit.cp(angle, i, j)

            # Single-qubit rotations
            for i in range(n_features):
                circuit.rz(data_point[i] * np.pi, i)

        return circuit

    def create_pauli_feature_map(self, n_features, data_point, entanglement="linear"):
        """Create Pauli feature map using Qiskit's built-in implementation."""
        # Use Qiskit's PauliFeatureMap
        feature_map = PauliFeatureMap(
            feature_dimension=n_features,
            reps=2,
            entanglement=entanglement,
            paulis=["Z", "ZZ"],
        )

        # Bind data
        bound_circuit = feature_map.bind_parameters(data_point)

        return bound_circuit

    def compute_kernel_matrix(self, X, feature_map_func, shots=1024):
        """Compute quantum kernel matrix."""
        n_samples = len(X)
        kernel_matrix = np.zeros((n_samples, n_samples))

        simulator = AerSimulator()

        for i in range(n_samples):
            for j in range(i, n_samples):
                # Create feature map circuits
                circuit_i = feature_map_func(X[i])
                circuit_j = feature_map_func(X[j])

                # Compute inner product |⟨φ(xi)|φ(xj)⟩|²
                kernel_value = self.compute_kernel_element(
                    circuit_i, circuit_j, simulator, shots
                )

                kernel_matrix[i, j] = kernel_value
                kernel_matrix[j, i] = kernel_value  # Symmetric

        return kernel_matrix

    def compute_kernel_element(self, circuit_i, circuit_j, simulator, shots):
        """Compute single kernel matrix element."""
        # Create circuit for inner product measurement
        n_qubits = circuit_i.num_qubits

        # Prepare |φ(xi)⟩
        kernel_circuit = QuantumCircuit(n_qubits, n_qubits)
        kernel_circuit.compose(circuit_i, inplace=True)

        # Apply adjoint of |φ(xj)⟩
        kernel_circuit.compose(circuit_j.inverse(), inplace=True)

        # Measure overlap
        kernel_circuit.measure_all()

        # Execute
        job = simulator.run(kernel_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Probability of measuring all zeros (overlap)
        zero_state = "0" * n_qubits
        overlap_prob = counts.get(zero_state, 0) / shots

        return overlap_prob

    def analyze_expressivity(self, feature_map_func, n_samples=100, n_features=2):
        """Analyze expressivity of feature map."""
        # Generate random data points
        X = np.random.uniform(-1, 1, (n_samples, n_features))

        # Compute all pairwise quantum state overlaps
        overlaps = []
        classical_distances = []

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Quantum overlap
                circuit_i = feature_map_func(X[i])
                circuit_j = feature_map_func(X[j])

                # Simplified overlap calculation using statevectors
                try:
                    state_i = Statevector.from_instruction(circuit_i)
                    state_j = Statevector.from_instruction(circuit_j)
                    overlap = abs(state_i.inner(state_j)) ** 2
                    overlaps.append(overlap)

                    # Classical Euclidean distance
                    distance = np.linalg.norm(X[i] - X[j])
                    classical_distances.append(distance)

                except Exception:
                    # Skip if statevector computation fails
                    continue

        return {
            "overlaps": np.array(overlaps),
            "classical_distances": np.array(classical_distances),
            "expressivity_measure": np.std(overlaps),  # Higher std = more expressive
        }

    def compare_feature_maps(self, X, y, test_size=0.3):
        """Compare different feature maps for classification."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        n_features = X.shape[1]
        results = {}

        # Define feature map functions
        feature_maps = {
            "Angle Encoding": lambda x: self.create_angle_encoding_map(n_features, x),
            "IQP": lambda x: self.create_iqp_feature_map(n_features, x),
            "Pauli": lambda x: self.create_pauli_feature_map(n_features, x),
        }

        for name, feature_map_func in feature_maps.items():
            if self.verbose:
                print(f"Testing {name} feature map...")

            try:
                # Compute kernel matrices
                K_train = self.compute_kernel_matrix(
                    X_train, feature_map_func, shots=512
                )
                K_test = self.compute_kernel_matrix(X_test, feature_map_func, shots=512)

                # Train quantum kernel SVM
                qsvm = SVC(kernel="precomputed")
                qsvm.fit(K_train, y_train)

                # Predict
                train_accuracy = qsvm.score(K_train, y_train)

                # For test set, need kernel between test and train
                K_test_train = np.zeros((len(X_test), len(X_train)))
                for i, x_test in enumerate(X_test):
                    for j, x_train in enumerate(X_train):
                        circuit_test = feature_map_func(x_test)
                        circuit_train = feature_map_func(x_train)
                        K_test_train[i, j] = self.compute_kernel_element(
                            circuit_test, circuit_train, AerSimulator(), 512
                        )

                test_predictions = qsvm.predict(K_test_train)
                test_accuracy = np.mean(test_predictions == y_test)

                results[name] = {
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "kernel_train": K_train,
                    "kernel_test": K_test,
                }

            except Exception as e:
                if self.verbose:
                    print(f"Error with {name}: {e}")
                results[name] = {
                    "train_accuracy": 0.0,
                    "test_accuracy": 0.0,
                    "error": str(e),
                }

        # Classical SVM comparison
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classical_svm = SVC(kernel="rbf")
        classical_svm.fit(X_train_scaled, y_train)

        results["Classical RBF"] = {
            "train_accuracy": classical_svm.score(X_train_scaled, y_train),
            "test_accuracy": classical_svm.score(X_test_scaled, y_test),
        }

        return results

    def visualize_feature_maps(self, comparison_results, expressivity_results=None):
        """Visualize feature map analysis."""
        fig = plt.figure(figsize=(16, 12))

        # Accuracy comparison
        ax1 = plt.subplot(2, 3, 1)
        methods = list(comparison_results.keys())
        train_accs = [comparison_results[m].get("train_accuracy", 0) for m in methods]
        test_accs = [comparison_results[m].get("test_accuracy", 0) for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        ax1.bar(
            x - width / 2, train_accs, width, label="Train", alpha=0.7, color="blue"
        )
        ax1.bar(x + width / 2, test_accs, width, label="Test", alpha=0.7, color="red")

        ax1.set_xlabel("Feature Map")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Classification Performance Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Kernel matrix visualization
        if "Angle Encoding" in comparison_results:
            ax2 = plt.subplot(2, 3, 2)
            kernel_data = comparison_results["Angle Encoding"]
            if "kernel_train" in kernel_data:
                K = kernel_data["kernel_train"]
                im = ax2.imshow(K, cmap="viridis", aspect="auto")
                ax2.set_title("Quantum Kernel Matrix\n(Angle Encoding)")
                ax2.set_xlabel("Sample Index")
                ax2.set_ylabel("Sample Index")
                plt.colorbar(im, ax=ax2)

        # Expressivity analysis
        if expressivity_results:
            ax3 = plt.subplot(2, 3, 3)
            overlaps = expressivity_results["overlaps"]
            distances = expressivity_results["classical_distances"]

            ax3.scatter(distances, overlaps, alpha=0.6, s=20)
            ax3.set_xlabel("Classical Distance")
            ax3.set_ylabel("Quantum Overlap")
            ax3.set_title("Feature Map Expressivity")
            ax3.grid(True, alpha=0.3)

            # Add expressivity measure
            expressivity = expressivity_results["expressivity_measure"]
            ax3.text(
                0.05,
                0.95,
                f"Expressivity: {expressivity:.3f}",
                transform=ax3.transAxes,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )

        # Feature encoding example
        ax4 = plt.subplot(2, 3, 4)
        # Example 2D data point
        example_point = np.array([0.5, -0.3])

        # Show different encodings
        encoding_methods = ["Angle", "IQP", "Pauli"]
        encoding_complexities = [2, 6, 8]  # Approximate circuit depths

        ax4.bar(encoding_methods, encoding_complexities, alpha=0.7, color="green")
        ax4.set_title("Feature Map Complexity\n(Circuit Depth)")
        ax4.set_ylabel("Approximate Depth")
        ax4.grid(True, alpha=0.3)

        # Performance vs complexity
        ax5 = plt.subplot(2, 3, 5)
        if len(test_accs) >= 3:  # Ensure we have quantum methods
            quantum_methods = methods[:-1]  # Exclude classical
            quantum_accs = test_accs[:-1]
            complexities = encoding_complexities[: len(quantum_accs)]

            ax5.scatter(
                complexities,
                quantum_accs,
                s=100,
                alpha=0.7,
                c=range(len(quantum_accs)),
                cmap="viridis",
            )

            for i, method in enumerate(quantum_methods):
                if i < len(complexities):
                    ax5.annotate(
                        method,
                        (complexities[i], quantum_accs[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

            ax5.set_xlabel("Circuit Complexity")
            ax5.set_ylabel("Test Accuracy")
            ax5.set_title("Accuracy vs Complexity Trade-off")
            ax5.grid(True, alpha=0.3)

        # Method comparison summary
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        # Create summary text
        summary_text = "Feature Map Summary:\n\n"
        for method, results in comparison_results.items():
            if "error" not in results:
                summary_text += f"{method}:\n"
                summary_text += f"  Train: {results['train_accuracy']:.3f}\n"
                summary_text += f"  Test:  {results['test_accuracy']:.3f}\n\n"

        ax6.text(
            0.1,
            0.9,
            summary_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.7),
        )

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum Feature Maps Analysis")
    parser.add_argument(
        "--dataset", choices=["iris", "moons", "random"], default="moons"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for random dataset",
    )
    parser.add_argument("--n-features", type=int, default=2, help="Number of features")
    parser.add_argument("--test-expressivity", action="store_true")
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 6: Quantum Machine Learning")
    print("Example 1: Quantum Feature Maps")
    print("=" * 42)

    feature_maps = QuantumFeatureMaps(verbose=args.verbose)

    try:
        # Load dataset
        if args.dataset == "iris":
            data = load_iris()
            X, y = data.data[:, :2], data.target  # Use first 2 features
            # Binary classification
            mask = y != 2
            X, y = X[mask], y[mask]
        elif args.dataset == "moons":
            X, y = make_moons(n_samples=args.n_samples, noise=0.1, random_state=42)
        else:  # random
            X, y = make_classification(
                n_samples=args.n_samples,
                n_features=args.n_features,
                n_redundant=0,
                n_informative=args.n_features,
                n_clusters_per_class=1,
                random_state=42,
            )

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"\n📊 Dataset: {args.dataset}")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")

        # Compare feature maps
        print(f"\n🔄 Comparing quantum feature maps...")
        comparison_results = feature_maps.compare_feature_maps(X, y)

        print(f"\n📈 Classification Results:")
        for method, results in comparison_results.items():
            if "error" not in results:
                print(
                    f"   {method:15s}: Train={results['train_accuracy']:.3f}, "
                    f"Test={results['test_accuracy']:.3f}"
                )
            else:
                print(f"   {method:15s}: Error - {results['error']}")

        # Expressivity analysis
        expressivity_results = None
        if args.test_expressivity:
            print(f"\n🎯 Analyzing feature map expressivity...")

            def angle_encoding_func(x):
                return feature_maps.create_angle_encoding_map(len(x), x)

            expressivity_results = feature_maps.analyze_expressivity(
                angle_encoding_func, n_samples=50, n_features=X.shape[1]
            )

            print(
                f"   Expressivity measure: {expressivity_results['expressivity_measure']:.4f}"
            )
            print(f"   Mean overlap: {np.mean(expressivity_results['overlaps']):.4f}")
            print(f"   Overlap std: {np.std(expressivity_results['overlaps']):.4f}")

        # Find best quantum method
        quantum_methods = {
            k: v for k, v in comparison_results.items() if k != "Classical RBF"
        }
        if quantum_methods:
            best_quantum = max(
                quantum_methods.items(), key=lambda x: x[1].get("test_accuracy", 0)
            )
            classical_acc = comparison_results["Classical RBF"]["test_accuracy"]

            print(f"\n🏆 Best Results:")
            print(
                f"   Best Quantum: {best_quantum[0]} ({best_quantum[1]['test_accuracy']:.3f})"
            )
            print(f"   Classical RBF: {classical_acc:.3f}")

            if best_quantum[1]["test_accuracy"] > classical_acc:
                print(
                    f"   🎉 Quantum advantage: +{best_quantum[1]['test_accuracy'] - classical_acc:.3f}"
                )
            else:
                print(
                    f"   📊 Classical leads by: +{classical_acc - best_quantum[1]['test_accuracy']:.3f}"
                )

        if args.show_visualization:
            feature_maps.visualize_feature_maps(
                comparison_results, expressivity_results
            )

        print(f"\n📚 Key Insights:")
        print(f"   • Feature maps encode classical data into quantum Hilbert space")
        print(f"   • Different encodings have varying expressivity and complexity")
        print(f"   • Quantum kernels enable non-linear classification")
        print(f"   • Expressivity vs. trainability trade-off is crucial")

        print(f"\n✅ Quantum feature map analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
