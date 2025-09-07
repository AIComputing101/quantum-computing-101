#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 6: Quantum Machine Learning
Example 2: Variational Quantum Classifier

Implementation of variational quantum classifiers with parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit.library import TwoLocal, EfficientSU2
from sklearn.datasets import make_classification, load_iris, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class VariationalQuantumClassifier:
    def __init__(self, n_qubits, depth=2, verbose=False):
        self.n_qubits = n_qubits
        self.depth = depth
        self.verbose = verbose
        self.parameters = None
        self.training_history = []

    def create_feature_map(self, x):
        """Create feature map to encode classical data."""
        circuit = QuantumCircuit(self.n_qubits, name="FeatureMap")

        # Angle encoding
        for i in range(min(len(x), self.n_qubits)):
            circuit.ry(x[i] * np.pi, i)

        # Entangling layer
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)

        return circuit

    def create_ansatz(self, parameters):
        """Create parameterized quantum circuit (ansatz)."""
        circuit = QuantumCircuit(self.n_qubits, name="Ansatz")
        param_idx = 0

        for layer in range(self.depth):
            # Rotation layer
            for qubit in range(self.n_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(parameters):
                    circuit.rz(parameters[param_idx], qubit)
                    param_idx += 1

            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            # Add circular entanglement for last layer
            if layer == self.depth - 1:
                circuit.cx(self.n_qubits - 1, 0)

        return circuit

    def create_vqc_circuit(self, x, parameters):
        """Create complete VQC circuit."""
        # Feature map
        circuit = self.create_feature_map(x)

        # Ansatz
        ansatz = self.create_ansatz(parameters)
        circuit.compose(ansatz, inplace=True)

        return circuit

    def measure_expectation(self, circuit, observable="Z0"):
        """Measure expectation value of observable."""
        # Add measurement
        meas_circuit = circuit.copy()
        meas_circuit.add_register(ClassicalRegister(1))
        meas_circuit.measure(0, 0)  # Measure first qubit for Z0 observable

        # Simulate
        simulator = AerSimulator()
        job = simulator.run(meas_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Calculate expectation value of Z
        prob_0 = counts.get("0", 0) / 1024
        prob_1 = counts.get("1", 0) / 1024
        expectation = prob_0 - prob_1  # ⟨Z⟩ = P(0) - P(1)

        return expectation

    def predict_single(self, x, parameters):
        """Predict single sample."""
        circuit = self.create_vqc_circuit(x, parameters)
        expectation = self.measure_expectation(circuit)

        # Convert expectation to class prediction
        return 1 if expectation > 0 else 0

    def predict(self, X, parameters):
        """Predict multiple samples."""
        predictions = []
        for x in X:
            pred = self.predict_single(x, parameters)
            predictions.append(pred)
        return np.array(predictions)

    def cost_function(self, parameters, X, y):
        """Cost function for optimization."""
        predictions = self.predict(X, parameters)

        # Binary cross-entropy loss
        epsilon = 1e-15  # Avoid log(0)

        # Convert expectations to probabilities
        probabilities = []
        for x in X:
            circuit = self.create_vqc_circuit(x, parameters)
            expectation = self.measure_expectation(circuit)
            prob = (expectation + 1) / 2  # Convert from [-1,1] to [0,1]
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        # Binary cross-entropy
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

        # Accuracy for monitoring
        accuracy = np.mean(predictions == y)

        self.training_history.append(
            {"loss": loss, "accuracy": accuracy, "parameters": parameters.copy()}
        )

        if self.verbose and len(self.training_history) % 10 == 0:
            print(
                f"Iteration {len(self.training_history)}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}"
            )

        return loss

    def fit(self, X, y, max_iter=100):
        """Train the VQC."""
        # Initialize parameters
        n_params = (
            2 * self.n_qubits * self.depth
        )  # 2 rotation gates per qubit per layer
        initial_params = np.random.uniform(0, 2 * np.pi, n_params)

        self.training_history = []

        # Optimize parameters
        if self.verbose:
            print(f"Training VQC with {n_params} parameters...")

        result = minimize(
            self.cost_function,
            initial_params,
            args=(X, y),
            method="COBYLA",
            options={"maxiter": max_iter},
        )

        self.parameters = result.x

        return self

    def score(self, X, y):
        """Calculate accuracy score."""
        if self.parameters is None:
            raise ValueError("Model not trained. Call fit() first.")

        predictions = self.predict(X, self.parameters)
        return accuracy_score(y, predictions)


class VQCAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compare_architectures(self, X_train, y_train, X_test, y_test):
        """Compare different VQC architectures."""
        results = {}

        # Different configurations
        configs = [
            {"n_qubits": 2, "depth": 1, "name": "Shallow (2q, d=1)"},
            {"n_qubits": 2, "depth": 2, "name": "Medium (2q, d=2)"},
            {"n_qubits": 3, "depth": 1, "name": "Wide (3q, d=1)"},
            {"n_qubits": 3, "depth": 2, "name": "Deep (3q, d=2)"},
        ]

        for config in configs:
            if self.verbose:
                print(f"Testing {config['name']}...")

            try:
                vqc = VariationalQuantumClassifier(
                    n_qubits=config["n_qubits"],
                    depth=config["depth"],
                    verbose=self.verbose,
                )

                # Pad or truncate features to match qubits
                X_train_padded = self.adjust_features(X_train, config["n_qubits"])
                X_test_padded = self.adjust_features(X_test, config["n_qubits"])

                vqc.fit(X_train_padded, y_train, max_iter=50)

                train_acc = vqc.score(X_train_padded, y_train)
                test_acc = vqc.score(X_test_padded, y_test)

                results[config["name"]] = {
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "n_parameters": 2 * config["n_qubits"] * config["depth"],
                    "training_history": vqc.training_history.copy(),
                }

            except Exception as e:
                if self.verbose:
                    print(f"Error with {config['name']}: {e}")
                results[config["name"]] = {
                    "train_accuracy": 0.0,
                    "test_accuracy": 0.0,
                    "error": str(e),
                }

        return results

    def adjust_features(self, X, n_qubits):
        """Adjust feature dimension to match number of qubits."""
        if X.shape[1] == n_qubits:
            return X
        elif X.shape[1] < n_qubits:
            # Pad with zeros
            padding = np.zeros((X.shape[0], n_qubits - X.shape[1]))
            return np.hstack([X, padding])
        else:
            # Truncate features
            return X[:, :n_qubits]

    def analyze_optimization_landscape(self, X, y, n_qubits=2, depth=1):
        """Analyze optimization landscape."""
        vqc = VariationalQuantumClassifier(n_qubits, depth, verbose=False)

        # Adjust features
        X_adj = self.adjust_features(X, n_qubits)

        # Sample parameter space
        n_params = 2 * n_qubits * depth
        n_samples = 100

        losses = []
        parameter_samples = []

        for _ in range(n_samples):
            params = np.random.uniform(0, 2 * np.pi, n_params)
            loss = vqc.cost_function(params, X_adj, y)
            losses.append(loss)
            parameter_samples.append(params)

        return {
            "losses": np.array(losses),
            "parameters": np.array(parameter_samples),
            "min_loss": np.min(losses),
            "mean_loss": np.mean(losses),
            "loss_std": np.std(losses),
        }

    def quantum_vs_classical_comparison(self, X_train, y_train, X_test, y_test):
        """Compare quantum vs classical approaches."""
        results = {}

        # Quantum VQC
        vqc = VariationalQuantumClassifier(n_qubits=2, depth=2, verbose=self.verbose)
        X_train_quantum = self.adjust_features(X_train, 2)
        X_test_quantum = self.adjust_features(X_test, 2)

        vqc.fit(X_train_quantum, y_train, max_iter=50)

        results["Quantum VQC"] = {
            "train_accuracy": vqc.score(X_train_quantum, y_train),
            "test_accuracy": vqc.score(X_test_quantum, y_test),
            "n_parameters": 2
            * 2
            * 2,  # 2 qubits, depth 2, 2 params per qubit per layer
        }

        # Classical Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        results["Random Forest"] = {
            "train_accuracy": rf.score(X_train, y_train),
            "test_accuracy": rf.score(X_test, y_test),
            "n_parameters": len(rf.estimators_) * X_train.shape[1],  # Approximate
        }

        return results

    def visualize_results(
        self,
        architecture_results,
        landscape_results=None,
        quantum_classical_results=None,
    ):
        """Visualize VQC analysis results."""
        fig = plt.figure(figsize=(16, 12))

        # Architecture comparison
        ax1 = plt.subplot(2, 3, 1)
        architectures = list(architecture_results.keys())
        train_accs = [
            architecture_results[arch].get("train_accuracy", 0)
            for arch in architectures
        ]
        test_accs = [
            architecture_results[arch].get("test_accuracy", 0) for arch in architectures
        ]

        x = np.arange(len(architectures))
        width = 0.35

        ax1.bar(
            x - width / 2, train_accs, width, label="Train", alpha=0.7, color="blue"
        )
        ax1.bar(x + width / 2, test_accs, width, label="Test", alpha=0.7, color="red")

        ax1.set_xlabel("Architecture")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("VQC Architecture Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(architectures, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training convergence
        ax2 = plt.subplot(2, 3, 2)

        # Plot training history for first architecture
        first_arch = list(architecture_results.keys())[0]
        if "training_history" in architecture_results[first_arch]:
            history = architecture_results[first_arch]["training_history"]
            iterations = range(len(history))
            losses = [h["loss"] for h in history]
            accuracies = [h["accuracy"] for h in history]

            ax2_twin = ax2.twinx()
            line1 = ax2.plot(iterations, losses, "b-", label="Loss")
            line2 = ax2_twin.plot(iterations, accuracies, "r-", label="Accuracy")

            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Loss", color="blue")
            ax2_twin.set_ylabel("Accuracy", color="red")
            ax2.set_title(f"Training Convergence\n({first_arch})")

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc="center right")

            ax2.grid(True, alpha=0.3)

        # Parameter count vs performance
        ax3 = plt.subplot(2, 3, 3)
        param_counts = [
            architecture_results[arch].get("n_parameters", 0) for arch in architectures
        ]

        # Filter out errors
        valid_indices = [
            i
            for i, arch in enumerate(architectures)
            if "error" not in architecture_results[arch]
        ]
        valid_architectures = [architectures[i] for i in valid_indices]
        valid_param_counts = [param_counts[i] for i in valid_indices]
        valid_test_accs = [test_accs[i] for i in valid_indices]

        if valid_param_counts:
            ax3.scatter(
                valid_param_counts,
                valid_test_accs,
                s=100,
                alpha=0.7,
                c=range(len(valid_param_counts)),
                cmap="viridis",
            )

            for i, arch in enumerate(valid_architectures):
                ax3.annotate(
                    arch,
                    (valid_param_counts[i], valid_test_accs[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            ax3.set_xlabel("Number of Parameters")
            ax3.set_ylabel("Test Accuracy")
            ax3.set_title("Parameters vs Performance")
            ax3.grid(True, alpha=0.3)

        # Optimization landscape
        if landscape_results:
            ax4 = plt.subplot(2, 3, 4)
            losses = landscape_results["losses"]

            ax4.hist(losses, bins=20, alpha=0.7, color="green", edgecolor="black")
            ax4.axvline(
                landscape_results["min_loss"],
                color="red",
                linestyle="--",
                label=f'Min: {landscape_results["min_loss"]:.3f}',
            )
            ax4.axvline(
                landscape_results["mean_loss"],
                color="blue",
                linestyle="--",
                label=f'Mean: {landscape_results["mean_loss"]:.3f}',
            )

            ax4.set_xlabel("Loss Value")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Optimization Landscape")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Quantum vs Classical
        if quantum_classical_results:
            ax5 = plt.subplot(2, 3, 5)
            methods = list(quantum_classical_results.keys())
            qc_train_accs = [
                quantum_classical_results[m]["train_accuracy"] for m in methods
            ]
            qc_test_accs = [
                quantum_classical_results[m]["test_accuracy"] for m in methods
            ]

            x = np.arange(len(methods))
            ax5.bar(
                x - width / 2,
                qc_train_accs,
                width,
                label="Train",
                alpha=0.7,
                color="blue",
            )
            ax5.bar(
                x + width / 2, qc_test_accs, width, label="Test", alpha=0.7, color="red"
            )

            ax5.set_xlabel("Method")
            ax5.set_ylabel("Accuracy")
            ax5.set_title("Quantum vs Classical")
            ax5.set_xticks(x)
            ax5.set_xticklabels(methods)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        # Create summary text
        summary_text = "VQC Analysis Summary:\n\n"

        best_arch = max(
            architecture_results.items(), key=lambda x: x[1].get("test_accuracy", 0)
        )

        summary_text += f"Best Architecture:\n{best_arch[0]}\n"
        summary_text += f"Test Accuracy: {best_arch[1].get('test_accuracy', 0):.3f}\n\n"

        if quantum_classical_results:
            quantum_acc = quantum_classical_results["Quantum VQC"]["test_accuracy"]
            classical_acc = quantum_classical_results["Random Forest"]["test_accuracy"]

            summary_text += f"Quantum vs Classical:\n"
            summary_text += f"VQC: {quantum_acc:.3f}\n"
            summary_text += f"RF:  {classical_acc:.3f}\n"

            if quantum_acc > classical_acc:
                summary_text += f"Quantum advantage: +{quantum_acc - classical_acc:.3f}"
            else:
                summary_text += f"Classical leads: +{classical_acc - quantum_acc:.3f}"

        ax6.text(
            0.1,
            0.9,
            summary_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Variational Quantum Classifier")
    parser.add_argument(
        "--dataset", choices=["iris", "moons", "random"], default="moons"
    )
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--compare-architectures", action="store_true")
    parser.add_argument("--analyze-landscape", action="store_true")
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 6: Quantum Machine Learning")
    print("Example 2: Variational Quantum Classifier")
    print("=" * 47)

    try:
        # Load dataset
        if args.dataset == "iris":
            data = load_iris()
            X, y = data.data[:, :2], data.target
            # Binary classification
            mask = y != 2
            X, y = X[mask], y[mask]
        elif args.dataset == "moons":
            X, y = make_moons(n_samples=args.n_samples, noise=0.1, random_state=42)
        else:  # random
            X, y = make_classification(
                n_samples=args.n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_clusters_per_class=1,
                random_state=42,
            )

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print(f"\n📊 Dataset: {args.dataset}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Classes: {len(np.unique(y))}")

        # Single VQC training
        print(f"\n🧠 Training VQC ({args.n_qubits} qubits, depth {args.depth})...")

        analyzer = VQCAnalyzer(verbose=args.verbose)

        vqc = VariationalQuantumClassifier(
            n_qubits=args.n_qubits, depth=args.depth, verbose=args.verbose
        )

        # Adjust features for quantum circuit
        X_train_quantum = analyzer.adjust_features(X_train, args.n_qubits)
        X_test_quantum = analyzer.adjust_features(X_test, args.n_qubits)

        vqc.fit(X_train_quantum, y_train, max_iter=args.max_iter)

        train_acc = vqc.score(X_train_quantum, y_train)
        test_acc = vqc.score(X_test_quantum, y_test)

        print(f"\n📈 VQC Results:")
        print(f"   Training accuracy: {train_acc:.3f}")
        print(f"   Test accuracy: {test_acc:.3f}")
        print(f"   Parameters optimized: {len(vqc.parameters)}")
        print(f"   Training iterations: {len(vqc.training_history)}")

        # Architecture comparison
        architecture_results = None
        if args.compare_architectures:
            print(f"\n🏗️  Comparing VQC architectures...")
            architecture_results = analyzer.compare_architectures(
                X_train, y_train, X_test, y_test
            )

            print(f"\n📊 Architecture Comparison:")
            for arch, results in architecture_results.items():
                if "error" not in results:
                    print(
                        f"   {arch:20s}: Test = {results['test_accuracy']:.3f}, "
                        f"Params = {results['n_parameters']}"
                    )
                else:
                    print(f"   {arch:20s}: Error")

        # Optimization landscape analysis
        landscape_results = None
        if args.analyze_landscape:
            print(f"\n🗺️  Analyzing optimization landscape...")
            landscape_results = analyzer.analyze_optimization_landscape(
                X_train, y_train, args.n_qubits, args.depth
            )

            print(f"   Loss statistics:")
            print(f"     Minimum: {landscape_results['min_loss']:.4f}")
            print(f"     Mean: {landscape_results['mean_loss']:.4f}")
            print(f"     Std: {landscape_results['loss_std']:.4f}")

        # Quantum vs Classical comparison
        print(f"\n⚖️  Quantum vs Classical comparison...")
        quantum_classical_results = analyzer.quantum_vs_classical_comparison(
            X_train, y_train, X_test, y_test
        )

        print(f"\n🏆 Method Comparison:")
        for method, results in quantum_classical_results.items():
            print(f"   {method:15s}: Test = {results['test_accuracy']:.3f}")

        quantum_acc = quantum_classical_results["Quantum VQC"]["test_accuracy"]
        classical_acc = quantum_classical_results["Random Forest"]["test_accuracy"]

        if quantum_acc > classical_acc:
            print(f"   🎉 Quantum advantage: +{quantum_acc - classical_acc:.3f}")
        else:
            print(f"   📊 Classical advantage: +{classical_acc - quantum_acc:.3f}")

        if args.show_visualization:
            if not architecture_results:
                # Create minimal architecture results for visualization
                architecture_results = {
                    f"VQC ({args.n_qubits}q, d={args.depth})": {
                        "train_accuracy": train_acc,
                        "test_accuracy": test_acc,
                        "n_parameters": len(vqc.parameters),
                        "training_history": vqc.training_history,
                    }
                }

            analyzer.visualize_results(
                architecture_results, landscape_results, quantum_classical_results
            )

        print(f"\n📚 Key Insights:")
        print(f"   • VQCs use parameterized quantum circuits for classification")
        print(f"   • Feature maps encode classical data into quantum states")
        print(f"   • Variational optimization trains quantum parameters")
        print(f"   • Circuit depth vs expressivity trade-off exists")

        print(f"\n✅ Variational quantum classifier analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    from qiskit import ClassicalRegister

    exit(main())
