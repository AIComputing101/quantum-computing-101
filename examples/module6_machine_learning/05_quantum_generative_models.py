#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 6: Quantum Machine Learning
Example 5: Quantum Generative Models

Implementation of quantum generative models including quantum GANs and Born machines.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class QuantumBornMachine:
    def __init__(self, n_qubits, n_layers=2, verbose=False):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.verbose = verbose
        self.parameters = None
        self.training_history = []

    def create_born_circuit(self, parameters):
        """Create Born machine circuit."""
        circuit = QuantumCircuit(self.n_qubits)
        param_idx = 0

        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(parameters):
                    circuit.rz(parameters[param_idx], qubit)
                    param_idx += 1

            # Entangling gates
            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

            # Circular entanglement
            if self.n_qubits > 2:
                circuit.cx(self.n_qubits - 1, 0)

        return circuit

    def generate_samples(self, parameters, n_samples=1000):
        """Generate samples from Born machine."""
        circuit = self.create_born_circuit(parameters)
        circuit.measure_all()

        # Execute circuit
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=n_samples)
        result = job.result()
        counts = result.get_counts()

        # Convert to probability distribution
        total_shots = sum(counts.values())
        probabilities = {}

        for state, count in counts.items():
            probabilities[state] = count / total_shots

        # Generate samples according to distribution
        samples = []
        states = list(probabilities.keys())
        probs = list(probabilities.values())

        for _ in range(n_samples):
            chosen_state = np.random.choice(states, p=probs)
            # Convert binary string to integer
            sample = int(chosen_state, 2)
            samples.append(sample)

        return np.array(samples), probabilities

    def compute_probability_distribution(self, parameters):
        """Compute exact probability distribution."""
        circuit = self.create_born_circuit(parameters)
        state = Statevector.from_instruction(circuit)

        # Born rule: |ψ|²
        probabilities = np.abs(state.data) ** 2

        return probabilities

    def mmd_loss(self, target_distribution, generated_samples):
        """Maximum Mean Discrepancy loss."""
        # Simple implementation using histograms
        n_bins = 2**self.n_qubits

        # Target histogram
        target_hist, _ = np.histogram(
            target_distribution, bins=n_bins, range=(0, n_bins), density=True
        )

        # Generated histogram
        gen_hist, _ = np.histogram(
            generated_samples, bins=n_bins, range=(0, n_bins), density=True
        )

        # MMD approximation
        mmd = np.mean((target_hist - gen_hist) ** 2)

        return mmd

    def kl_divergence(self, target_probs, generated_probs):
        """Compute KL divergence."""
        epsilon = 1e-15
        target_probs = np.clip(target_probs, epsilon, 1.0)
        generated_probs = np.clip(generated_probs, epsilon, 1.0)

        # Ensure same length
        min_len = min(len(target_probs), len(generated_probs))
        target_probs = target_probs[:min_len]
        generated_probs = generated_probs[:min_len]

        kl_div = np.sum(target_probs * np.log(target_probs / generated_probs))

        return kl_div

    def fit(self, target_data, max_iter=100, learning_rate=0.01):
        """Train Born machine to match target distribution."""
        # Initialize parameters
        n_params = 2 * self.n_qubits * self.n_layers
        self.parameters = np.random.uniform(0, 2 * np.pi, n_params)

        self.training_history = []

        # Convert target data to distribution
        n_bins = 2**self.n_qubits
        target_hist, _ = np.histogram(
            target_data, bins=n_bins, range=(0, n_bins), density=True
        )

        if self.verbose:
            print(f"Training Born machine with {n_params} parameters...")

        for iteration in range(max_iter):
            # Generate samples
            generated_samples, _ = self.generate_samples(self.parameters, 1000)

            # Compute loss
            loss = self.mmd_loss(target_data, generated_samples)

            # Simple parameter update (gradient-free)
            best_params = self.parameters.copy()
            best_loss = loss

            for _ in range(5):  # Multiple random updates per iteration
                # Random perturbation
                noise = np.random.normal(0, learning_rate, n_params)
                test_params = self.parameters + noise

                test_samples, _ = self.generate_samples(test_params, 500)
                test_loss = self.mmd_loss(target_data, test_samples)

                if test_loss < best_loss:
                    best_params = test_params
                    best_loss = test_loss

            self.parameters = best_params

            # Store history
            self.training_history.append(
                {
                    "iteration": iteration,
                    "loss": best_loss,
                    "parameters": self.parameters.copy(),
                }
            )

            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {best_loss:.4f}")

            # Decay learning rate
            learning_rate *= 0.995

        return self


class QuantumGAN:
    def __init__(self, n_qubits_gen=3, n_qubits_disc=3, verbose=False):
        self.n_qubits_gen = n_qubits_gen
        self.n_qubits_disc = n_qubits_disc
        self.verbose = verbose
        self.generator_params = None
        self.discriminator_params = None
        self.training_history = []

    def create_generator(self, parameters, noise_input):
        """Create quantum generator circuit."""
        circuit = QuantumCircuit(self.n_qubits_gen)
        param_idx = 0

        # Encode noise input
        for i, noise in enumerate(noise_input[: self.n_qubits_gen]):
            circuit.ry(noise * np.pi, i)

        # Parameterized layers
        for layer in range(2):
            for qubit in range(self.n_qubits_gen):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(parameters):
                    circuit.rz(parameters[param_idx], qubit)
                    param_idx += 1

            # Entangling gates
            for qubit in range(self.n_qubits_gen - 1):
                circuit.cx(qubit, qubit + 1)

        return circuit

    def create_discriminator(self, parameters, data_input):
        """Create quantum discriminator circuit."""
        circuit = QuantumCircuit(self.n_qubits_disc, 1)
        param_idx = 0

        # Encode data input
        for i, data in enumerate(data_input[: self.n_qubits_disc]):
            circuit.ry(data * np.pi, i)

        # Parameterized layers
        for layer in range(2):
            for qubit in range(self.n_qubits_disc):
                if param_idx < len(parameters):
                    circuit.ry(parameters[param_idx], qubit)
                    param_idx += 1
                if param_idx < len(parameters):
                    circuit.rz(parameters[param_idx], qubit)
                    param_idx += 1

            # Entangling gates
            for qubit in range(self.n_qubits_disc - 1):
                circuit.cx(qubit, qubit + 1)

        # Measurement for binary classification
        circuit.measure(0, 0)

        return circuit

    def discriminator_output(self, parameters, data_batch):
        """Get discriminator outputs for batch of data."""
        outputs = []

        for data_point in data_batch:
            circuit = self.create_discriminator(parameters, data_point)

            simulator = AerSimulator()
            job = simulator.run(circuit, shots=100)
            result = job.result()
            counts = result.get_counts()

            # Probability of measuring 1 (real data)
            prob_real = counts.get("1", 0) / 100
            outputs.append(prob_real)

        return np.array(outputs)

    def generator_output(self, parameters, noise_batch):
        """Generate data from noise using quantum generator."""
        generated_data = []

        for noise in noise_batch:
            circuit = self.create_generator(parameters, noise)

            # Get state vector and convert to data
            state = Statevector.from_instruction(circuit)
            amplitudes = np.abs(state.data) ** 2

            # Sample from distribution
            sample_idx = np.random.choice(len(amplitudes), p=amplitudes)

            # Convert to continuous data (simplified)
            generated_point = [
                (sample_idx & 1) * 2 - 1,  # First bit
                ((sample_idx >> 1) & 1) * 2 - 1,  # Second bit
            ]

            generated_data.append(generated_point)

        return np.array(generated_data)

    def train_step(self, real_data, batch_size=10):
        """Single training step for quantum GAN."""
        # Sample mini-batches
        real_batch = real_data[np.random.choice(len(real_data), batch_size)]
        noise_batch = np.random.uniform(-1, 1, (batch_size, self.n_qubits_gen))

        # Generate fake data
        fake_data = self.generator_output(self.generator_params, noise_batch)

        # Train discriminator
        real_outputs = self.discriminator_output(self.discriminator_params, real_batch)
        fake_outputs = self.discriminator_output(self.discriminator_params, fake_data)

        # Discriminator loss (wants to classify correctly)
        disc_loss = -np.mean(np.log(real_outputs + 1e-15)) - np.mean(
            np.log(1 - fake_outputs + 1e-15)
        )

        # Train generator
        # Generator wants discriminator to classify its outputs as real
        gen_outputs = self.discriminator_output(self.discriminator_params, fake_data)
        gen_loss = -np.mean(np.log(gen_outputs + 1e-15))

        return disc_loss, gen_loss

    def fit(self, real_data, max_iter=50):
        """Train quantum GAN."""
        # Initialize parameters
        n_gen_params = (
            4 * self.n_qubits_gen * 2
        )  # 2 layers, 2 params per qubit per layer
        n_disc_params = 4 * self.n_qubits_disc * 2

        self.generator_params = np.random.uniform(0, 2 * np.pi, n_gen_params)
        self.discriminator_params = np.random.uniform(0, 2 * np.pi, n_disc_params)

        self.training_history = []

        if self.verbose:
            print(f"Training Quantum GAN...")
            print(f"Generator parameters: {n_gen_params}")
            print(f"Discriminator parameters: {n_disc_params}")

        for iteration in range(max_iter):
            disc_loss, gen_loss = self.train_step(real_data)

            # Simple parameter updates (in practice would use proper gradients)
            # Update discriminator
            disc_noise = np.random.normal(0, 0.01, len(self.discriminator_params))
            self.discriminator_params += disc_noise

            # Update generator
            gen_noise = np.random.normal(0, 0.01, len(self.generator_params))
            self.generator_params += gen_noise

            # Store history
            self.training_history.append(
                {"iteration": iteration, "disc_loss": disc_loss, "gen_loss": gen_loss}
            )

            if self.verbose and iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: D_loss = {disc_loss:.4f}, G_loss = {gen_loss:.4f}"
                )

        return self


class GenerativeModelAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def compare_models(self, target_data, n_samples=1000):
        """Compare different generative models."""
        results = {}

        # Quantum Born Machine
        if self.verbose:
            print("Training Quantum Born Machine...")

        # Discretize target data for Born machine
        n_qubits = 3
        n_bins = 2**n_qubits
        target_discrete = np.digitize(
            target_data.flatten(),
            bins=np.linspace(target_data.min(), target_data.max(), n_bins),
        )
        target_discrete = np.clip(target_discrete - 1, 0, n_bins - 1)

        born_machine = QuantumBornMachine(n_qubits=n_qubits, verbose=False)
        born_machine.fit(target_discrete, max_iter=50)

        # Generate samples
        born_samples, born_probs = born_machine.generate_samples(
            born_machine.parameters, n_samples
        )

        # Convert back to continuous
        born_continuous = np.interp(
            born_samples,
            range(n_bins),
            np.linspace(target_data.min(), target_data.max(), n_bins),
        )

        results["Quantum Born Machine"] = {
            "samples": born_continuous,
            "training_history": born_machine.training_history,
            "final_loss": (
                born_machine.training_history[-1]["loss"]
                if born_machine.training_history
                else 0
            ),
        }

        # Quantum GAN
        if self.verbose:
            print("Training Quantum GAN...")

        # Normalize target data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        target_normalized = scaler.fit_transform(target_data.reshape(-1, 1)).flatten()

        # Convert to 2D for GAN
        if len(target_normalized) % 2 == 1:
            target_normalized = target_normalized[:-1]

        target_2d = target_normalized.reshape(-1, 2)

        qgan = QuantumGAN(n_qubits_gen=3, n_qubits_disc=3, verbose=False)
        qgan.fit(target_2d, max_iter=30)

        # Generate samples
        noise = np.random.uniform(-1, 1, (n_samples, 3))
        gan_samples_2d = qgan.generator_output(qgan.generator_params, noise)
        gan_samples = gan_samples_2d.flatten()

        # Denormalize
        gan_samples_denorm = scaler.inverse_transform(
            gan_samples.reshape(-1, 1)
        ).flatten()

        results["Quantum GAN"] = {
            "samples": gan_samples_denorm,
            "training_history": qgan.training_history,
            "final_disc_loss": (
                qgan.training_history[-1]["disc_loss"] if qgan.training_history else 0
            ),
            "final_gen_loss": (
                qgan.training_history[-1]["gen_loss"] if qgan.training_history else 0
            ),
        }

        # Classical baseline (simple Gaussian fit)
        classical_mean = np.mean(target_data)
        classical_std = np.std(target_data)
        classical_samples = np.random.normal(classical_mean, classical_std, n_samples)

        results["Classical Gaussian"] = {
            "samples": classical_samples,
            "mean": classical_mean,
            "std": classical_std,
        }

        return results

    def evaluate_quality(self, target_data, generated_samples):
        """Evaluate quality of generated samples."""
        metrics = {}

        # Wasserstein distance
        try:
            wd = wasserstein_distance(
                target_data.flatten(), generated_samples.flatten()
            )
            metrics["wasserstein_distance"] = wd
        except:
            metrics["wasserstein_distance"] = float("inf")

        # Statistical moments
        target_mean = np.mean(target_data)
        gen_mean = np.mean(generated_samples)
        target_std = np.std(target_data)
        gen_std = np.std(generated_samples)

        metrics["mean_error"] = abs(target_mean - gen_mean)
        metrics["std_error"] = abs(target_std - gen_std)

        # Histogram comparison
        bins = 20
        target_hist, _ = np.histogram(target_data, bins=bins, density=True)
        gen_hist, _ = np.histogram(generated_samples, bins=bins, density=True)

        metrics["histogram_mse"] = mean_squared_error(target_hist, gen_hist)

        return metrics

    def visualize_results(self, target_data, comparison_results):
        """Visualize generative model results."""
        fig = plt.figure(figsize=(16, 12))

        # Data distributions
        ax1 = plt.subplot(2, 3, 1)

        ax1.hist(
            target_data.flatten(),
            bins=30,
            alpha=0.7,
            label="Target",
            color="blue",
            density=True,
            edgecolor="black",
        )

        for model_name, results in comparison_results.items():
            if "samples" in results:
                ax1.hist(
                    results["samples"],
                    bins=30,
                    alpha=0.5,
                    label=model_name,
                    density=True,
                )

        ax1.set_xlabel("Value")
        ax1.set_ylabel("Density")
        ax1.set_title("Generated Data Distributions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Training convergence (Born Machine)
        ax2 = plt.subplot(2, 3, 2)

        if "Quantum Born Machine" in comparison_results:
            history = comparison_results["Quantum Born Machine"]["training_history"]
            if history:
                iterations = [h["iteration"] for h in history]
                losses = [h["loss"] for h in history]

                ax2.plot(iterations, losses, "b-", linewidth=2, label="Born Machine")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Loss")
                ax2.set_title("Born Machine Training")
                ax2.grid(True, alpha=0.3)

        # GAN training curves
        ax3 = plt.subplot(2, 3, 3)

        if "Quantum GAN" in comparison_results:
            history = comparison_results["Quantum GAN"]["training_history"]
            if history:
                iterations = [h["iteration"] for h in history]
                disc_losses = [h["disc_loss"] for h in history]
                gen_losses = [h["gen_loss"] for h in history]

                ax3.plot(
                    iterations, disc_losses, "r-", linewidth=2, label="Discriminator"
                )
                ax3.plot(iterations, gen_losses, "g-", linewidth=2, label="Generator")
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Loss")
                ax3.set_title("GAN Training")
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        # Quality metrics comparison
        ax4 = plt.subplot(2, 3, 4)

        model_names = []
        wasserstein_distances = []
        mean_errors = []

        for model_name, results in comparison_results.items():
            if "samples" in results:
                metrics = self.evaluate_quality(target_data, results["samples"])

                model_names.append(model_name)
                wasserstein_distances.append(metrics["wasserstein_distance"])
                mean_errors.append(metrics["mean_error"])

        if model_names:
            x = np.arange(len(model_names))
            width = 0.35

            ax4.bar(x, wasserstein_distances, alpha=0.7, color="purple")
            ax4.set_xlabel("Model")
            ax4.set_ylabel("Wasserstein Distance")
            ax4.set_title("Distribution Distance")
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names, rotation=45, ha="right")
            ax4.grid(True, alpha=0.3)

        # Statistical comparison
        ax5 = plt.subplot(2, 3, 5)

        target_stats = {
            "Mean": np.mean(target_data),
            "Std": np.std(target_data),
            "Skew": 0,  # Simplified
            "Kurt": 0,  # Simplified
        }

        if model_names:
            stats_comparison = {}
            for model_name, results in comparison_results.items():
                if "samples" in results:
                    stats_comparison[model_name] = {
                        "Mean": np.mean(results["samples"]),
                        "Std": np.std(results["samples"]),
                    }

            # Plot mean comparison
            models = list(stats_comparison.keys())
            means = [stats_comparison[m]["Mean"] for m in models]
            target_mean = target_stats["Mean"]

            bars = ax5.bar(
                models, means, alpha=0.7, color=["red", "green", "blue"][: len(models)]
            )
            ax5.axhline(
                y=target_mean,
                color="black",
                linestyle="--",
                label=f"Target: {target_mean:.3f}",
            )

            ax5.set_ylabel("Mean Value")
            ax5.set_title("Statistical Comparison")
            ax5.tick_params(axis="x", rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Summary and insights
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        summary_text = "Generative Models Summary:\n\n"

        # Find best model by Wasserstein distance
        if model_names and wasserstein_distances:
            best_idx = np.argmin(wasserstein_distances)
            best_model = model_names[best_idx]
            best_distance = wasserstein_distances[best_idx]

            summary_text += f"Best Model: {best_model}\n"
            summary_text += f"Wasserstein Dist: {best_distance:.4f}\n\n"

        summary_text += "Model Characteristics:\n\n"
        summary_text += "Born Machine:\n"
        summary_text += "• Learns probability distributions\n"
        summary_text += "• Uses Born rule |ψ|²\n"
        summary_text += "• Good for discrete data\n\n"

        summary_text += "Quantum GAN:\n"
        summary_text += "• Adversarial training\n"
        summary_text += "• Generator vs Discriminator\n"
        summary_text += "• Better for continuous data\n\n"

        summary_text += "Applications:\n"
        summary_text += "• Data augmentation\n"
        summary_text += "• Anomaly detection\n"
        summary_text += "• Distribution learning"

        ax6.text(
            0.1,
            0.9,
            summary_text,
            transform=ax6.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

        plt.tight_layout()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum Generative Models")
    parser.add_argument(
        "--dataset",
        choices=["moons", "circles", "blobs", "gaussian"],
        default="gaussian",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--model", choices=["born", "gan", "both"], default="both")
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 6: Quantum Machine Learning")
    print("Example 5: Quantum Generative Models")
    print("=" * 43)

    try:
        # Generate target dataset
        if args.dataset == "moons":
            X, _ = make_moons(n_samples=args.n_samples, noise=0.1, random_state=42)
            target_data = X.flatten()
        elif args.dataset == "circles":
            X, _ = make_circles(n_samples=args.n_samples, noise=0.1, random_state=42)
            target_data = X.flatten()
        elif args.dataset == "blobs":
            X, _ = make_blobs(
                n_samples=args.n_samples, centers=2, cluster_std=1.0, random_state=42
            )
            target_data = X.flatten()
        else:  # gaussian
            target_data = np.random.normal(0, 1, args.n_samples)

        # Normalize target data
        scaler = StandardScaler()
        target_data = scaler.fit_transform(target_data.reshape(-1, 1)).flatten()

        print(f"\n📊 Target Dataset: {args.dataset}")
        print(f"   Samples: {len(target_data)}")
        print(f"   Mean: {np.mean(target_data):.3f}")
        print(f"   Std: {np.std(target_data):.3f}")
        print(f"   Range: [{np.min(target_data):.3f}, {np.max(target_data):.3f}]")

        # Analyze with generative models
        analyzer = GenerativeModelAnalyzer(verbose=args.verbose)

        print(f"\n🔄 Training generative models...")
        comparison_results = analyzer.compare_models(
            target_data, n_samples=args.n_samples
        )

        print(f"\n📈 Model Comparison Results:")

        for model_name, results in comparison_results.items():
            if "samples" in results:
                # Evaluate quality
                metrics = analyzer.evaluate_quality(target_data, results["samples"])

                print(f"\n   {model_name}:")
                print(
                    f"     Wasserstein distance: {metrics['wasserstein_distance']:.4f}"
                )
                print(f"     Mean error: {metrics['mean_error']:.4f}")
                print(f"     Std error: {metrics['std_error']:.4f}")
                print(f"     Histogram MSE: {metrics['histogram_mse']:.4f}")

                # Model-specific metrics
                if "final_loss" in results:
                    print(f"     Final training loss: {results['final_loss']:.4f}")

                if "final_disc_loss" in results and "final_gen_loss" in results:
                    print(
                        f"     Final discriminator loss: {results['final_disc_loss']:.4f}"
                    )
                    print(f"     Final generator loss: {results['final_gen_loss']:.4f}")

        # Find best model
        best_model = None
        best_distance = float("inf")

        for model_name, results in comparison_results.items():
            if "samples" in results:
                metrics = analyzer.evaluate_quality(target_data, results["samples"])
                if metrics["wasserstein_distance"] < best_distance:
                    best_distance = metrics["wasserstein_distance"]
                    best_model = model_name

        if best_model:
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   Wasserstein distance: {best_distance:.4f}")

        # Training analysis
        print(f"\n📚 Training Analysis:")

        if "Quantum Born Machine" in comparison_results:
            born_history = comparison_results["Quantum Born Machine"][
                "training_history"
            ]
            if born_history:
                initial_loss = born_history[0]["loss"]
                final_loss = born_history[-1]["loss"]
                improvement = (initial_loss - final_loss) / initial_loss * 100
                print(f"   Born Machine improvement: {improvement:.1f}%")

        if "Quantum GAN" in comparison_results:
            gan_history = comparison_results["Quantum GAN"]["training_history"]
            if gan_history:
                print(f"   GAN training steps: {len(gan_history)}")
                final_disc = gan_history[-1]["disc_loss"]
                final_gen = gan_history[-1]["gen_loss"]
                print(f"   Final discriminator loss: {final_disc:.4f}")
                print(f"   Final generator loss: {final_gen:.4f}")

        if args.show_visualization:
            analyzer.visualize_results(target_data, comparison_results)

        print(f"\n📚 Key Insights:")
        print(f"   • Quantum Born machines learn probability distributions using |ψ|²")
        print(f"   • Quantum GANs use adversarial training with quantum circuits")
        print(f"   • Both models can capture complex data distributions")
        print(f"   • Performance depends on circuit expressivity and training")

        print(f"\n🎯 Applications:")
        print(f"   • Data augmentation for quantum datasets")
        print(f"   • Generating synthetic quantum states")
        print(f"   • Distribution learning in quantum feature spaces")
        print(f"   • Anomaly detection in quantum systems")

        print(f"\n✅ Quantum generative model analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
