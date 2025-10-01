"""
Quantum Computing 101 - Visualization Utilities

This module provides enhanced visualization tools for quantum computing concepts,
making it easier to create consistent and educational plots across all examples.

Author: Quantum Computing 101 Course
License: MIT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from qiskit.visualization import plot_bloch_multivector, plot_histogram, circuit_drawer
from qiskit.quantum_info import Statevector
import seaborn as sns


def setup_plot_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use("default")
    sns.set_palette("husl")

    # Custom parameters
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2,
            "grid.alpha": 0.3,
        }
    )


def plot_bloch_comparison(
    states, labels, title="Qubit State Comparison", figsize=(15, 5)
):
    """Plot multiple qubit states on Bloch spheres for comparison.

    Args:
        states: List of Statevector objects or quantum circuits
        labels: List of labels for each state
        title: Overall title for the plot
        figsize: Figure size tuple

    Returns:
        List of figure objects (one per state)
    """
    setup_plot_style()

    figures = []

    for i, (state, label) in enumerate(zip(states, labels)):
        # Convert to Statevector if needed
        if hasattr(state, "data"):  # Quantum circuit
            state = Statevector.from_instruction(state)

        # Create individual figure for each Bloch sphere
        # plot_bloch_multivector no longer accepts ax parameter
        fig = plot_bloch_multivector(state, title=label, figsize=(5, 5))
        figures.append(fig)

    return figures


def plot_measurement_comparison(
    results_dict, title="Measurement Results", figsize=(12, 4)
):
    """Plot measurement results for multiple experiments.

    Args:
        results_dict: Dictionary with experiment names as keys, counts as values
        title: Plot title
        figsize: Figure size

    Returns:
        Figure and axes objects
    """
    setup_plot_style()

    n_experiments = len(results_dict)
    fig, axes = plt.subplots(1, n_experiments, figsize=figsize)
    if n_experiments == 1:
        axes = [axes]

    for i, (name, counts) in enumerate(results_dict.items()):
        # plot_histogram no longer accepts ax parameter in newer Qiskit
        if n_experiments == 1:
            fig_hist = plot_histogram(counts, title=name)
            return fig_hist, None
        else:
            # For multiple plots, create individual histograms
            axes[i].bar(counts.keys(), counts.values())
            axes[i].set_title(name, fontsize=11)
            axes[i].set_xlabel("Measurement Outcome")
            axes[i].set_ylabel("Counts")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    return fig, axes


def plot_probability_evolution(
    states_over_time, labels=None, title="Quantum State Evolution"
):
    """Plot how quantum state probabilities evolve over time/steps.

    Args:
        states_over_time: List of Statevector objects
        labels: List of step labels
        title: Plot title

    Returns:
        Figure and axes objects
    """
    setup_plot_style()

    if labels is None:
        labels = [f"Step {i}" for i in range(len(states_over_time))]

    # Extract probabilities
    prob_0 = []
    prob_1 = []

    for state in states_over_time:
        if hasattr(state, "data"):  # Quantum circuit
            state = Statevector.from_instruction(state)

        prob_0.append(abs(state[0]) ** 2)
        prob_1.append(abs(state[1]) ** 2)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(states_over_time))
    ax.plot(x, prob_0, "bo-", label="P(|0⟩)", markersize=8, linewidth=2)
    ax.plot(x, prob_1, "ro-", label="P(|1⟩)", markersize=8, linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Probability")
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    return fig, ax


def plot_quantum_circuit_with_explanation(circuit, explanation_text, figsize=(14, 8)):
    """Plot a quantum circuit with explanatory text.

    Args:
        circuit: Qiskit QuantumCircuit
        explanation_text: Text explaining the circuit
        figsize: Figure size

    Returns:
        Figure object
    """
    setup_plot_style()

    fig = plt.figure(figsize=figsize)

    # Circuit diagram (top 2/3)
    ax_circuit = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    # circuit_drawer no longer accepts ax parameter, create separate figure and copy
    circuit_fig = circuit_drawer(
        circuit, output="mpl", style={"backgroundcolor": "#EEEEEE"}
    )
    ax_circuit.set_title(
        f"Quantum Circuit (Depth: {circuit.depth()}, Qubits: {circuit.num_qubits})"
    )
    # Note: In modern Qiskit, you might need to handle circuit drawing differently

    # Explanation text (bottom 1/3)
    ax_text = plt.subplot2grid((3, 1), (2, 0))
    ax_text.text(
        0.05,
        0.5,
        explanation_text,
        transform=ax_text.transAxes,
        fontsize=11,
        verticalalignment="center",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )
    ax_text.axis("off")

    plt.tight_layout()

    return fig


def plot_correlation_matrix(correlations, labels, title="Quantum Correlations"):
    """Plot correlation matrix for quantum measurements.

    Args:
        correlations: 2D array of correlation values
        labels: List of measurement labels
        title: Plot title

    Returns:
        Figure and axes objects
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation Coefficient")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(
                j,
                i,
                f"{correlations[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    return fig, ax


def plot_quantum_vs_classical(
    quantum_data,
    classical_data,
    metric_name="Value",
    title="Quantum vs Classical Comparison",
):
    """Compare quantum and classical results side by side.

    Args:
        quantum_data: Array of quantum results
        classical_data: Array of classical results
        metric_name: Name of the metric being compared
        title: Plot title

    Returns:
        Figure and axes objects
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Quantum histogram
    axes[0].hist(quantum_data, bins=20, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_title("Quantum Results")
    axes[0].set_xlabel(metric_name)
    axes[0].set_ylabel("Frequency")

    # Classical histogram
    axes[1].hist(classical_data, bins=20, alpha=0.7, color="red", edgecolor="black")
    axes[1].set_title("Classical Results")
    axes[1].set_xlabel(metric_name)
    axes[1].set_ylabel("Frequency")

    # Side-by-side comparison
    axes[2].boxplot([quantum_data, classical_data], labels=["Quantum", "Classical"])
    axes[2].set_title("Distribution Comparison")
    axes[2].set_ylabel(metric_name)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig, axes


def create_educational_animation_frames(
    states, frame_labels, save_prefix="animation_frame"
):
    """Create frames for educational animations of quantum state evolution.

    Args:
        states: List of quantum states
        frame_labels: Labels for each frame
        save_prefix: Prefix for saved frame files

    Returns:
        List of saved frame filenames
    """
    setup_plot_style()

    frame_files = []

    for i, (state, label) in enumerate(zip(states, frame_labels)):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Convert to Statevector if needed
        if hasattr(state, "data"):
            state = Statevector.from_instruction(state)

        # Plot Bloch sphere
        # plot_bloch_multivector no longer accepts ax parameter
        plt.close()  # Close the subplot figure
        bloch_fig = plot_bloch_multivector(state, title=f"{label}\nStep {i+1}")

        # Save the Bloch sphere figure directly

        # Save frame
        filename = f"{save_prefix}_{i:03d}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        frame_files.append(filename)
        plt.close()

    return frame_files


def plot_algorithm_complexity(
    problem_sizes,
    quantum_times,
    classical_times,
    algorithm_name="Algorithm",
    log_scale=True,
):
    """Plot algorithm complexity comparison between quantum and classical.

    Args:
        problem_sizes: Array of problem sizes
        quantum_times: Array of quantum algorithm times/complexities
        classical_times: Array of classical algorithm times/complexities
        algorithm_name: Name of the algorithm
        log_scale: Whether to use log scale

    Returns:
        Figure and axes objects
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        problem_sizes, quantum_times, "b-o", label="Quantum", linewidth=3, markersize=8
    )
    ax.plot(
        problem_sizes,
        classical_times,
        "r-s",
        label="Classical",
        linewidth=3,
        markersize=8,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")

    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Time/Complexity")
    ax.set_title(f"{algorithm_name} - Quantum vs Classical Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax


def save_figure_with_metadata(fig, filename, metadata=None):
    """Save figure with educational metadata.

    Args:
        fig: Matplotlib figure object
        filename: Output filename
        metadata: Dictionary of metadata to include
    """
    if metadata is None:
        metadata = {}

    # Add default metadata
    default_metadata = {
        "Creator": "Quantum Computing 101 Course",
        "Subject": "Quantum Computing Education",
        "Title": filename.split(".")[0].replace("_", " ").title(),
    }

    combined_metadata = {**default_metadata, **metadata}

    # Save with metadata
    fig.savefig(filename, dpi=300, bbox_inches="tight", metadata=combined_metadata)

    print(f"📊 Saved visualization: {filename}")


# Example usage and testing
if __name__ == "__main__":
    print("🎨 Quantum Computing 101 - Visualization Utilities")
    print("This module provides visualization tools for quantum computing education.")
    print()

    # Test basic functionality
    setup_plot_style()
    print("✅ Plot style configured")

    # Create a simple test plot
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    # Test states
    qc1 = QuantumCircuit(1)
    qc2 = QuantumCircuit(1)
    qc2.h(0)

    states = [Statevector.from_instruction(qc1), Statevector.from_instruction(qc2)]
    labels = ["|0⟩", "|+⟩"]

    fig, axes = plot_bloch_comparison(states, labels, "Test Visualization")
    save_figure_with_metadata(
        fig,
        "test_visualization.png",
        {"Description": "Test of visualization utilities"},
    )
    plt.close()

    print("✅ Visualization utilities test completed")
