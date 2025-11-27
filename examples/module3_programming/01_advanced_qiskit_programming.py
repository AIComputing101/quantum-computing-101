#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 3, Example 1
Advanced Qiskit Programming

This example demonstrates advanced Qiskit programming techniques including
custom gates, circuit optimization, and sophisticated quantum program design.

Learning objectives:
- Master advanced circuit construction techniques
- Create custom quantum gates and decompositions
- Optimize circuits for different backends
- Use advanced Qiskit features effectively

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from qiskit import (
    QuantumCircuit,
    ClassicalRegister,
    ClassicalRegister,
    QuantumRegister,
    transpile,
)
from qiskit.circuit import Gate, Parameter, ParameterVector
from qiskit.circuit.library import RYGate, CXGate, RZGate
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.transpiler import CouplingMap, Layout
from qiskit_aer import AerSimulator

# Note: Fake providers have been updated in newer Qiskit versions
try:
    from qiskit.providers.fake_provider import FakeVigo, FakeMontreal
except ImportError:
    # Use generic fake backend for newer Qiskit versions
    try:
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.providers.models import BackendConfiguration

        # Create minimal fake backends for demonstration
        FakeVigo = lambda: AerSimulator()
        FakeMontreal = lambda: AerSimulator()
        print("ℹ️  Using AerSimulator instead of deprecated fake providers")
    except ImportError:
        FakeVigo = lambda: AerSimulator()
        FakeMontreal = lambda: AerSimulator()
        print("ℹ️  Using AerSimulator for hardware simulation")
import time


def demonstrate_custom_gates():
    """Demonstrate creation and use of custom quantum gates."""
    print("=== CUSTOM QUANTUM GATES ===")
    print()

    # Create a custom gate - controlled rotation
    def create_controlled_ry_gate(theta):
        """Create a controlled RY gate."""
        qc = QuantumCircuit(2, name=f"CRY({theta:.2f})")
        qc.cry(theta, 0, 1)
        return qc.to_gate()

    # Create a custom multi-qubit gate - Quantum Fourier Transform on 3 qubits
    def create_qft3_gate():
        """Create a 3-qubit QFT gate."""
        qc = QuantumCircuit(3, name="QFT3")

        # QFT implementation
        qc.h(2)
        qc.cp(np.pi / 2, 1, 2)
        qc.cp(np.pi / 4, 0, 2)
        qc.h(1)
        qc.cp(np.pi / 2, 0, 1)
        qc.h(0)

        # Swap qubits to get correct order
        qc.swap(0, 2)

        return qc.to_gate()

    # Create a parameterized custom gate
    def create_variational_gate(params):
        """Create a variational quantum gate."""
        qc = QuantumCircuit(2, name="VAR")
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 0)
        qc.ry(params[3], 1)
        return qc.to_gate()

    # Demonstrate usage
    main_circuit = QuantumCircuit(4, 4)

    # Add custom gates
    cry_gate = create_controlled_ry_gate(np.pi / 3)
    main_circuit.append(cry_gate, [0, 1])

    qft3_gate = create_qft3_gate()
    main_circuit.append(qft3_gate, [1, 2, 3])

    var_params = [np.pi / 4, np.pi / 6, np.pi / 8, np.pi / 12]
    var_gate = create_variational_gate(var_params)
    main_circuit.append(var_gate, [0, 1])

    print("Circuit with custom gates:")
    print(main_circuit.draw())
    print()

    # Decompose custom gates
    decomposed = main_circuit.decompose()
    print(f"Original circuit depth: {main_circuit.depth()}")
    print(f"Decomposed circuit depth: {decomposed.depth()}")
    print(f"Original gate count: {main_circuit.count_ops()}")
    print(f"Decomposed gate count: {decomposed.count_ops()}")
    print()

    return main_circuit, decomposed


def demonstrate_parameterized_circuits():
    """Demonstrate parameterized quantum circuits."""
    print("=== PARAMETERIZED QUANTUM CIRCUITS ===")
    print()

    # Create parameterized circuit
    n_qubits = 3
    n_layers = 2

    # Define parameters
    params = ParameterVector("θ", n_qubits * n_layers * 2)

    # Build parameterized circuit
    qc = QuantumCircuit(n_qubits)

    param_idx = 0
    for layer in range(n_layers):
        # Rotation layer
        for qubit in range(n_qubits):
            qc.ry(params[param_idx], qubit)
            param_idx += 1

        # Entangling layer
        for qubit in range(n_qubits):
            qc.rz(params[param_idx], qubit)
            param_idx += 1
            if qubit < n_qubits - 1:
                qc.cx(qubit, qubit + 1)

        # Add final entangling connection
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    print("Parameterized circuit structure:")
    print(f"Parameters: {qc.parameters}")
    print(f"Number of parameters: {qc.num_parameters}")
    print(qc.draw())
    print()

    # Bind parameters and execute
    parameter_values = np.random.uniform(0, 2 * np.pi, qc.num_parameters)
    parameter_dict = dict(zip(params, parameter_values))

    # Handle different Qiskit versions for parameter binding
    try:
        bound_circuit = qc.bind_parameters(parameter_dict)
    except AttributeError:
        # For newer Qiskit versions
        bound_circuit = qc.assign_parameters(parameter_dict)

    print("Circuit with bound parameters:")
    print(f"Depth: {bound_circuit.depth()}")
    print(f"Gate count: {bound_circuit.count_ops()}")
    print()

    # Analyze different parameter bindings
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, scale in enumerate([0.1, 1.0, 10.0]):
        param_vals = np.random.uniform(0, scale, qc.num_parameters)
        param_dict = dict(zip(params, param_vals))
        # Handle different Qiskit versions for parameter binding
        try:
            circuit = qc.bind_parameters(param_dict)
        except AttributeError:
            # For newer Qiskit versions
            circuit = qc.assign_parameters(param_dict)

        # Get statevector
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities()

        # Plot probability distribution
        axes[i].bar(range(len(probabilities)), probabilities, alpha=0.7)
        axes[i].set_title(f"Parameter scale: {scale}")
        axes[i].set_xlabel("Computational basis state")
        axes[i].set_ylabel("Probability")
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("module3_01_parameterized_circuits.png", dpi=300, bbox_inches="tight")
    plt.close()

    return qc, parameter_dict


def demonstrate_circuit_optimization():
    """Demonstrate circuit optimization techniques."""
    print("=== CIRCUIT OPTIMIZATION ===")
    print()

    # Create a circuit with optimization opportunities
    qc = QuantumCircuit(4)

    # Add redundant operations
    qc.h(0)
    qc.x(0)
    qc.x(0)  # Double X = Identity
    qc.h(0)  # H-X-X-H = H-H = I

    qc.cx(0, 1)
    qc.cx(0, 1)  # Double CNOT = Identity

    qc.z(2)
    qc.z(2)  # Double Z = Identity

    qc.h(3)
    qc.z(3)
    qc.h(3)  # H-Z-H = X

    print("Original circuit (with redundancies):")
    print(qc.draw())
    print(f"Depth: {qc.depth()}, Gates: {sum(qc.count_ops().values())}")
    print()

    # Optimization level comparison
    fake_backend = FakeVigo()

    optimization_levels = [0, 1, 2, 3]
    optimized_circuits = {}

    for level in optimization_levels:
        optimized = transpile(qc, backend=fake_backend, optimization_level=level)
        optimized_circuits[level] = optimized

        print(f"Optimization level {level}:")
        print(f"  Depth: {optimized.depth()}")
        print(f"  Gates: {sum(optimized.count_ops().values())}")
        print(f"  Gate types: {optimized.count_ops()}")
        print()

    # Visualize optimization effects
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, (level, circuit) in enumerate(optimized_circuits.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Draw circuit
        circuit_drawer(
            circuit, output="mpl", ax=ax, style={"backgroundcolor": "#EEEEEE"}
        )
        ax.set_title(
            f"Optimization Level {level}\nDepth: {circuit.depth()}, Gates: {sum(circuit.count_ops().values())}"
        )

    plt.tight_layout()
    plt.savefig("module3_01_circuit_optimization.png", dpi=300, bbox_inches="tight")
    plt.close()

    return optimized_circuits


def demonstrate_backend_specific_programming():
    """Demonstrate programming for specific quantum backends."""
    print("=== BACKEND-SPECIFIC PROGRAMMING ===")
    print()

    # Create a circuit that will be transpiled for different backends
    qc = QuantumCircuit(5)

    # Create a circuit with specific structure
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)

    qc.ry(np.pi / 4, 2)
    qc.rz(np.pi / 3, 3)

    for i in range(3, 0, -1):
        qc.cx(i, i - 1)

    print("Original circuit:")
    print(qc.draw())
    print()

    # Different backend configurations
    backends_info = {
        "Simulator": {
            "backend": AerSimulator(),
            "coupling_map": None,
            "basis_gates": None,
        }
    }

    # Add fake backends if available
    try:
        fake_vigo = FakeVigo()
        backends_info["FakeVigo (5 qubits)"] = {
            "backend": fake_vigo,
            "coupling_map": getattr(fake_vigo, "coupling_map", None),
            "basis_gates": getattr(
                fake_vigo, "basis_gates", ["cx", "id", "rz", "sx", "x"]
            ),
        }
    except:
        print("ℹ️  FakeVigo not available, using default configuration")

    try:
        fake_montreal = FakeMontreal()
        backends_info["FakeMontreal (27 qubits)"] = {
            "backend": fake_montreal,
            "coupling_map": getattr(fake_montreal, "coupling_map", None),
            "basis_gates": getattr(
                fake_montreal, "basis_gates", ["cx", "id", "rz", "sx", "x"]
            ),
        }
    except:
        print("ℹ️  FakeMontreal not available, using default configuration")

    transpiled_circuits = {}

    for name, info in backends_info.items():
        print(f"Transpiling for {name}:")

        # Transpile circuit
        transpiled = transpile(qc, backend=info["backend"], optimization_level=2)

        transpiled_circuits[name] = transpiled

        print(f"  Original depth: {qc.depth()}")
        print(f"  Transpiled depth: {transpiled.depth()}")
        print(f"  Original gates: {sum(qc.count_ops().values())}")
        print(f"  Transpiled gates: {sum(transpiled.count_ops().values())}")

        if info["coupling_map"]:
            print(f"  Coupling map size: {len(info['coupling_map'].get_edges())} edges")
            print(
                f"  Physical qubits used: {set(transpiled.layout.get_physical_bits().values())}"
            )

        print(
            f"  Basis gates: {info['basis_gates'][:5] if info['basis_gates'] else 'All gates'}..."
        )
        print()

    return transpiled_circuits


def demonstrate_advanced_measurements():
    """Demonstrate advanced measurement techniques."""
    print("=== ADVANCED MEASUREMENT TECHNIQUES ===")
    print()

    # Conditional measurements
    qc_conditional = QuantumCircuit(3, 3)

    # Create Bell state
    qc_conditional.h(0)
    qc_conditional.cx(0, 1)

    # Measure first qubit
    qc_conditional.measure(0, 0)

    # Conditional operation based on measurement (using if_test for newer Qiskit)
    try:
        # Try newer Qiskit API with dynamic circuits
        with qc_conditional.if_test((qc_conditional.cregs[0], 1)):
            qc_conditional.x(2)
    except AttributeError:
        # Fallback for older Qiskit versions
        qc_conditional.x(2).c_if(qc_conditional.cregs[0], 1)

    # Measure remaining qubits
    qc_conditional.measure(1, 1)
    qc_conditional.measure(2, 2)

    print("Conditional measurement circuit:")
    print(qc_conditional.draw())
    print()

    # Mid-circuit measurements with reset
    qc_midcircuit = QuantumCircuit(2, 2)

    # Create superposition
    qc_midcircuit.h(0)
    qc_midcircuit.cx(0, 1)

    # Mid-circuit measurement and reset
    qc_midcircuit.measure(0, 0)
    qc_midcircuit.reset(0)

    # Continue computation
    qc_midcircuit.h(0)
    qc_midcircuit.cx(0, 1)
    qc_midcircuit.measure_all()

    print("Mid-circuit measurement with reset:")
    print(qc_midcircuit.draw())
    print()

    # Partial measurements (measuring subset of qubits)
    qc_partial = QuantumCircuit(4, 2)

    # Create GHZ state
    qc_partial.h(0)
    for i in range(3):
        qc_partial.cx(i, i + 1)

    # Measure only first two qubits
    qc_partial.measure([0, 1], [0, 1])

    print("Partial measurement (GHZ state):")
    print(qc_partial.draw())
    print()

    # Execute and analyze results
    simulator = AerSimulator()
    shots = 1000

    circuits = {
        "Conditional": qc_conditional,
        "Mid-circuit": qc_midcircuit,
        "Partial": qc_partial,
    }

    results = {}
    for name, circuit in circuits.items():
        job = simulator.run(transpile(circuit, simulator), shots=shots)
        result = job.result()
        counts = result.get_counts()
        results[name] = counts

        print(f"{name} measurement results:")
        for outcome, count in sorted(counts.items()):
            percentage = (count / shots) * 100
            print(f"  {outcome}: {count} ({percentage:.1f}%)")
        print()

    # Visualize results
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for i, (name, counts) in enumerate(results.items()):
        # plot_histogram no longer accepts ax parameter in Qiskit 2.x
        try:
            # Use matplotlib bar plot instead
            axes[i].bar(list(counts.keys()), list(counts.values()))
            axes[i].set_xlabel("Measurement Outcome")
            axes[i].set_ylabel("Counts")
        except Exception as e:
            print(f"⚠️ Could not create histogram: {e}")
        axes[i].set_title(f"{name} Measurements")

    plt.tight_layout()
    plt.savefig("module3_01_advanced_measurements.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results


def benchmark_circuit_construction():
    """Benchmark different circuit construction approaches."""
    print("=== CIRCUIT CONSTRUCTION BENCHMARKING ===")
    print()

    n_qubits = 10
    n_layers = 5

    def method1_sequential():
        """Sequential gate addition."""
        qc = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                qc.h(qubit)
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)
        return qc

    def method2_batch():
        """Batch gate addition."""
        qc = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            # Add all H gates at once
            for qubit in range(n_qubits):
                qc.h(qubit)
            # Add all CNOT gates at once
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)
        return qc

    def method3_compose():
        """Circuit composition."""
        # Create basic layer
        layer = QuantumCircuit(n_qubits)
        for qubit in range(n_qubits):
            layer.h(qubit)
        for qubit in range(n_qubits - 1):
            layer.cx(qubit, qubit + 1)

        # Compose layers
        qc = QuantumCircuit(n_qubits)
        for _ in range(n_layers):
            qc = qc.compose(layer)
        return qc

    methods = {
        "Sequential": method1_sequential,
        "Batch": method2_batch,
        "Compose": method3_compose,
    }

    # Benchmark construction times
    times = {}
    circuits = {}

    for name, method in methods.items():
        start_time = time.time()
        circuit = method()
        end_time = time.time()

        times[name] = end_time - start_time
        circuits[name] = circuit

        print(f"{name} method:")
        print(f"  Construction time: {times[name]:.4f} seconds")
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Gate count: {sum(circuit.count_ops().values())}")
        print()

    # Visualize benchmark results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Construction time comparison
    methods_list = list(times.keys())
    times_list = list(times.values())

    bars1 = ax1.bar(methods_list, times_list, alpha=0.7, color=["blue", "green", "red"])
    ax1.set_ylabel("Construction Time (seconds)")
    ax1.set_title("Circuit Construction Time")
    ax1.grid(True, alpha=0.3)

    # Add time values on bars
    for bar, time_val in zip(bars1, times_list):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{time_val:.4f}s",
            ha="center",
            va="bottom",
        )

    # Circuit properties comparison
    depths = [circuits[name].depth() for name in methods_list]
    gate_counts = [sum(circuits[name].count_ops().values()) for name in methods_list]

    x = np.arange(len(methods_list))
    width = 0.35

    bars2a = ax2.bar(x - width / 2, depths, width, label="Depth", alpha=0.7)
    bars2b = ax2.bar(x + width / 2, gate_counts, width, label="Gate Count", alpha=0.7)

    ax2.set_xlabel("Method")
    ax2.set_ylabel("Count")
    ax2.set_title("Circuit Properties")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("module3_01_construction_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()

    return times, circuits


def main():
    """Main function to run all demonstrations."""
    parser = argparse.ArgumentParser(description="Advanced Qiskit Programming Demo")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true", help="Skip time-consuming benchmarks"
    )
    args = parser.parse_args()

    print("🚀 Quantum Computing 101 - Module 3, Example 1")
    print("Advanced Qiskit Programming")
    print("=" * 50)
    print()

    try:
        # Custom gates
        original, decomposed = demonstrate_custom_gates()

        # Parameterized circuits
        param_circuit, param_dict = demonstrate_parameterized_circuits()

        # Circuit optimization
        optimized_circuits = demonstrate_circuit_optimization()

        # Backend-specific programming
        transpiled_circuits = demonstrate_backend_specific_programming()

        # Advanced measurements
        measurement_results = demonstrate_advanced_measurements()

        # Benchmarking (optional)
        if not args.skip_benchmarks:
            benchmark_times, benchmark_circuits = benchmark_circuit_construction()

        print("✅ Example completed successfully!")
        print()
        print("Generated files:")
        print("• module3_01_parameterized_circuits.png - Parameter analysis")
        print("• module3_01_circuit_optimization.png - Optimization comparison")
        print("• module3_01_advanced_measurements.png - Measurement strategies")
        if not args.skip_benchmarks:
            print("• module3_01_construction_benchmark.png - Performance benchmarks")
        print()
        print("🎯 Key takeaways:")
        print("• Custom gates enable modular circuit design")
        print("• Parameterized circuits support variational algorithms")
        print("• Circuit optimization reduces depth and gate count")
        print("• Backend-specific transpilation is crucial for real hardware")
        print("• Advanced measurements enable complex quantum protocols")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
