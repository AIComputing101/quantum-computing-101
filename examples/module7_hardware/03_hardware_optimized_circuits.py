#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms
Example 3: Hardware-Optimized Circuits

Implementation of hardware-specific circuit optimization and transpilation strategies.
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
from qiskit.transpiler import PassManager, Layout

# Handle different Qiskit versions for transpiler passes
USING_DUMMY_PASSES = False
try:
    from qiskit.transpiler.passes import (
        BasicSwap,
        LookaheadSwap,
        SabreSwap,
        Optimize1qGates,
        CommutativeCancellation,
        CXCancellation,
        OptimizeSwapBeforeMeasure,
    )
except ImportError:
    # For newer Qiskit versions, some passes may be in different locations
    try:
        from qiskit.transpiler.passes.routing import BasicSwap, LookaheadSwap, SabreSwap
        from qiskit.transpiler.passes.optimization import (
            Optimize1qGates,
            CommutativeCancellation,
            CXCancellation,
            OptimizeSwapBeforeMeasure,
        )
    except ImportError:
        print("ℹ️  Some transpiler passes not available, using basic transpile instead")

        # Create dummy classes for educational purposes
        class DummyPass:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, dag):
                return dag

        BasicSwap = DummyPass
        LookaheadSwap = DummyPass
        SabreSwap = DummyPass
        Optimize1qGates = DummyPass
        CommutativeCancellation = DummyPass
        CXCancellation = DummyPass
        OptimizeSwapBeforeMeasure = DummyPass

        # Flag that dummy passes are being used
        USING_DUMMY_PASSES = True
from qiskit.circuit.library import QFT, QuantumVolume

# Handle FakeBackend import for different Qiskit versions
try:
    from qiskit.providers.fake_provider import FakeBackend
except ImportError:
    print("ℹ️  FakeBackend not available, using mock backend implementation")

    # Create a simple mock backend class
    class FakeBackend:
        def __init__(self):
            pass


from qiskit_aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity

# Handle visualization imports for different Qiskit versions
try:
    from qiskit.visualization import plot_circuit_layout, plot_gate_map
except ImportError:
    print("ℹ️  Some visualization functions not available")

    # Create dummy functions for educational purposes
    def plot_circuit_layout(*args, **kwargs):
        print("Circuit layout plotting not available")
        return None

    def plot_gate_map(*args, **kwargs):
        print("Gate map plotting not available")
        return None


try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("ℹ️  networkx not available, some topology analysis features will be limited")

import warnings

warnings.filterwarnings("ignore")


class HardwareOptimizer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.backends = {}
        self.coupling_maps = {}

    def create_mock_backends(self):
        """Create mock backends with different architectures."""
        backends = {}

        # Linear coupling topology
        linear_coupling = [(i, i + 1) for i in range(4)]
        backends["linear_5q"] = {
            "name": "Linear 5-qubit",
            "n_qubits": 5,
            "coupling_map": linear_coupling,
            "basis_gates": ["cx", "id", "rz", "sx", "x"],
            "gate_errors": {"cx": 0.01, "sx": 0.001, "rz": 0.0001},
            "readout_errors": [0.01] * 5,
            "coherence_times": {"T1": [50e-6] * 5, "T2": [30e-6] * 5},
        }

        # Grid coupling topology
        grid_coupling = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
        ]
        backends["grid_10q"] = {
            "name": "Grid 10-qubit",
            "n_qubits": 10,
            "coupling_map": grid_coupling,
            "basis_gates": ["cx", "id", "rz", "sx", "x"],
            "gate_errors": {"cx": 0.02, "sx": 0.002, "rz": 0.0002},
            "readout_errors": [0.02] * 10,
            "coherence_times": {"T1": [40e-6] * 10, "T2": [25e-6] * 10},
        }

        # Heavy-hex coupling (IBM-like)
        hex_coupling = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 4),
            (4, 7),
            (7, 10),
            (10, 12),
            (2, 5),
            (5, 8),
            (8, 11),
            (11, 14),
            (3, 6),
            (6, 9),
            (9, 13),
            (13, 15),
            (4, 5),
            (5, 6),
            (7, 8),
            (8, 9),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
        ]
        backends["heavy_hex_16q"] = {
            "name": "Heavy-Hex 16-qubit",
            "n_qubits": 16,
            "coupling_map": hex_coupling,
            "basis_gates": ["cx", "id", "rz", "sx", "x"],
            "gate_errors": {"cx": 0.015, "sx": 0.0015, "rz": 0.00015},
            "readout_errors": [0.015] * 16,
            "coherence_times": {"T1": [60e-6] * 16, "T2": [40e-6] * 16},
        }

        # All-to-all (IonQ-like)
        all_to_all_coupling = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        backends["all_to_all_5q"] = {
            "name": "All-to-All 5-qubit",
            "n_qubits": 5,
            "coupling_map": all_to_all_coupling,
            "basis_gates": ["rxx", "ry", "rz"],
            "gate_errors": {"rxx": 0.005, "ry": 0.0005, "rz": 0.0001},
            "readout_errors": [0.005] * 5,
            "coherence_times": {"T1": [100e-6] * 5, "T2": [80e-6] * 5},
        }

        self.backends = backends

        # Store coupling maps for analysis
        for name, backend in backends.items():
            self.coupling_maps[name] = backend["coupling_map"]

        return backends

    def analyze_circuit_requirements(self, circuit):
        """Analyze circuit requirements for hardware optimization."""
        analysis = {
            "n_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "gate_counts": {},
            "two_qubit_gates": [],
            "qubit_connectivity": set(),
            "critical_path_length": 0,
        }

        # Count gates and analyze connectivity
        for instruction in circuit.data:
            gate_name = instruction[0].name
            # Handle different Qiskit versions for qubit indexing
            try:
                qubits = [q.index for q in instruction[1]]
            except AttributeError:
                # For newer Qiskit versions, use find_bit
                qubits = [circuit.find_bit(q).index for q in instruction[1]]

            # Count gates
            if gate_name in analysis["gate_counts"]:
                analysis["gate_counts"][gate_name] += 1
            else:
                analysis["gate_counts"][gate_name] = 1

            # Track two-qubit gates
            if len(qubits) == 2:
                analysis["two_qubit_gates"].append(tuple(sorted(qubits)))
                analysis["qubit_connectivity"].add(tuple(sorted(qubits)))

        # Required connectivity graph
        required_edges = list(analysis["qubit_connectivity"])
        analysis["required_connectivity"] = required_edges
        analysis["connectivity_degree"] = len(required_edges)

        return analysis

    def optimize_for_backend(self, circuit, backend_name, optimization_level=3):
        """
        Optimize circuit for specific backend architecture.
        
        Key Concepts - Hardware Optimization:
        -------------------------------------
        1. Transpilation: Convert logical circuit → physical gates
        2. Routing: Map qubits to hardware topology (add SWAPs if needed)
        3. Optimization: Reduce depth/gates while preserving function
        
        Hardware Constraints:
        - Limited connectivity (not all qubit pairs connected)
        - Native gate set (only certain gates implemented)
        - Noise characteristics (gate fidelities vary)
        
        Goal: Minimize circuit depth & gate count for real hardware
        """
        if backend_name not in self.backends:
            raise ValueError(f"Backend {backend_name} not available")

        backend_info = self.backends[backend_name]
        coupling_map = backend_info["coupling_map"]
        basis_gates = backend_info["basis_gates"]

        optimization_results = {}

        # Original circuit analysis
        original_analysis = self.analyze_circuit_requirements(circuit)

        # Create AerSimulator with backend properties
        simulator = AerSimulator()

        # Transpile with different strategies
        strategies = {
            "basic": {"routing_method": "basic", "optimization_level": 1},
            "stochastic": {"routing_method": "stochastic", "optimization_level": 2},
            "sabre": {"routing_method": "sabre", "optimization_level": 3},
            "lookahead": {"routing_method": "lookahead", "optimization_level": 2},
        }

        for strategy_name, transpile_options in strategies.items():
            try:
                # Handle different Qiskit versions for transpile
                from qiskit.transpiler import CouplingMap

                if isinstance(coupling_map, list):
                    coupling_map_obj = CouplingMap(coupling_map)
                else:
                    coupling_map_obj = coupling_map

                transpiled = transpile(
                    circuit,
                    coupling_map=coupling_map_obj,
                    basis_gates=basis_gates,
                    **transpile_options,
                )

                # Analyze transpiled circuit
                transpiled_analysis = self.analyze_circuit_requirements(transpiled)

                # Calculate optimization metrics
                depth_reduction = (
                    original_analysis["depth"] - transpiled_analysis["depth"]
                ) / original_analysis["depth"]
                gate_count_original = sum(original_analysis["gate_counts"].values())
                gate_count_transpiled = sum(transpiled_analysis["gate_counts"].values())
                gate_overhead = (
                    gate_count_transpiled - gate_count_original
                ) / gate_count_original

                optimization_results[strategy_name] = {
                    "circuit": transpiled,
                    "analysis": transpiled_analysis,
                    "depth_reduction": depth_reduction,
                    "gate_overhead": gate_overhead,
                    "success": True,
                }

                if self.verbose:
                    print(
                        f"   {strategy_name}: depth {transpiled_analysis['depth']} "
                        f"(reduction: {depth_reduction:.1%}), "
                        f"gates {gate_count_transpiled} (overhead: {gate_overhead:.1%})"
                    )

            except Exception as e:
                optimization_results[strategy_name] = {
                    "success": False,
                    "error": str(e),
                }
                if self.verbose:
                    print(f"   {strategy_name}: Failed - {e}")

        # Find best strategy
        successful_strategies = {
            k: v for k, v in optimization_results.items() if v["success"]
        }

        if successful_strategies:
            # Score based on depth reduction and gate overhead
            best_strategy = None
            best_score = float("-inf")

            for strategy_name, results in successful_strategies.items():
                score = results["depth_reduction"] - 0.1 * results["gate_overhead"]
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name

            optimization_results["best_strategy"] = best_strategy
            optimization_results["best_circuit"] = successful_strategies[best_strategy][
                "circuit"
            ]

        optimization_results["original_analysis"] = original_analysis
        optimization_results["backend_info"] = backend_info

        return optimization_results

    def compare_backend_suitability(self, circuit):
        """Compare circuit suitability across different backend architectures."""
        circuit_analysis = self.analyze_circuit_requirements(circuit)
        suitability_scores = {}

        for backend_name, backend_info in self.backends.items():
            score = 0
            details = {}

            # Qubit count compatibility
            if circuit_analysis["n_qubits"] <= backend_info["n_qubits"]:
                score += 20
                details["qubit_count"] = "compatible"
            else:
                score -= 50
                details["qubit_count"] = "insufficient"

            # Connectivity analysis
            required_connections = set(circuit_analysis["required_connectivity"])
            available_connections = set(
                tuple(sorted(edge)) for edge in backend_info["coupling_map"]
            )

            if required_connections.issubset(available_connections):
                score += 30
                details["connectivity"] = "native"
            else:
                missing_connections = required_connections - available_connections
                penalty = len(missing_connections) * 5
                score -= penalty
                details["connectivity"] = f"{len(missing_connections)} swaps needed"

            # Gate set compatibility
            circuit_gates = set(circuit_analysis["gate_counts"].keys())
            backend_gates = set(backend_info["basis_gates"])

            if circuit_gates.issubset(backend_gates):
                score += 20
                details["gate_set"] = "native"
            else:
                non_native_gates = circuit_gates - backend_gates
                score -= len(non_native_gates) * 3
                details["gate_set"] = (
                    f"{len(non_native_gates)} gates need decomposition"
                )

            # Estimated fidelity based on gate errors
            gate_errors = backend_info.get("gate_errors", {})
            estimated_fidelity = 1.0

            for gate, count in circuit_analysis["gate_counts"].items():
                if gate in gate_errors:
                    estimated_fidelity *= (1 - gate_errors[gate]) ** count
                else:
                    estimated_fidelity *= 0.99**count  # Default error rate

            score += estimated_fidelity * 20
            details["estimated_fidelity"] = estimated_fidelity

            suitability_scores[backend_name] = {
                "score": score,
                "details": details,
                "backend_info": backend_info,
            }

        # Rank backends
        ranked_backends = sorted(
            suitability_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        return dict(ranked_backends), circuit_analysis

    def create_calibration_aware_optimization(self, circuit, backend_name):
        """Optimize circuit considering calibration data and error rates."""
        if backend_name not in self.backends:
            raise ValueError(f"Backend {backend_name} not available")

        backend_info = self.backends[backend_name]

        # Simulate calibration-aware optimization
        optimization_passes = []

        # Custom pass manager with error handling
        pm = None
        try:
            # Skip PassManager if using dummy passes
            if USING_DUMMY_PASSES:
                raise ImportError("Using dummy passes, fall back to transpile")

            pm = PassManager()

            # Basic optimizations
            pm.append(Optimize1qGates())
            pm.append(CommutativeCancellation())
            pm.append(CXCancellation())

            # Routing with error-aware mapping
            if backend_info["coupling_map"]:
                # Choose best routing method based on backend
                if len(backend_info["coupling_map"]) > 20:
                    # Use SABRE for larger devices
                    pm.append(SabreSwap(coupling_map=backend_info["coupling_map"]))
                else:
                    # Use LookaheadSwap for smaller devices
                    pm.append(LookaheadSwap(coupling_map=backend_info["coupling_map"]))

            # Final optimizations
            pm.append(OptimizeSwapBeforeMeasure())
            pm.append(Optimize1qGates())

            # Apply optimization
            optimized_circuit = pm.run(circuit)

        except Exception as e:
            print(f"ℹ️  PassManager optimization failed, using basic transpile: {e}")
            # Fallback to basic transpilation
            from qiskit.transpiler import CouplingMap

            coupling_map = backend_info["coupling_map"]
            if isinstance(coupling_map, list):
                coupling_map_obj = CouplingMap(coupling_map)
            else:
                coupling_map_obj = coupling_map

            optimized_circuit = transpile(
                circuit,
                coupling_map=coupling_map_obj,
                basis_gates=backend_info["basis_gates"],
                optimization_level=2,
            )
            pm = None  # Set pm to None when using fallback

        # Calculate expected fidelity
        expected_fidelity = self.estimate_circuit_fidelity(
            optimized_circuit, backend_info
        )

        results = {
            "optimized_circuit": optimized_circuit,
            "pass_manager": pm,
            "expected_fidelity": expected_fidelity,
            "optimization_summary": {
                "original_depth": circuit.depth(),
                "optimized_depth": optimized_circuit.depth(),
                "original_gates": sum(dict(circuit.count_ops()).values()),
                "optimized_gates": sum(dict(optimized_circuit.count_ops()).values()),
            },
        }

        return results

    def estimate_circuit_fidelity(self, circuit, backend_info):
        """Estimate circuit fidelity based on gate errors and coherence."""
        gate_errors = backend_info.get("gate_errors", {})
        coherence_times = backend_info.get("coherence_times", {})

        # Gate error contribution
        gate_fidelity = 1.0
        for gate, count in circuit.count_ops().items():
            error_rate = gate_errors.get(gate, 0.01)  # Default 1% error
            gate_fidelity *= (1 - error_rate) ** count

        # Coherence error contribution (simplified)
        circuit_time = circuit.depth() * 100e-9  # Assume 100ns per layer
        t1_times = coherence_times.get("T1", [50e-6] * circuit.num_qubits)
        t2_times = coherence_times.get("T2", [30e-6] * circuit.num_qubits)

        coherence_fidelity = 1.0
        for t1, t2 in zip(t1_times, t2_times):
            # Simplified coherence decay
            coherence_fidelity *= np.exp(-circuit_time / t1) * np.exp(
                -circuit_time / t2
            )

        # Readout error contribution
        readout_errors = backend_info.get("readout_errors", [0.01] * circuit.num_qubits)
        readout_fidelity = np.prod([1 - error for error in readout_errors])

        total_fidelity = gate_fidelity * coherence_fidelity * readout_fidelity

        return total_fidelity


class CircuitBenchmark:
    def __init__(self, optimizer, verbose=False):
        self.optimizer = optimizer
        self.verbose = verbose

    def create_benchmark_circuits(self):
        """Create various benchmark circuits for testing."""
        circuits = {}

        # Quantum Volume circuit
        qv_circuit = QuantumVolume(4, depth=4, seed=42)
        circuits["quantum_volume"] = qv_circuit.decompose()

        # QFT circuit
        qft_circuit = QFT(4)
        circuits["qft"] = qft_circuit.decompose()

        # Random circuit with heavy connectivity
        random_circuit = QuantumCircuit(5, 5)
        np.random.seed(42)
        for _ in range(20):
            gate_type = np.random.choice(["h", "rx", "ry", "rz", "cx"])
            if gate_type == "h":
                qubit = np.random.randint(5)
                random_circuit.h(qubit)
            elif gate_type in ["rx", "ry", "rz"]:
                qubit = np.random.randint(5)
                angle = np.random.uniform(0, 2 * np.pi)
                getattr(random_circuit, gate_type)(angle, qubit)
            elif gate_type == "cx":
                control = np.random.randint(5)
                target = np.random.randint(5)
                if control != target:
                    random_circuit.cx(control, target)
        random_circuit.measure_all()
        circuits["random_heavy"] = random_circuit

        # Bell state preparation
        bell_circuit = QuantumCircuit(2, 2)
        bell_circuit.h(0)
        bell_circuit.cx(0, 1)
        bell_circuit.measure_all()
        circuits["bell_state"] = bell_circuit

        # GHZ state with measurements
        ghz_circuit = QuantumCircuit(4, 4)
        ghz_circuit.h(0)
        for i in range(1, 4):
            ghz_circuit.cx(0, i)
        ghz_circuit.measure_all()
        circuits["ghz_state"] = ghz_circuit

        return circuits

    def run_optimization_benchmark(self):
        """Run comprehensive optimization benchmark."""
        circuits = self.create_benchmark_circuits()
        backends = self.optimizer.backends

        benchmark_results = {}

        for circuit_name, circuit in circuits.items():
            if self.verbose:
                print(f"\n🔧 Optimizing circuit: {circuit_name}")

            circuit_results = {}

            for backend_name in backends.keys():
                if self.verbose:
                    print(f"   Backend: {backend_name}")

                try:
                    # Optimize for backend
                    optimization_result = self.optimizer.optimize_for_backend(
                        circuit, backend_name, optimization_level=3
                    )

                    # Calibration-aware optimization
                    calibration_result = (
                        self.optimizer.create_calibration_aware_optimization(
                            circuit, backend_name
                        )
                    )

                    circuit_results[backend_name] = {
                        "optimization": optimization_result,
                        "calibration": calibration_result,
                        "success": True,
                    }

                except Exception as e:
                    circuit_results[backend_name] = {"success": False, "error": str(e)}
                    if self.verbose:
                        print(f"     ❌ Failed: {e}")

            benchmark_results[circuit_name] = circuit_results

        return benchmark_results

    def analyze_optimization_effectiveness(self, benchmark_results):
        """Analyze effectiveness of optimization across circuits and backends."""
        analysis = {
            "depth_improvements": {},
            "fidelity_estimates": {},
            "best_backend_per_circuit": {},
            "best_strategy_per_backend": {},
        }

        for circuit_name, circuit_results in benchmark_results.items():
            depth_improvements = []
            fidelity_estimates = []
            best_backend = None
            best_fidelity = 0

            for backend_name, results in circuit_results.items():
                if results["success"]:
                    opt_result = results["optimization"]
                    cal_result = results["calibration"]

                    # Depth improvement
                    if "best_strategy" in opt_result:
                        best_strategy = opt_result["best_strategy"]
                        depth_reduction = opt_result[best_strategy]["depth_reduction"]
                        depth_improvements.append(depth_reduction)

                    # Fidelity estimate
                    fidelity = cal_result["expected_fidelity"]
                    fidelity_estimates.append(fidelity)

                    if fidelity > best_fidelity:
                        best_fidelity = fidelity
                        best_backend = backend_name

            analysis["depth_improvements"][circuit_name] = depth_improvements
            analysis["fidelity_estimates"][circuit_name] = fidelity_estimates
            analysis["best_backend_per_circuit"][circuit_name] = best_backend

        # Strategy analysis
        strategy_performance = {}
        for circuit_name, circuit_results in benchmark_results.items():
            for backend_name, results in circuit_results.items():
                if results["success"] and "optimization" in results:
                    opt_result = results["optimization"]
                    for strategy, strategy_result in opt_result.items():
                        if (
                            isinstance(strategy_result, dict)
                            and "depth_reduction" in strategy_result
                        ):
                            if strategy not in strategy_performance:
                                strategy_performance[strategy] = []
                            strategy_performance[strategy].append(
                                strategy_result["depth_reduction"]
                            )

        # Find best strategy overall
        best_strategy_overall = None
        best_avg_improvement = float("-inf")

        for strategy, improvements in strategy_performance.items():
            avg_improvement = np.mean(improvements)
            if avg_improvement > best_avg_improvement:
                best_avg_improvement = avg_improvement
                best_strategy_overall = strategy

        analysis["best_strategy_overall"] = best_strategy_overall
        analysis["strategy_performance"] = strategy_performance

        return analysis


def visualize_optimization_results(optimizer, benchmark_results, analysis):
    """Visualize optimization results and backend comparisons."""
    fig = plt.figure(figsize=(16, 12))

    # Backend architecture visualization
    ax1 = plt.subplot(2, 3, 1)

    # Plot coupling maps
    if HAS_NETWORKX:
        for i, (backend_name, backend_info) in enumerate(optimizer.backends.items()):
            if i < 2:  # Show first two backends
                coupling_map = backend_info["coupling_map"]

                # Create graph
                G = nx.Graph()
                G.add_edges_from(coupling_map)

                # Position nodes
                if "linear" in backend_name.lower():
                    pos = {i: (i, 0) for i in range(backend_info["n_qubits"])}
                elif "grid" in backend_name.lower():
                    pos = {i: (i % 5, i // 5) for i in range(backend_info["n_qubits"])}
                else:
                    pos = nx.spring_layout(G)

                # Plot
                nx.draw(
                    G,
                    pos,
                    ax=ax1,
                    node_color=f"C{i}",
                    node_size=200,
                    alpha=0.7,
                    label=backend_name,
                )

        ax1.set_title("Backend Architectures")
        ax1.legend()
        ax1.axis("equal")
    else:
        ax1.text(0.5, 0.5, "Network visualization\nrequires networkx", 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Backend Architectures (networkx required)")
        ax1.axis('off')

    # Optimization effectiveness
    ax2 = plt.subplot(2, 3, 2)

    circuit_names = []
    avg_improvements = []

    for circuit_name, improvements in analysis["depth_improvements"].items():
        if improvements:
            circuit_names.append(circuit_name.replace("_", "\n"))
            avg_improvements.append(np.mean(improvements))

    if circuit_names:
        bars = ax2.bar(circuit_names, avg_improvements, alpha=0.7, color="skyblue")
        ax2.set_ylabel("Avg Depth Reduction")
        ax2.set_title("Optimization Effectiveness")
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, improvement in zip(bars, avg_improvements):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{improvement:.1%}",
                ha="center",
                va="bottom",
            )

    # Fidelity estimates
    ax3 = plt.subplot(2, 3, 3)

    circuit_names = []
    avg_fidelities = []

    for circuit_name, fidelities in analysis["fidelity_estimates"].items():
        if fidelities:
            circuit_names.append(circuit_name.replace("_", "\n"))
            avg_fidelities.append(np.mean(fidelities))

    if circuit_names:
        bars = ax3.bar(circuit_names, avg_fidelities, alpha=0.7, color="lightcoral")
        ax3.set_ylabel("Avg Expected Fidelity")
        ax3.set_title("Circuit Fidelity Estimates")
        ax3.tick_params(axis="x", rotation=45)
        ax3.set_ylim(0, 1)

        # Add value labels
        for bar, fidelity in zip(bars, avg_fidelities):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{fidelity:.3f}",
                ha="center",
                va="bottom",
            )

    # Strategy comparison
    ax4 = plt.subplot(2, 3, 4)

    if "strategy_performance" in analysis:
        strategy_names = []
        strategy_scores = []

        for strategy, improvements in analysis["strategy_performance"].items():
            strategy_names.append(strategy)
            strategy_scores.append(np.mean(improvements))

        if strategy_names:
            bars = ax4.bar(
                strategy_names, strategy_scores, alpha=0.7, color="lightgreen"
            )
            ax4.set_ylabel("Avg Depth Reduction")
            ax4.set_title("Strategy Comparison")
            ax4.tick_params(axis="x", rotation=45)

            # Highlight best strategy
            if "best_strategy_overall" in analysis:
                best_strategy = analysis["best_strategy_overall"]
                if best_strategy in strategy_names:
                    best_idx = strategy_names.index(best_strategy)
                    bars[best_idx].set_color("gold")

    # Backend suitability heatmap
    ax5 = plt.subplot(2, 3, 5)

    # Create heatmap data
    circuits = list(benchmark_results.keys())
    backends = list(optimizer.backends.keys())

    heatmap_data = np.zeros((len(circuits), len(backends)))

    for i, circuit_name in enumerate(circuits):
        for j, backend_name in enumerate(backends):
            if (
                circuit_name in benchmark_results
                and backend_name in benchmark_results[circuit_name]
                and benchmark_results[circuit_name][backend_name]["success"]
            ):

                # Use fidelity as metric
                fidelity = benchmark_results[circuit_name][backend_name]["calibration"][
                    "expected_fidelity"
                ]
                heatmap_data[i, j] = fidelity

    im = ax5.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax5.set_xticks(range(len(backends)))
    ax5.set_xticklabels([b.replace("_", "\n") for b in backends], rotation=45)
    ax5.set_yticks(range(len(circuits)))
    ax5.set_yticklabels([c.replace("_", "\n") for c in circuits])
    ax5.set_title("Circuit-Backend Compatibility")

    # Add colorbar
    plt.colorbar(im, ax=ax5, label="Expected Fidelity")

    # Summary and insights
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = "Hardware Optimization Summary:\n\n"

    if "best_strategy_overall" in analysis:
        summary_text += f"Best Strategy: {analysis['best_strategy_overall']}\n\n"

    summary_text += "Key Insights:\n\n"
    summary_text += "Architecture Impact:\n"
    summary_text += "• All-to-all: Best for dense circuits\n"
    summary_text += "• Linear: Good for sequential ops\n"
    summary_text += "• Grid: Balanced performance\n"
    summary_text += "• Heavy-hex: IBM quantum ready\n\n"

    summary_text += "Optimization Strategies:\n"
    summary_text += "• SABRE: Best for complex routing\n"
    summary_text += "• Lookahead: Good for small circuits\n"
    summary_text += "• Basic: Fast but limited\n"
    summary_text += "• Stochastic: Good average case\n\n"

    summary_text += "Best Practices:\n"
    summary_text += "• Match topology to algorithm\n"
    summary_text += "• Consider gate set compatibility\n"
    summary_text += "• Account for calibration data\n"
    summary_text += "• Minimize circuit depth\n"
    summary_text += "• Use native gates when possible"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )

    plt.tight_layout()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hardware-Optimized Circuits")
    parser.add_argument(
        "--circuit",
        choices=["qft", "quantum_volume", "random_heavy", "bell_state", "ghz_state"],
        default="qft",
        help="Circuit to optimize",
    )
    parser.add_argument(
        "--backend",
        choices=["linear_5q", "grid_10q", "heavy_hex_16q", "all_to_all_5q"],
        default="linear_5q",
        help="Target backend",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        help="Transpiler optimization level",
    )
    parser.add_argument(
        "--compare-backends", action="store_true", help="Compare backend suitability"
    )
    parser.add_argument(
        "--run-benchmark", action="store_true", help="Run full optimization benchmark"
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms")
    print("Example 3: Hardware-Optimized Circuits")
    print("=" * 53)

    try:
        # Initialize optimizer
        optimizer = HardwareOptimizer(verbose=args.verbose)
        backends = optimizer.create_mock_backends()

        print(f"\n🔧 Available Backend Architectures:")
        for name, backend in backends.items():
            print(
                f"   {name}: {backend['n_qubits']} qubits, "
                f"{len(backend['coupling_map'])} connections"
            )

        # Create benchmark
        benchmark = CircuitBenchmark(optimizer, verbose=args.verbose)
        circuits = benchmark.create_benchmark_circuits()

        # Single circuit optimization
        if not args.run_benchmark and not args.compare_backends:
            circuit = circuits[args.circuit]

            print(f"\n🎯 Optimizing circuit: {args.circuit}")
            print(f"   Original qubits: {circuit.num_qubits}")
            print(f"   Original depth: {circuit.depth()}")
            print(f"   Original gates: {sum(dict(circuit.count_ops()).values())}")

            print(f"\n🔧 Optimizing for backend: {args.backend}")

            # Optimize for specific backend
            optimization_result = optimizer.optimize_for_backend(
                circuit, args.backend, args.optimization_level
            )

            if "best_strategy" in optimization_result:
                best_strategy = optimization_result["best_strategy"]
                best_circuit = optimization_result["best_circuit"]

                print(f"\n✅ Best optimization strategy: {best_strategy}")
                print(f"   Optimized depth: {best_circuit.depth()}")
                print(
                    f"   Optimized gates: {sum(dict(best_circuit.count_ops()).values())}"
                )

                original_analysis = optimization_result["original_analysis"]
                best_analysis = optimization_result[best_strategy]["analysis"]

                depth_reduction = optimization_result[best_strategy]["depth_reduction"]
                gate_overhead = optimization_result[best_strategy]["gate_overhead"]

                print(f"   Depth reduction: {depth_reduction:.1%}")
                print(f"   Gate overhead: {gate_overhead:.1%}")

                print(f"\n📊 Detailed Analysis:")
                print(f"   Strategy Results:")
                for strategy, result in optimization_result.items():
                    if (
                        isinstance(result, dict)
                        and "success" in result
                        and result["success"]
                    ):
                        print(
                            f"     {strategy}: depth {result['analysis']['depth']}, "
                            f"reduction {result['depth_reduction']:.1%}"
                        )

            # Calibration-aware optimization
            print(f"\n🎛️  Calibration-aware optimization:")
            cal_result = optimizer.create_calibration_aware_optimization(
                circuit, args.backend
            )

            print(f"   Expected fidelity: {cal_result['expected_fidelity']:.3f}")
            print(f"   Optimized depth: {cal_result['optimized_circuit'].depth()}")

            summary = cal_result["optimization_summary"]
            print(
                f"   Depth change: {summary['original_depth']} → {summary['optimized_depth']}"
            )
            print(
                f"   Gate change: {summary['original_gates']} → {summary['optimized_gates']}"
            )

        # Backend comparison
        if args.compare_backends:
            circuit = circuits[args.circuit]

            print(f"\n🔄 Comparing backend suitability for: {args.circuit}")

            suitability_scores, circuit_analysis = (
                optimizer.compare_backend_suitability(circuit)
            )

            print(f"\n📊 Backend Ranking:")
            for i, (backend_name, score_info) in enumerate(suitability_scores.items()):
                rank_icon = (
                    "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📋"
                )
                print(f"   {rank_icon} {backend_name}: Score {score_info['score']:.1f}")

                details = score_info["details"]
                for aspect, info in details.items():
                    if aspect == "estimated_fidelity":
                        print(f"     {aspect}: {info:.3f}")
                    else:
                        print(f"     {aspect}: {info}")

        # Full benchmark
        benchmark_results = None
        analysis = None

        if args.run_benchmark:
            print(f"\n🏁 Running optimization benchmark...")

            benchmark_results = benchmark.run_optimization_benchmark()
            analysis = benchmark.analyze_optimization_effectiveness(benchmark_results)

            print(f"\n📈 Benchmark Results:")

            for circuit_name, circuit_results in benchmark_results.items():
                print(f"\n   {circuit_name}:")

                successful_backends = [
                    name
                    for name, result in circuit_results.items()
                    if result["success"]
                ]
                print(
                    f"     Successful backends: {len(successful_backends)}/{len(circuit_results)}"
                )

                if circuit_name in analysis["best_backend_per_circuit"]:
                    best_backend = analysis["best_backend_per_circuit"][circuit_name]
                    if best_backend:
                        print(f"     Best backend: {best_backend}")

                        if best_backend in circuit_results:
                            fidelity = circuit_results[best_backend]["calibration"][
                                "expected_fidelity"
                            ]
                            print(f"     Expected fidelity: {fidelity:.3f}")

            if "best_strategy_overall" in analysis:
                print(
                    f"\n🏆 Best optimization strategy overall: {analysis['best_strategy_overall']}"
                )

                if "strategy_performance" in analysis:
                    print(f"\n📊 Strategy Performance:")
                    for strategy, improvements in analysis[
                        "strategy_performance"
                    ].items():
                        avg_improvement = np.mean(improvements)
                        print(
                            f"     {strategy}: {avg_improvement:.1%} avg depth reduction"
                        )

        # Visualization
        if args.show_visualization and benchmark_results and analysis:
            visualize_optimization_results(optimizer, benchmark_results, analysis)

        print(f"\n📚 Key Insights:")
        print(f"   • Different architectures suit different quantum algorithms")
        print(f"   • Optimization strategy choice significantly impacts performance")
        print(f"   • Calibration data is crucial for realistic fidelity estimates")
        print(
            f"   • Circuit-backend matching can improve results by orders of magnitude"
        )

        print(f"\n🎯 Optimization Guidelines:")
        print(f"   • Use all-to-all connectivity for dense quantum circuits")
        print(f"   • Linear topologies work well for sequential algorithms")
        print(f"   • Grid topologies provide balanced performance")
        print(f"   • Always consider native gate sets to minimize decomposition")
        print(f"   • Account for realistic error rates and coherence times")

        print(f"\n✅ Hardware optimization analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
