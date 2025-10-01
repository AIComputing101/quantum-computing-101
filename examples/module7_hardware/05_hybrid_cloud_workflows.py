#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms
Example 5: Hybrid Cloud Workflows

Implementation of hybrid classical-quantum workflows across multiple cloud platforms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import argparse
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from qiskit import (
    QuantumCircuit,
    ClassicalRegister,
    transpile,
    QuantumRegister,
    ClassicalRegister,
)
from qiskit_aer import AerSimulator
from qiskit.circuit.library import TwoLocal, EfficientSU2
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


class CloudResourceManager:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.providers = {}
        self.resource_pools = {}
        self.job_queue = []
        self.completed_jobs = []

    def register_provider(self, provider_name, capabilities):
        """Register a cloud provider with its capabilities."""
        self.providers[provider_name] = {
            "name": provider_name,
            "capabilities": capabilities,
            "status": "available",
            "current_load": 0,
            "max_concurrent_jobs": capabilities.get("max_concurrent_jobs", 5),
            "cost_per_shot": capabilities.get("cost_per_shot", 0.001),
            "queue_time_estimate": 0,
        }

        if self.verbose:
            print(f"✅ Registered provider: {provider_name}")

    def initialize_default_providers(self):
        """Initialize default cloud providers for demonstration."""
        # IBM Quantum
        self.register_provider(
            "IBM_Quantum",
            {
                "qubits": 127,
                "connectivity": "heavy_hex",
                "gate_set": ["cx", "id", "rz", "sx", "x"],
                "max_concurrent_jobs": 3,
                "cost_per_shot": 0.00075,
                "typical_queue_time": 60,  # seconds
                "gate_fidelity": 0.99,
                "backend_types": ["simulator", "hardware"],
            },
        )

        # AWS Braket - IonQ
        self.register_provider(
            "AWS_IonQ",
            {
                "qubits": 32,
                "connectivity": "all_to_all",
                "gate_set": ["rxx", "ry", "rz"],
                "max_concurrent_jobs": 2,
                "cost_per_shot": 0.01,
                "typical_queue_time": 120,
                "gate_fidelity": 0.995,
                "backend_types": ["hardware"],
            },
        )

        # AWS Braket - Rigetti
        self.register_provider(
            "AWS_Rigetti",
            {
                "qubits": 80,
                "connectivity": "limited",
                "gate_set": ["rx", "ry", "rz", "cz"],
                "max_concurrent_jobs": 4,
                "cost_per_shot": 0.005,
                "typical_queue_time": 30,
                "gate_fidelity": 0.985,
                "backend_types": ["hardware"],
            },
        )

        # Google Quantum AI
        self.register_provider(
            "Google_QAI",
            {
                "qubits": 70,
                "connectivity": "grid",
                "gate_set": ["fsim", "sqrt_iswap", "rz", "ry"],
                "max_concurrent_jobs": 3,
                "cost_per_shot": 0.003,
                "typical_queue_time": 45,
                "gate_fidelity": 0.992,
                "backend_types": ["simulator", "hardware"],
            },
        )

        # Local simulators
        self.register_provider(
            "Local_Simulator",
            {
                "qubits": 32,
                "connectivity": "all_to_all",
                "gate_set": ["cx", "u1", "u2", "u3", "id"],
                "max_concurrent_jobs": 10,
                "cost_per_shot": 0.0,
                "typical_queue_time": 0,
                "gate_fidelity": 1.0,
                "backend_types": ["simulator"],
            },
        )

    def estimate_job_cost(self, provider_name, shots, circuit_complexity=1.0):
        """Estimate cost for running a job on a provider."""
        if provider_name not in self.providers:
            return float("inf")

        provider = self.providers[provider_name]
        base_cost = shots * provider["cost_per_shot"]
        complexity_multiplier = 1 + (circuit_complexity - 1) * 0.5

        return base_cost * complexity_multiplier

    def estimate_queue_time(self, provider_name):
        """Estimate queue time for a provider."""
        if provider_name not in self.providers:
            return float("inf")

        provider = self.providers[provider_name]
        base_time = provider["typical_queue_time"]
        load_multiplier = 1 + provider["current_load"] / provider["max_concurrent_jobs"]

        return base_time * load_multiplier

    def select_optimal_provider(self, circuit, shots, priority="cost"):
        """Select optimal provider based on criteria."""
        scores = {}

        circuit_complexity = self.analyze_circuit_complexity(circuit)

        for provider_name, provider in self.providers.items():
            if circuit.num_qubits > provider["capabilities"]["qubits"]:
                continue  # Skip if not enough qubits

            # Calculate metrics
            cost = self.estimate_job_cost(provider_name, shots, circuit_complexity)
            queue_time = self.estimate_queue_time(provider_name)
            fidelity = provider["capabilities"]["gate_fidelity"]

            # Scoring based on priority
            if priority == "cost":
                score = 1 / (cost + 1e-6)
            elif priority == "speed":
                score = 1 / (queue_time + 1)
            elif priority == "fidelity":
                score = fidelity
            else:  # balanced
                normalized_cost = cost / 10.0  # Normalize
                normalized_time = queue_time / 100.0  # Normalize
                score = fidelity / (1 + normalized_cost + normalized_time)

            scores[provider_name] = {
                "score": score,
                "cost": cost,
                "queue_time": queue_time,
                "fidelity": fidelity,
            }

        if not scores:
            return None

        # Return best provider
        best_provider = max(scores.items(), key=lambda x: x[1]["score"])
        return best_provider[0], scores

    def analyze_circuit_complexity(self, circuit):
        """Analyze circuit complexity for cost estimation."""
        gate_counts = dict(circuit.count_ops())

        # Weight different gates by complexity
        complexity_weights = {
            "cx": 10,
            "cz": 10,
            "ccx": 20,
            "h": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "rx": 2,
            "ry": 2,
            "rz": 2,
            "u1": 2,
            "u2": 4,
            "u3": 6,
        }

        total_complexity = 0
        for gate, count in gate_counts.items():
            weight = complexity_weights.get(gate, 5)  # Default weight
            total_complexity += count * weight

        # Normalize by circuit size
        return total_complexity / max(circuit.num_qubits, 1)


class HybridWorkflowManager:
    def __init__(self, resource_manager, verbose=False):
        self.resource_manager = resource_manager
        self.verbose = verbose
        self.workflows = {}

    def create_vqe_workflow(self, hamiltonian_coeffs, n_qubits=4):
        """Create VQE workflow that can be distributed across providers."""
        workflow = {
            "name": "VQE_Optimization",
            "type": "variational",
            "tasks": [],
            "parameters": {
                "n_qubits": n_qubits,
                "hamiltonian": hamiltonian_coeffs,
                "max_iterations": 50,
                "convergence_threshold": 1e-6,
            },
        }

        # Create parameterized ansatz
        ansatz = TwoLocal(n_qubits, "ry", "cz", reps=2)

        # Split into multiple circuits for parallel execution
        n_params = ansatz.num_parameters
        param_ranges = np.array_split(range(n_params), min(4, n_params))

        for i, param_range in enumerate(param_ranges):
            task = {
                "task_id": f"vqe_task_{i}",
                "circuit_template": ansatz,
                "parameter_range": list(param_range),
                "shots": 1000,
                "priority": "balanced",
            }
            workflow["tasks"].append(task)

        return workflow

    def create_qaoa_workflow(self, graph_edges, n_layers=2):
        """Create QAOA workflow for Max-Cut problem."""
        n_qubits = max(max(edge) for edge in graph_edges) + 1

        workflow = {
            "name": "QAOA_MaxCut",
            "type": "approximate",
            "tasks": [],
            "parameters": {
                "n_qubits": n_qubits,
                "graph_edges": graph_edges,
                "n_layers": n_layers,
                "max_iterations": 30,
            },
        }

        # Create QAOA circuits for different parameter values
        for layer in range(n_layers):
            for angle_set in range(3):  # Different starting points
                task = {
                    "task_id": f"qaoa_layer_{layer}_set_{angle_set}",
                    "circuit_template": self.create_qaoa_circuit(graph_edges, n_layers),
                    "parameter_set": angle_set,
                    "shots": 2000,
                    "priority": "cost",
                }
                workflow["tasks"].append(task)

        return workflow

    def create_qaoa_circuit(self, graph_edges, n_layers):
        """Create QAOA circuit for Max-Cut."""
        n_qubits = max(max(edge) for edge in graph_edges) + 1
        qc = QuantumCircuit(n_qubits)

        # Initial superposition
        for qubit in range(n_qubits):
            qc.h(qubit)

        # QAOA layers
        for layer in range(n_layers):
            # Problem Hamiltonian (cost function)
            for edge in graph_edges:
                qc.cx(edge[0], edge[1])
                qc.rz(0.5, edge[1])  # Placeholder parameter
                qc.cx(edge[0], edge[1])

            # Mixer Hamiltonian
            for qubit in range(n_qubits):
                qc.rx(0.5, qubit)  # Placeholder parameter

        qc.measure_all()
        return qc

    def create_quantum_ml_workflow(self, dataset_size, n_features=4):
        """Create quantum machine learning workflow."""
        workflow = {
            "name": "Quantum_ML_Training",
            "type": "machine_learning",
            "tasks": [],
            "parameters": {
                "dataset_size": dataset_size,
                "n_features": n_features,
                "n_qubits": n_features + 1,  # +1 for ancilla
                "training_epochs": 20,
            },
        }

        # Create feature map and variational form
        feature_map = self.create_feature_map(n_features)
        var_form = self.create_variational_form(n_features + 1)

        # Split training into batches for parallel processing
        n_batches = min(4, dataset_size // 10)

        for batch in range(n_batches):
            task = {
                "task_id": f"qml_batch_{batch}",
                "circuit_template": self.combine_circuits(feature_map, var_form),
                "batch_id": batch,
                "shots": 500,
                "priority": "fidelity",
            }
            workflow["tasks"].append(task)

        return workflow

    def create_feature_map(self, n_features):
        """Create quantum feature map."""
        qc = QuantumCircuit(n_features)

        # Angle encoding
        for i in range(n_features):
            qc.ry(0.5, i)  # Placeholder for data encoding

        # Entangling layers
        for i in range(n_features - 1):
            qc.cx(i, i + 1)

        return qc

    def create_variational_form(self, n_qubits):
        """Create variational quantum circuit."""
        return EfficientSU2(n_qubits, reps=2)

    def combine_circuits(self, circuit1, circuit2):
        """Combine two quantum circuits."""
        combined = circuit1.copy()
        combined.compose(circuit2, inplace=True)
        combined.measure_all()
        return combined

    def execute_workflow(self, workflow, max_parallel_jobs=5):
        """Execute workflow with optimal resource allocation."""
        if self.verbose:
            print(f"🚀 Executing workflow: {workflow['name']}")
            print(f"   Tasks: {len(workflow['tasks'])}")

        results = {}
        execution_plan = self.create_execution_plan(workflow, max_parallel_jobs)

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
            future_to_task = {}

            for task in workflow["tasks"]:
                future = executor.submit(self.execute_task, task)
                future_to_task[future] = task

            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task["task_id"]] = result

                    if self.verbose:
                        print(f"   ✅ Completed: {task['task_id']}")

                except Exception as e:
                    if self.verbose:
                        print(f"   ❌ Failed: {task['task_id']} - {e}")
                    results[task["task_id"]] = {"error": str(e)}

        # Aggregate results
        aggregated_results = self.aggregate_workflow_results(workflow, results)

        return aggregated_results

    def create_execution_plan(self, workflow, max_parallel_jobs):
        """Create optimal execution plan for workflow."""
        plan = {
            "total_tasks": len(workflow["tasks"]),
            "estimated_cost": 0,
            "estimated_time": 0,
            "provider_allocation": {},
        }

        for task in workflow["tasks"]:
            # Select optimal provider for each task
            circuit = task["circuit_template"]
            shots = task["shots"]
            priority = task["priority"]

            provider, scores = self.resource_manager.select_optimal_provider(
                circuit, shots, priority
            )

            if provider:
                if provider not in plan["provider_allocation"]:
                    plan["provider_allocation"][provider] = []

                plan["provider_allocation"][provider].append(task["task_id"])
                plan["estimated_cost"] += scores[provider]["cost"]
                plan["estimated_time"] = max(
                    plan["estimated_time"], scores[provider]["queue_time"]
                )

        return plan

    def execute_task(self, task):
        """Execute a single task."""
        circuit = task["circuit_template"]
        shots = task["shots"]
        priority = task["priority"]

        # Select provider
        provider, scores = self.resource_manager.select_optimal_provider(
            circuit, shots, priority
        )

        if not provider:
            raise Exception("No suitable provider found")

        # Simulate task execution
        start_time = time.time()

        # Add realistic delay based on provider
        queue_time = scores[provider]["queue_time"]
        time.sleep(min(queue_time / 100, 0.5))  # Scale down for demo

        # Simulate quantum execution
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=shots)
        result = job.result()

        execution_time = time.time() - start_time

        return {
            "task_id": task["task_id"],
            "provider": provider,
            "result": result,
            "execution_time": execution_time,
            "cost": scores[provider]["cost"],
            "shots": shots,
            "success": True,
        }

    def aggregate_workflow_results(self, workflow, task_results):
        """Aggregate results from all tasks in workflow."""
        successful_tasks = [r for r in task_results.values() if r.get("success", False)]
        failed_tasks = [r for r in task_results.values() if not r.get("success", False)]

        total_cost = sum(r.get("cost", 0) for r in successful_tasks)
        total_time = max(
            (r.get("execution_time", 0) for r in successful_tasks), default=0
        )
        total_shots = sum(r.get("shots", 0) for r in successful_tasks)

        # Provider usage statistics
        provider_stats = {}
        for result in successful_tasks:
            provider = result.get("provider", "unknown")
            if provider not in provider_stats:
                provider_stats[provider] = {"tasks": 0, "cost": 0, "shots": 0}

            provider_stats[provider]["tasks"] += 1
            provider_stats[provider]["cost"] += result.get("cost", 0)
            provider_stats[provider]["shots"] += result.get("shots", 0)

        aggregated = {
            "workflow_name": workflow["name"],
            "total_tasks": len(task_results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "total_cost": total_cost,
            "total_execution_time": total_time,
            "total_shots": total_shots,
            "provider_statistics": provider_stats,
            "task_results": task_results,
        }

        return aggregated


class WorkflowOptimizer:
    def __init__(self, resource_manager, verbose=False):
        self.resource_manager = resource_manager
        self.verbose = verbose

    def optimize_workflow_allocation(self, workflow, constraints=None):
        """Optimize task allocation across providers."""
        if not constraints:
            constraints = {
                "max_cost": 100.0,
                "max_time": 300,  # seconds
                "min_fidelity": 0.95,
            }

        # Get all possible allocations
        allocations = self.generate_allocations(workflow)

        # Score allocations
        best_allocation = None
        best_score = float("-inf")

        for allocation in allocations:
            score = self.score_allocation(allocation, constraints)

            if score > best_score:
                best_score = score
                best_allocation = allocation

        return best_allocation, best_score

    def generate_allocations(self, workflow):
        """Generate possible task allocations."""
        # Simplified: just return a few representative allocations
        allocations = []

        # Cost-optimized allocation
        cost_allocation = {}
        for task in workflow["tasks"]:
            circuit = task["circuit_template"]
            shots = task["shots"]

            provider, _ = self.resource_manager.select_optimal_provider(
                circuit, shots, "cost"
            )

            if provider:
                if provider not in cost_allocation:
                    cost_allocation[provider] = []
                cost_allocation[provider].append(task["task_id"])

        allocations.append(("cost_optimized", cost_allocation))

        # Speed-optimized allocation
        speed_allocation = {}
        for task in workflow["tasks"]:
            circuit = task["circuit_template"]
            shots = task["shots"]

            provider, _ = self.resource_manager.select_optimal_provider(
                circuit, shots, "speed"
            )

            if provider:
                if provider not in speed_allocation:
                    speed_allocation[provider] = []
                speed_allocation[provider].append(task["task_id"])

        allocations.append(("speed_optimized", speed_allocation))

        # Balanced allocation
        balanced_allocation = {}
        for task in workflow["tasks"]:
            circuit = task["circuit_template"]
            shots = task["shots"]

            provider, _ = self.resource_manager.select_optimal_provider(
                circuit, shots, "balanced"
            )

            if provider:
                if provider not in balanced_allocation:
                    balanced_allocation[provider] = []
                balanced_allocation[provider].append(task["task_id"])

        allocations.append(("balanced", balanced_allocation))

        return allocations

    def score_allocation(self, allocation, constraints):
        """Score an allocation based on constraints."""
        name, provider_allocation = allocation

        total_cost = 0
        max_time = 0
        min_fidelity = 1.0

        for provider, task_ids in provider_allocation.items():
            if provider in self.resource_manager.providers:
                provider_info = self.resource_manager.providers[provider]

                # Estimate metrics for this provider
                n_tasks = len(task_ids)
                estimated_cost = (
                    n_tasks * 1000 * provider_info["cost_per_shot"]
                )  # Rough estimate
                estimated_time = provider_info["typical_queue_time"]
                fidelity = provider_info["capabilities"]["gate_fidelity"]

                total_cost += estimated_cost
                max_time = max(max_time, estimated_time)
                min_fidelity = min(min_fidelity, fidelity)

        # Penalty for constraint violations
        score = 100  # Base score

        if total_cost > constraints["max_cost"]:
            score -= (total_cost - constraints["max_cost"]) * 10

        if max_time > constraints["max_time"]:
            score -= (max_time - constraints["max_time"]) * 0.1

        if min_fidelity < constraints["min_fidelity"]:
            score -= (constraints["min_fidelity"] - min_fidelity) * 1000

        # Bonus for efficiency
        score += (constraints["max_cost"] - total_cost) * 0.1  # Cost efficiency
        score += (constraints["max_time"] - max_time) * 0.01  # Time efficiency
        score += min_fidelity * 10  # Fidelity bonus

        return score


def visualize_workflow_results(workflow_results, resource_stats):
    """Visualize hybrid workflow execution results."""
    fig = plt.figure(figsize=(16, 12))

    # Workflow performance comparison
    ax1 = plt.subplot(2, 3, 1)

    workflow_names = []
    success_rates = []
    costs = []

    for workflow_name, results in workflow_results.items():
        workflow_names.append(workflow_name.replace("_", "\n"))
        success_rate = results["successful_tasks"] / results["total_tasks"]
        success_rates.append(success_rate)
        costs.append(results["total_cost"])

    if workflow_names:
        # Success rates
        bars = ax1.bar(workflow_names, success_rates, alpha=0.7, color="green")
        ax1.set_ylabel("Success Rate")
        ax1.set_title("Workflow Success Rates")
        ax1.set_ylim(0, 1.1)

        # Add value labels
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )

    # Cost comparison
    ax2 = plt.subplot(2, 3, 2)

    if workflow_names and costs:
        bars = ax2.bar(workflow_names, costs, alpha=0.7, color="gold")
        ax2.set_ylabel("Total Cost ($)")
        ax2.set_title("Workflow Costs")

        # Add value labels
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${cost:.2f}",
                ha="center",
                va="bottom",
            )

    # Provider utilization
    ax3 = plt.subplot(2, 3, 3)

    # Aggregate provider usage across all workflows
    provider_usage = {}
    for results in workflow_results.values():
        for provider, stats in results["provider_statistics"].items():
            if provider not in provider_usage:
                provider_usage[provider] = {"tasks": 0, "cost": 0}

            provider_usage[provider]["tasks"] += stats["tasks"]
            provider_usage[provider]["cost"] += stats["cost"]

    if provider_usage:
        providers = list(provider_usage.keys())
        task_counts = [provider_usage[p]["tasks"] for p in providers]

        bars = ax3.bar(
            [p.replace("_", "\n") for p in providers],
            task_counts,
            alpha=0.7,
            color="skyblue",
        )
        ax3.set_ylabel("Tasks Executed")
        ax3.set_title("Provider Utilization")
        ax3.tick_params(axis="x", rotation=45)

    # Execution time analysis
    ax4 = plt.subplot(2, 3, 4)

    if workflow_names:
        execution_times = [
            workflow_results[wf.replace("\n", "_")]["total_execution_time"]
            for wf in workflow_names
        ]

        bars = ax4.bar(workflow_names, execution_times, alpha=0.7, color="coral")
        ax4.set_ylabel("Execution Time (s)")
        ax4.set_title("Workflow Execution Times")
        ax4.tick_params(axis="x", rotation=45)

    # Cost-effectiveness analysis
    ax5 = plt.subplot(2, 3, 5)

    if workflow_names and costs and success_rates:
        # Cost per successful task
        cost_effectiveness = []
        for i, (cost, rate, results) in enumerate(
            zip(costs, success_rates, workflow_results.values())
        ):
            if results["successful_tasks"] > 0:
                cost_per_task = cost / results["successful_tasks"]
                cost_effectiveness.append(cost_per_task)
            else:
                cost_effectiveness.append(0)

        bars = ax5.bar(workflow_names, cost_effectiveness, alpha=0.7, color="purple")
        ax5.set_ylabel("Cost per Successful Task ($)")
        ax5.set_title("Cost Effectiveness")
        ax5.tick_params(axis="x", rotation=45)

    # Summary and insights
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = "Hybrid Cloud Workflow Summary:\n\n"

    if workflow_results:
        total_workflows = len(workflow_results)
        total_tasks = sum(r["total_tasks"] for r in workflow_results.values())
        total_successful = sum(r["successful_tasks"] for r in workflow_results.values())
        total_cost = sum(r["total_cost"] for r in workflow_results.values())

        summary_text += f"Workflows Executed: {total_workflows}\n"
        summary_text += f"Total Tasks: {total_tasks}\n"
        summary_text += f"Successful Tasks: {total_successful}\n"
        summary_text += f"Overall Success Rate: {total_successful/total_tasks:.1%}\n"
        summary_text += f"Total Cost: ${total_cost:.2f}\n\n"

    summary_text += "Key Benefits:\n\n"
    summary_text += "Resource Optimization:\n"
    summary_text += "• Automatic provider selection\n"
    summary_text += "• Cost and performance balance\n"
    summary_text += "• Parallel task execution\n\n"

    summary_text += "Flexibility:\n"
    summary_text += "• Multi-provider support\n"
    summary_text += "• Fallback mechanisms\n"
    summary_text += "• Scalable architecture\n\n"

    summary_text += "Best Practices:\n"
    summary_text += "• Choose providers by algorithm\n"
    summary_text += "• Monitor costs continuously\n"
    summary_text += "• Implement error handling\n"
    summary_text += "• Use local sims for development"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    plt.tight_layout()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid Cloud Workflows")
    parser.add_argument(
        "--workflow",
        choices=["vqe", "qaoa", "qml", "all"],
        default="vqe",
        help="Workflow type to execute",
    )
    parser.add_argument(
        "--max-parallel-jobs", type=int, default=3, help="Maximum parallel jobs"
    )
    parser.add_argument(
        "--optimize-allocation", action="store_true", help="Optimize task allocation"
    )
    parser.add_argument(
        "--cost-limit", type=float, default=50.0, help="Maximum cost constraint"
    )
    parser.add_argument(
        "--time-limit", type=int, default=300, help="Maximum time constraint (seconds)"
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare different allocation strategies",
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms")
    print("Example 5: Hybrid Cloud Workflows")
    print("=" * 48)

    try:
        # Initialize resource manager
        resource_manager = CloudResourceManager(verbose=args.verbose)
        resource_manager.initialize_default_providers()

        print(f"\n🌐 Available Cloud Providers:")
        for name, provider in resource_manager.providers.items():
            capabilities = provider["capabilities"]
            print(f"   {name}:")
            print(f"     Qubits: {capabilities['qubits']}")
            print(f"     Cost/shot: ${capabilities['cost_per_shot']:.5f}")
            print(f"     Queue time: ~{capabilities['typical_queue_time']}s")
            print(f"     Fidelity: {capabilities['gate_fidelity']:.3f}")

        # Initialize workflow manager
        workflow_manager = HybridWorkflowManager(resource_manager, verbose=args.verbose)

        # Create workflows
        workflows = {}

        if args.workflow == "vqe" or args.workflow == "all":
            hamiltonian = [1.0, -0.5, 0.3, -0.2]  # Example coefficients
            workflows["VQE_Optimization"] = workflow_manager.create_vqe_workflow(
                hamiltonian, n_qubits=4
            )

        if args.workflow == "qaoa" or args.workflow == "all":
            graph_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]  # Example graph
            workflows["QAOA_MaxCut"] = workflow_manager.create_qaoa_workflow(
                graph_edges, n_layers=2
            )

        if args.workflow == "qml" or args.workflow == "all":
            workflows["Quantum_ML_Training"] = (
                workflow_manager.create_quantum_ml_workflow(
                    dataset_size=100, n_features=4
                )
            )

        print(f"\n📋 Created Workflows:")
        for name, workflow in workflows.items():
            print(f"   {name}: {len(workflow['tasks'])} tasks")

        # Optimization analysis
        if args.optimize_allocation:
            print(f"\n🔧 Optimizing workflow allocation...")

            optimizer = WorkflowOptimizer(resource_manager, verbose=args.verbose)

            constraints = {
                "max_cost": args.cost_limit,
                "max_time": args.time_limit,
                "min_fidelity": 0.95,
            }

            for workflow_name, workflow in workflows.items():
                print(f"\n   Optimizing {workflow_name}:")

                best_allocation, best_score = optimizer.optimize_workflow_allocation(
                    workflow, constraints
                )

                if best_allocation:
                    allocation_name, provider_allocation = best_allocation
                    print(f"     Best strategy: {allocation_name}")
                    print(f"     Score: {best_score:.2f}")
                    print(f"     Provider allocation:")

                    for provider, task_ids in provider_allocation.items():
                        print(f"       {provider}: {len(task_ids)} tasks")

        # Execute workflows
        workflow_results = {}

        print(f"\n🚀 Executing workflows...")

        for workflow_name, workflow in workflows.items():
            if args.verbose:
                print(f"\n📊 Executing {workflow_name}...")

            start_time = time.time()
            results = workflow_manager.execute_workflow(
                workflow, args.max_parallel_jobs
            )
            execution_time = time.time() - start_time

            results["actual_execution_time"] = execution_time
            workflow_results[workflow_name] = results

            print(
                f"   ✅ {workflow_name}: {results['successful_tasks']}/{results['total_tasks']} tasks successful"
            )
            print(f"   💰 Cost: ${results['total_cost']:.2f}")
            print(f"   ⏱️  Time: {execution_time:.1f}s")

            if results["provider_statistics"]:
                print(f"   🌐 Provider usage:")
                for provider, stats in results["provider_statistics"].items():
                    print(
                        f"     {provider}: {stats['tasks']} tasks, ${stats['cost']:.2f}"
                    )

        # Strategy comparison
        if args.compare_strategies:
            print(f"\n🔄 Comparing allocation strategies...")

            # This would involve running the same workflow with different strategies
            # For demonstration, we'll show the concept
            strategies = ["cost_optimized", "speed_optimized", "balanced"]

            for strategy in strategies:
                print(f"\n   Strategy: {strategy}")
                # In a real implementation, you would re-run workflows with different priorities
                print(f"     Estimated performance based on provider selection")

        # Calculate resource statistics
        resource_stats = {
            "total_providers": len(resource_manager.providers),
            "total_workflows": len(workflow_results),
            "total_cost": sum(r["total_cost"] for r in workflow_results.values()),
            "avg_success_rate": np.mean(
                [
                    r["successful_tasks"] / r["total_tasks"]
                    for r in workflow_results.values()
                ]
            ),
        }

        print(f"\n📈 Overall Statistics:")
        print(f"   Total providers: {resource_stats['total_providers']}")
        print(f"   Total workflows: {resource_stats['total_workflows']}")
        print(f"   Total cost: ${resource_stats['total_cost']:.2f}")
        print(f"   Average success rate: {resource_stats['avg_success_rate']:.1%}")

        # Visualization
        if args.show_visualization:
            visualize_workflow_results(workflow_results, resource_stats)

        print(f"\n📚 Key Insights:")
        print(f"   • Hybrid workflows enable optimal resource utilization")
        print(f"   • Different providers excel at different tasks")
        print(f"   • Parallel execution significantly reduces total time")
        print(f"   • Cost optimization vs performance is a key trade-off")

        print(f"\n🎯 Best Practices:")
        print(f"   • Match algorithm requirements to provider strengths")
        print(f"   • Use simulators for development and validation")
        print(f"   • Implement robust error handling and fallbacks")
        print(f"   • Monitor costs and performance continuously")
        print(f"   • Consider queue times in production scheduling")

        print(f"\n✅ Hybrid cloud workflow demonstration completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
