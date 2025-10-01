#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms
Example 2: AWS Braket Integration

Implementation of Amazon Braket integration with multiple quantum providers.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import json
from datetime import datetime

try:
    import boto3
    from braket.aws import AwsDevice
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    from braket.tasks import GateModelQuantumTaskResult
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("Warning: AWS Braket not available. Using simulation mode.")
    boto3 = None
    AwsDevice = None
    Circuit = None
    LocalSimulator = None

from qiskit import QuantumCircuit
import warnings

warnings.filterwarnings("ignore")


class BraketManager:
    def __init__(self, aws_region="us-east-1", s3_bucket=None, verbose=False):
        self.aws_region = aws_region
        self.s3_bucket = s3_bucket
        self.verbose = verbose
        self.session = None
        self.available_devices = []
        self.providers = {
            "IonQ": [],
            "Rigetti": [],
            "OQC": [],
            "Xanadu": [],
            "Simulators": [],
        }

    def authenticate(self):
        """Authenticate with AWS services."""
        try:
            if boto3:
                # Initialize session
                self.session = boto3.Session(region_name=self.aws_region)

                # Test credentials
                sts = self.session.client("sts")
                identity = sts.get_caller_identity()

                if self.verbose:
                    print(f"✅ AWS authentication successful")
                    print(f"   Account: {identity.get('Account', 'Unknown')}")
                    print(f"   Region: {self.aws_region}")

                return True
            else:
                if self.verbose:
                    print("⚠️  AWS Braket not available, using local simulation")
                return False

        except (ClientError, NoCredentialsError) as e:
            if self.verbose:
                print(f"❌ AWS authentication failed: {e}")
                print("💡 Using local simulation mode")
            return False
        except Exception as e:
            if self.verbose:
                print(f"❌ Unexpected error: {e}")
            return False

    def discover_devices(self):
        """Discover available quantum devices across providers."""
        devices = []

        if boto3 and self.session:
            try:
                # Get available devices
                from braket.aws import AwsDevice

                # Simulators
                simulator_arns = [
                    "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                    "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
                    "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
                ]

                for arn in simulator_arns:
                    try:
                        device = AwsDevice(arn)
                        device_info = {
                            "name": device.name,
                            "arn": arn,
                            "type": "SIMULATOR",
                            "provider": "Amazon",
                            "status": device.status,
                            "properties": (
                                device.properties.dict()
                                if hasattr(device, "properties")
                                else {}
                            ),
                        }
                        devices.append(device_info)
                        self.providers["Simulators"].append(device_info)
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️  Could not access simulator {arn}: {e}")

                # Hardware devices (would require actual device ARNs)
                # Note: Real device ARNs change and require permissions
                hardware_examples = [
                    {
                        "name": "IonQ Device (Example)",
                        "arn": "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony",
                        "type": "QPU",
                        "provider": "IonQ",
                        "status": "OFFLINE",  # Example status
                        "properties": {"qubit_count": 11, "connectivity": "all-to-all"},
                    },
                    {
                        "name": "Rigetti Device (Example)",
                        "arn": "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
                        "type": "QPU",
                        "provider": "Rigetti",
                        "status": "OFFLINE",  # Example status
                        "properties": {"qubit_count": 80, "connectivity": "limited"},
                    },
                ]

                for device_info in hardware_examples:
                    devices.append(device_info)
                    provider = device_info["provider"]
                    if provider in self.providers:
                        self.providers[provider].append(device_info)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Device discovery error: {e}")

        # Local simulator fallback
        if not devices or not boto3:
            local_sim = {
                "name": "Local Simulator",
                "arn": "local:simulator",
                "type": "SIMULATOR",
                "provider": "Local",
                "status": "AVAILABLE",
                "properties": {"qubit_count": 25, "shot_limit": 1000000},
            }
            devices.append(local_sim)
            self.providers["Simulators"].append(local_sim)

        self.available_devices = devices
        return devices

    def create_braket_circuit(self, qiskit_circuit):
        """Convert Qiskit circuit to Braket circuit."""
        if not Circuit:
            return None

        # Create Braket circuit
        n_qubits = qiskit_circuit.num_qubits
        braket_circuit = Circuit()

        # Convert Qiskit instructions to Braket
        for instruction in qiskit_circuit.data:
            gate = instruction[0]
            qubits = instruction[1]

            if gate.name == "h":
                braket_circuit.h(qubits[0].index)
            elif gate.name == "x":
                braket_circuit.x(qubits[0].index)
            elif gate.name == "y":
                braket_circuit.y(qubits[0].index)
            elif gate.name == "z":
                braket_circuit.z(qubits[0].index)
            elif gate.name == "cx":
                braket_circuit.cnot(qubits[0].index, qubits[1].index)
            elif gate.name == "ry":
                angle = float(gate.params[0])
                braket_circuit.ry(qubits[0].index, angle)
            elif gate.name == "rz":
                angle = float(gate.params[0])
                braket_circuit.rz(qubits[0].index, angle)
            elif gate.name == "measure":
                # Braket handles measurements differently
                pass

        return braket_circuit

    def submit_task(self, circuit, device_arn, shots=1000):
        """Submit quantum task to Braket device."""
        task_result = {}

        try:
            if device_arn == "local:simulator":
                # Use local simulator
                if LocalSimulator:
                    device = LocalSimulator()

                    if self.verbose:
                        print(f"🔬 Running on local simulator...")

                    task = device.run(circuit, shots=shots)
                    result = task.result()

                    task_result = {
                        "task_id": f"local_{int(time.time())}",
                        "device_arn": device_arn,
                        "status": "COMPLETED",
                        "shots": shots,
                        "result": result,
                        "success": True,
                    }
                else:
                    raise Exception("Local simulator not available")

            elif boto3 and self.session:
                # Use AWS Braket
                device = AwsDevice(device_arn)

                if self.verbose:
                    print(f"🚀 Submitting task to {device.name}...")

                task = device.run(
                    circuit,
                    shots=shots,
                    s3_destination_folder=(self.s3_bucket, "braket-results"),
                )

                if self.verbose:
                    print(f"📋 Task ID: {task.id}")
                    print("⏳ Waiting for completion...")

                # Wait for completion (with timeout)
                start_time = time.time()
                timeout = 300  # 5 minutes

                while task.state() not in ["COMPLETED", "FAILED", "CANCELLED"]:
                    if time.time() - start_time > timeout:
                        raise Exception("Task timeout")
                    time.sleep(5)

                    if self.verbose:
                        print(f"   Status: {task.state()}")

                if task.state() == "COMPLETED":
                    result = task.result()

                    task_result = {
                        "task_id": task.id,
                        "device_arn": device_arn,
                        "status": task.state(),
                        "shots": shots,
                        "result": result,
                        "success": True,
                    }
                else:
                    task_result = {
                        "task_id": task.id,
                        "device_arn": device_arn,
                        "status": task.state(),
                        "shots": shots,
                        "success": False,
                        "error": f"Task failed with status: {task.state()}",
                    }
            else:
                raise Exception("AWS Braket not available")

        except Exception as e:
            if self.verbose:
                print(f"❌ Task submission failed: {e}")

            task_result = {
                "task_id": None,
                "device_arn": device_arn,
                "status": "FAILED",
                "shots": shots,
                "success": False,
                "error": str(e),
            }

        return task_result

    def get_device_capabilities(self, device_arn):
        """Get device capabilities and properties."""
        capabilities = {}

        try:
            if device_arn == "local:simulator":
                capabilities = {
                    "name": "Local Simulator",
                    "type": "SIMULATOR",
                    "provider": "Local",
                    "qubit_count": 25,
                    "supported_gates": ["H", "X", "Y", "Z", "CNOT", "RX", "RY", "RZ"],
                    "connectivity": "all-to-all",
                    "shot_limit": 1000000,
                    "coherence_time": None,
                    "gate_fidelity": 1.0,
                }
            elif boto3 and AwsDevice:
                device = AwsDevice(device_arn)
                properties = device.properties

                capabilities = {
                    "name": device.name,
                    "type": device.type,
                    "provider": device.provider_name,
                    "status": device.status,
                    "properties": (
                        properties.dict()
                        if hasattr(properties, "dict")
                        else str(properties)
                    ),
                }

                # Extract specific capabilities if available
                if hasattr(properties, "paradigm"):
                    paradigm = properties.paradigm
                    if hasattr(paradigm, "qubit_count"):
                        capabilities["qubit_count"] = paradigm.qubit_count
                    if hasattr(paradigm, "connectivity"):
                        capabilities["connectivity"] = str(paradigm.connectivity)

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error getting device capabilities: {e}")

            capabilities = {
                "name": "Unknown Device",
                "type": "UNKNOWN",
                "error": str(e),
            }

        return capabilities


class BraketAnalyzer:
    def __init__(self, manager, verbose=False):
        self.manager = manager
        self.verbose = verbose

    def compare_providers(self, test_circuits=None):
        """Compare different quantum providers on Braket."""
        if not test_circuits:
            test_circuits = [
                self.create_bell_circuit(),
                self.create_ghz_circuit(3),
                self.create_random_circuit(3, 5),
            ]

        comparison_results = {}
        available_devices = self.manager.available_devices

        # Test each available device
        for device in available_devices[:3]:  # Limit to avoid long execution
            device_name = device["name"]
            device_arn = device["arn"]

            if self.verbose:
                print(f"\n🔍 Testing {device_name}...")

            device_results = {
                "capabilities": self.manager.get_device_capabilities(device_arn),
                "circuit_results": [],
            }

            for i, circuit in enumerate(test_circuits):
                circuit_name = f"circuit_{i+1}"

                if self.verbose:
                    print(f"   Running {circuit_name}...")

                # Convert to Braket circuit if needed
                if isinstance(circuit, QuantumCircuit):
                    braket_circuit = self.manager.create_braket_circuit(circuit)
                else:
                    braket_circuit = circuit

                if braket_circuit:
                    start_time = time.time()
                    task_result = self.manager.submit_task(
                        braket_circuit, device_arn, shots=100
                    )
                    execution_time = time.time() - start_time

                    circuit_result = {
                        "circuit_name": circuit_name,
                        "execution_time": execution_time,
                        "success": task_result["success"],
                        "task_id": task_result.get("task_id"),
                        "shots": task_result["shots"],
                    }

                    if task_result["success"] and "result" in task_result:
                        result = task_result["result"]

                        # Extract measurement counts
                        if hasattr(result, "measurement_counts"):
                            counts = result.measurement_counts
                            circuit_result["counts"] = dict(counts)
                            circuit_result["total_shots"] = sum(counts.values())
                        elif hasattr(result, "get_counts"):
                            counts = result.get_counts()
                            circuit_result["counts"] = counts
                            circuit_result["total_shots"] = sum(counts.values())
                        else:
                            circuit_result["counts"] = {}
                            circuit_result["total_shots"] = 0
                    else:
                        circuit_result["error"] = task_result.get(
                            "error", "Unknown error"
                        )

                    device_results["circuit_results"].append(circuit_result)
                else:
                    if self.verbose:
                        print(f"   ❌ Circuit conversion failed")

            comparison_results[device_name] = device_results

        return comparison_results

    def create_bell_circuit(self):
        """Create Bell state circuit."""
        # Return Qiskit circuit (will be converted to Braket)
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    def create_ghz_circuit(self, n_qubits):
        """Create GHZ state circuit."""
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        qc.measure_all()
        return qc

    def create_random_circuit(self, n_qubits, depth):
        """Create random quantum circuit."""
        qc = QuantumCircuit(n_qubits, n_qubits)

        gates = ["h", "x", "y", "z", "ry", "rz"]

        for _ in range(depth):
            gate = np.random.choice(gates)
            qubit = np.random.randint(n_qubits)

            if gate in ["h", "x", "y", "z"]:
                getattr(qc, gate)(qubit)
            elif gate in ["ry", "rz"]:
                angle = np.random.uniform(0, 2 * np.pi)
                getattr(qc, gate)(angle, qubit)

            # Add some CNOT gates
            if np.random.random() < 0.3 and n_qubits > 1:
                control = np.random.randint(n_qubits)
                target = np.random.randint(n_qubits)
                if control != target:
                    qc.cx(control, target)

        qc.measure_all()
        return qc

    def analyze_costs(self, provider_results):
        """Analyze cost implications of different providers."""
        cost_analysis = {}

        # Estimated costs (these would be real in production)
        cost_per_shot = {
            "Local": 0.0,
            "Amazon": 0.00075,  # Simulator cost estimate
            "IonQ": 0.01,  # Hardware cost estimate
            "Rigetti": 0.005,  # Hardware cost estimate
            "OQC": 0.008,  # Hardware cost estimate
        }

        for device_name, results in provider_results.items():
            provider = results["capabilities"].get("provider", "Unknown")

            total_shots = 0
            total_time = 0
            successful_runs = 0

            for circuit_result in results["circuit_results"]:
                if circuit_result["success"]:
                    total_shots += circuit_result.get("total_shots", 0)
                    total_time += circuit_result["execution_time"]
                    successful_runs += 1

            cost_per_shot_provider = cost_per_shot.get(provider, 0.01)
            estimated_cost = total_shots * cost_per_shot_provider

            cost_analysis[device_name] = {
                "provider": provider,
                "total_shots": total_shots,
                "total_time": total_time,
                "successful_runs": successful_runs,
                "cost_per_shot": cost_per_shot_provider,
                "estimated_cost": estimated_cost,
                "shots_per_second": total_shots / total_time if total_time > 0 else 0,
            }

        return cost_analysis

    def visualize_results(self, provider_results, cost_analysis=None):
        """Visualize provider comparison results."""
        fig = plt.figure(figsize=(16, 12))

        # Provider performance comparison
        ax1 = plt.subplot(2, 3, 1)

        device_names = []
        execution_times = []
        success_rates = []

        for device_name, results in provider_results.items():
            circuit_results = results["circuit_results"]
            if circuit_results:
                device_names.append(device_name.replace(" ", "\n"))

                avg_time = np.mean([r["execution_time"] for r in circuit_results])
                execution_times.append(avg_time)

                success_count = sum(1 for r in circuit_results if r["success"])
                success_rate = success_count / len(circuit_results)
                success_rates.append(success_rate)

        if device_names:
            colors = [
                "green" if sr >= 0.8 else "orange" if sr >= 0.5 else "red"
                for sr in success_rates
            ]
            bars = ax1.bar(device_names, execution_times, color=colors, alpha=0.7)

            ax1.set_ylabel("Avg Execution Time (s)")
            ax1.set_title("Provider Performance Comparison")
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, time_val in zip(bars, execution_times):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time_val:.2f}s",
                    ha="center",
                    va="bottom",
                )

        # Success rates
        ax2 = plt.subplot(2, 3, 2)

        if device_names and success_rates:
            bars = ax2.bar(device_names, success_rates, alpha=0.7, color="skyblue")
            ax2.set_ylabel("Success Rate")
            ax2.set_title("Task Success Rates")
            ax2.set_ylim(0, 1.1)
            ax2.tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.1%}",
                    ha="center",
                    va="bottom",
                )

        # Device capabilities
        ax3 = plt.subplot(2, 3, 3)

        qubit_counts = []
        device_types = []

        for device_name, results in provider_results.items():
            capabilities = results["capabilities"]
            qubit_count = capabilities.get("qubit_count", 0)
            device_type = capabilities.get("type", "UNKNOWN")

            if qubit_count > 0:
                qubit_counts.append(qubit_count)
                device_types.append(device_type)

        if qubit_counts:
            # Group by device type
            simulator_qubits = [
                q for q, t in zip(qubit_counts, device_types) if t == "SIMULATOR"
            ]
            hardware_qubits = [
                q for q, t in zip(qubit_counts, device_types) if t == "QPU"
            ]

            data_to_plot = []
            labels = []

            if simulator_qubits:
                data_to_plot.append(simulator_qubits)
                labels.append("Simulators")
            if hardware_qubits:
                data_to_plot.append(hardware_qubits)
                labels.append("Hardware")

            if data_to_plot:
                ax3.boxplot(data_to_plot, labels=labels)
                ax3.set_ylabel("Number of Qubits")
                ax3.set_title("Device Qubit Capacity")

        # Cost analysis (if available)
        if cost_analysis:
            ax4 = plt.subplot(2, 3, 4)

            providers = []
            costs = []

            for device_name, cost_info in cost_analysis.items():
                providers.append(device_name.replace(" ", "\n"))
                costs.append(cost_info["estimated_cost"])

            if providers:
                bars = ax4.bar(providers, costs, alpha=0.7, color="gold")
                ax4.set_ylabel("Estimated Cost ($)")
                ax4.set_title("Cost Comparison")
                ax4.tick_params(axis="x", rotation=45)

                # Add value labels
                for bar, cost in zip(bars, costs):
                    height = bar.get_height()
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"${cost:.3f}",
                        ha="center",
                        va="bottom",
                    )

        # Throughput analysis
        ax5 = plt.subplot(2, 3, 5)

        if cost_analysis:
            providers = []
            throughputs = []

            for device_name, cost_info in cost_analysis.items():
                if cost_info["shots_per_second"] > 0:
                    providers.append(device_name.replace(" ", "\n"))
                    throughputs.append(cost_info["shots_per_second"])

            if providers:
                bars = ax5.bar(providers, throughputs, alpha=0.7, color="lightcoral")
                ax5.set_ylabel("Shots per Second")
                ax5.set_title("Throughput Comparison")
                ax5.tick_params(axis="x", rotation=45)

        # Summary and insights
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        summary_text = "AWS Braket Summary:\n\n"

        if provider_results:
            total_devices = len(provider_results)
            successful_devices = sum(
                1
                for results in provider_results.values()
                if any(r["success"] for r in results["circuit_results"])
            )

            summary_text += f"Devices Tested: {total_devices}\n"
            summary_text += f"Successful: {successful_devices}\n\n"

        summary_text += "Provider Features:\n\n"
        summary_text += "Amazon Simulators:\n"
        summary_text += "• High-performance\n"
        summary_text += "• Cost-effective\n"
        summary_text += "• No queue times\n\n"

        summary_text += "IonQ Hardware:\n"
        summary_text += "• All-to-all connectivity\n"
        summary_text += "• High fidelity\n"
        summary_text += "• Trapped ion technology\n\n"

        summary_text += "Rigetti Hardware:\n"
        summary_text += "• Superconducting qubits\n"
        summary_text += "• Fast gate operations\n"
        summary_text += "• Parametric gates\n\n"

        summary_text += "Best Practices:\n"
        summary_text += "• Compare costs carefully\n"
        summary_text += "• Use simulators for development\n"
        summary_text += "• Consider queue times\n"
        summary_text += "• Optimize for provider strengths"

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
    parser = argparse.ArgumentParser(description="AWS Braket Integration")
    parser.add_argument(
        "--aws-region", type=str, default="us-east-1", help="AWS region"
    )
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket for results")
    parser.add_argument(
        "--device-arn", type=str, default="local:simulator", help="Device ARN to use"
    )
    parser.add_argument("--shots", type=int, default=100, help="Number of shots")
    parser.add_argument(
        "--compare-providers", action="store_true", help="Compare different providers"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available devices"
    )
    parser.add_argument("--show-costs", action="store_true", help="Show cost analysis")
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms")
    print("Example 2: AWS Braket Integration")
    print("=" * 49)

    try:
        # Check for S3 bucket in environment if not provided
        if not args.s3_bucket:
            args.s3_bucket = os.getenv("AWS_BRAKET_S3_BUCKET")

        # Initialize Braket manager
        manager = BraketManager(
            aws_region=args.aws_region, s3_bucket=args.s3_bucket, verbose=args.verbose
        )

        # Authenticate
        auth_success = manager.authenticate()

        if auth_success:
            print("✅ AWS Braket authentication successful")
        else:
            print("⚠️  Using local simulation mode")

        # Discover devices
        print(f"\n🔍 Discovering quantum devices...")
        devices = manager.discover_devices()

        if args.list_devices:
            print(f"\n📋 Available Devices:")

            for provider, provider_devices in manager.providers.items():
                if provider_devices:
                    print(f"\n   {provider}:")
                    for device in provider_devices:
                        status_icon = (
                            "🟢" if device.get("status") == "AVAILABLE" else "🔴"
                        )
                        qubit_info = (
                            f" ({device.get('properties', {}).get('qubit_count', '?')} qubits)"
                            if "properties" in device
                            else ""
                        )
                        print(f"     {status_icon} {device['name']}{qubit_info}")
                        print(f"        ARN: {device['arn']}")

        # Get device capabilities
        print(f"\n🔍 Device Capabilities: {args.device_arn}")
        capabilities = manager.get_device_capabilities(args.device_arn)

        for key, value in capabilities.items():
            if key == "properties" and isinstance(value, dict):
                print(f"   {key}:")
                for prop_key, prop_value in value.items():
                    print(f"     {prop_key}: {prop_value}")
            else:
                print(f"   {key}: {value}")

        # Initialize analyzer
        analyzer = BraketAnalyzer(manager, verbose=args.verbose)

        # Compare providers
        provider_results = None
        cost_analysis = None

        if args.compare_providers:
            print(f"\n🔄 Comparing quantum providers...")

            provider_results = analyzer.compare_providers()

            print(f"\n📊 Provider Comparison Results:")
            for device_name, results in provider_results.items():
                print(f"\n   {device_name}:")

                capabilities = results["capabilities"]
                print(f"     Provider: {capabilities.get('provider', 'Unknown')}")
                print(f"     Type: {capabilities.get('type', 'Unknown')}")
                print(f"     Qubits: {capabilities.get('qubit_count', 'Unknown')}")

                circuit_results = results["circuit_results"]
                successful = sum(1 for r in circuit_results if r["success"])
                total = len(circuit_results)

                print(
                    f"     Success rate: {successful}/{total} ({successful/total:.1%})"
                )

                if successful > 0:
                    avg_time = np.mean(
                        [r["execution_time"] for r in circuit_results if r["success"]]
                    )
                    print(f"     Avg execution time: {avg_time:.2f}s")

            # Cost analysis
            if args.show_costs:
                print(f"\n💰 Cost Analysis:")
                cost_analysis = analyzer.analyze_costs(provider_results)

                for device_name, cost_info in cost_analysis.items():
                    print(f"\n   {device_name}:")
                    print(f"     Total shots: {cost_info['total_shots']}")
                    print(f"     Cost per shot: ${cost_info['cost_per_shot']:.5f}")
                    print(
                        f"     Estimated total cost: ${cost_info['estimated_cost']:.3f}"
                    )
                    print(
                        f"     Throughput: {cost_info['shots_per_second']:.1f} shots/s"
                    )

        # Single device test
        if not args.compare_providers:
            print(f"\n🚀 Testing device: {args.device_arn}")

            # Create test circuit
            test_circuit = analyzer.create_bell_circuit()
            braket_circuit = manager.create_braket_circuit(test_circuit)

            if braket_circuit:
                print(f"   Circuit qubits: {test_circuit.num_qubits}")
                print(f"   Circuit depth: {test_circuit.depth()}")

                # Submit task
                task_result = manager.submit_task(
                    braket_circuit, args.device_arn, shots=args.shots
                )

                if task_result["success"]:
                    print(f"✅ Task completed successfully")
                    print(f"   Task ID: {task_result['task_id']}")
                    print(f"   Shots: {task_result['shots']}")

                    if "result" in task_result:
                        result = task_result["result"]

                        # Get measurement counts
                        if hasattr(result, "measurement_counts"):
                            counts = dict(result.measurement_counts)
                        elif hasattr(result, "get_counts"):
                            counts = result.get_counts()
                        else:
                            counts = {}

                        if counts:
                            print(f"\n📊 Measurement Results:")
                            total_shots = sum(counts.values())
                            for state, count in sorted(counts.items()):
                                probability = count / total_shots
                                print(f"   |{state}⟩: {count} ({probability:.3f})")
                else:
                    print(
                        f"❌ Task failed: {task_result.get('error', 'Unknown error')}"
                    )
            else:
                print(f"❌ Circuit conversion failed")

        # Visualization
        if args.show_visualization and provider_results:
            analyzer.visualize_results(provider_results, cost_analysis)

        print(f"\n📚 Key Insights:")
        print(f"   • AWS Braket provides access to multiple quantum providers")
        print(f"   • Different providers have unique strengths and capabilities")
        print(f"   • Cost and performance vary significantly between providers")
        print(f"   • Local simulators are excellent for development and testing")

        print(f"\n🎯 Provider Selection Guidelines:")
        print(f"   • IonQ: High-fidelity, all-to-all connectivity, premium cost")
        print(f"   • Rigetti: Fast gates, superconducting, moderate cost")
        print(f"   • Amazon Simulators: High performance, cost-effective")
        print(f"   • Choose based on algorithm requirements and budget")

        print(f"\n✅ AWS Braket integration demonstration completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
