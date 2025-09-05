#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms
Example 1: IBM Quantum Platform Access

Implementation of IBM Quantum platform access, backend management, and job execution.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime
try:
    from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Estimator
    from qiskit_ibm_provider import IBMProvider
    from qiskit.providers.jobstatus import JobStatus
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.visualization import plot_histogram
    import qiskit.tools.monitor as monitor
except ImportError:
    print("Warning: Qiskit IBM Runtime not available. Using simulation mode.")
    QiskitRuntimeService = None
    IBMProvider = None

import warnings
warnings.filterwarnings('ignore')

class IBMQuantumManager:
    def __init__(self, token=None, instance=None, verbose=False):
        self.token = token
        self.instance = instance
        self.verbose = verbose
        self.service = None
        self.provider = None
        self.backends = {}
        self.available_backends = []
        
    def authenticate(self):
        """Authenticate with IBM Quantum services."""
        try:
            if self.token:
                # Save token for future use
                if QiskitRuntimeService:
                    QiskitRuntimeService.save_account(
                        token=self.token, 
                        instance=self.instance,
                        overwrite=True
                    )
                    
            # Initialize service
            if QiskitRuntimeService:
                self.service = QiskitRuntimeService()
                if self.verbose:
                    print("✅ IBM Quantum Runtime service initialized")
                    
            # Also try legacy provider for comparison
            if IBMProvider:
                try:
                    self.provider = IBMProvider()
                    if self.verbose:
                        print("✅ IBM Quantum provider initialized")
                except:
                    if self.verbose:
                        print("⚠️  Legacy IBM provider not available")
                        
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Authentication failed: {e}")
                print("💡 Using simulation mode")
            return False
    
    def get_available_backends(self, simulator=True, real_hardware=False):
        """Get list of available quantum backends."""
        backends = []
        
        try:
            if self.service:
                # Get all backends
                all_backends = self.service.backends()
                
                for backend in all_backends:
                    backend_info = {
                        'name': backend.name,
                        'status': 'operational' if backend.status().operational else 'offline',
                        'pending_jobs': backend.status().pending_jobs if hasattr(backend.status(), 'pending_jobs') else 0,
                        'simulator': backend.configuration().simulator,
                        'n_qubits': backend.configuration().n_qubits,
                        'quantum_volume': getattr(backend.configuration(), 'quantum_volume', None)
                    }
                    
                    # Filter based on preferences
                    if simulator and backend_info['simulator']:
                        backends.append(backend_info)
                    elif real_hardware and not backend_info['simulator']:
                        backends.append(backend_info)
                        
            else:
                # Fallback simulation backends
                backends = [
                    {
                        'name': 'aer_simulator',
                        'status': 'operational',
                        'pending_jobs': 0,
                        'simulator': True,
                        'n_qubits': 32,
                        'quantum_volume': None
                    },
                    {
                        'name': 'qasm_simulator',
                        'status': 'operational', 
                        'pending_jobs': 0,
                        'simulator': True,
                        'n_qubits': 32,
                        'quantum_volume': None
                    }
                ]
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error getting backends: {e}")
                
        self.available_backends = backends
        return backends
    
    def get_backend_properties(self, backend_name):
        """Get detailed properties of a specific backend."""
        properties = {}
        
        try:
            if self.service:
                backend = self.service.backend(backend_name)
                config = backend.configuration()
                
                properties = {
                    'name': backend_name,
                    'n_qubits': config.n_qubits,
                    'simulator': config.simulator,
                    'local': config.local,
                    'coupling_map': config.coupling_map,
                    'basis_gates': config.basis_gates,
                    'max_shots': config.max_shots,
                    'max_experiments': getattr(config, 'max_experiments', 1),
                    'quantum_volume': getattr(config, 'quantum_volume', None),
                    'processor_type': getattr(config, 'processor_type', 'unknown')
                }
                
                # Add calibration data if available
                try:
                    calibration = backend.properties()
                    if calibration:
                        properties['calibration_date'] = calibration.last_update_date
                        
                        # Average gate errors
                        gate_errors = []
                        for gate in calibration.gates:
                            if hasattr(gate, 'parameters'):
                                for param in gate.parameters:
                                    if param.name == 'gate_error':
                                        gate_errors.append(param.value)
                        
                        if gate_errors:
                            properties['avg_gate_error'] = np.mean(gate_errors)
                            properties['max_gate_error'] = np.max(gate_errors)
                        
                        # Readout errors
                        readout_errors = []
                        for qubit_info in calibration.qubits:
                            for param in qubit_info:
                                if param.name == 'readout_error':
                                    readout_errors.append(param.value)
                        
                        if readout_errors:
                            properties['avg_readout_error'] = np.mean(readout_errors)
                            properties['max_readout_error'] = np.max(readout_errors)
                            
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Could not get calibration data: {e}")
                        
            else:
                # Simulated properties
                properties = {
                    'name': backend_name,
                    'n_qubits': 32,
                    'simulator': True,
                    'local': True,
                    'coupling_map': None,
                    'basis_gates': ['cx', 'id', 'rz', 'sx', 'x'],
                    'max_shots': 8192,
                    'max_experiments': 300,
                    'quantum_volume': None,
                    'processor_type': 'simulator'
                }
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Error getting backend properties: {e}")
                
        return properties
    
    def submit_job(self, circuits, backend_name, shots=1000, optimization_level=1):
        """Submit quantum circuit job to backend."""
        job_results = {}
        
        try:
            if self.service:
                backend = self.service.backend(backend_name)
                
                # Transpile circuits for backend
                if isinstance(circuits, QuantumCircuit):
                    circuits = [circuits]
                
                transpiled_circuits = transpile(
                    circuits, 
                    backend=backend,
                    optimization_level=optimization_level
                )
                
                # Submit job using Runtime
                with Session(service=self.service, backend=backend_name) as session:
                    sampler = Sampler(session=session)
                    
                    if self.verbose:
                        print(f"🚀 Submitting job to {backend_name}...")
                    
                    job = sampler.run(transpiled_circuits, shots=shots)
                    
                    if self.verbose:
                        print(f"📋 Job ID: {job.job_id()}")
                        print("⏳ Waiting for results...")
                    
                    # Wait for completion
                    result = job.result()
                    
                    job_results = {
                        'job_id': job.job_id(),
                        'backend': backend_name,
                        'shots': shots,
                        'success': True,
                        'result': result,
                        'transpiled_circuits': transpiled_circuits,
                        'original_circuits': circuits
                    }
                    
            else:
                # Simulation fallback
                from qiskit_aer import AerSimulator
                simulator = AerSimulator()
                
                if isinstance(circuits, QuantumCircuit):
                    circuits = [circuits]
                
                transpiled_circuits = transpile(circuits, simulator)
                
                if self.verbose:
                    print(f"🔬 Running simulation with {shots} shots...")
                
                job = simulator.run(transpiled_circuits, shots=shots)
                result = job.result()
                
                job_results = {
                    'job_id': 'sim_' + str(int(time.time())),
                    'backend': 'aer_simulator',
                    'shots': shots,
                    'success': True,
                    'result': result,
                    'transpiled_circuits': transpiled_circuits,
                    'original_circuits': circuits
                }
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Job submission failed: {e}")
            
            job_results = {
                'job_id': None,
                'backend': backend_name,
                'shots': shots,
                'success': False,
                'error': str(e)
            }
            
        return job_results
    
    def monitor_job(self, job_id):
        """Monitor job status and progress."""
        if not self.service:
            return {'status': 'completed', 'message': 'Simulation completed instantly'}
            
        try:
            job = self.service.job(job_id)
            status = job.status()
            
            status_info = {
                'job_id': job_id,
                'status': status.name,
                'creation_date': job.creation_date,
                'queue_position': getattr(job, 'queue_position', lambda: None)()
            }
            
            if self.verbose:
                print(f"📊 Job {job_id}: {status.name}")
                if status_info['queue_position']:
                    print(f"   Queue position: {status_info['queue_position']}")
                    
            return status_info
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error monitoring job: {e}")
            return {'status': 'error', 'error': str(e)}

class IBMQuantumAnalyzer:
    def __init__(self, manager, verbose=False):
        self.manager = manager
        self.verbose = verbose
        
    def compare_backends(self, backends_to_compare=None):
        """Compare different IBM Quantum backends."""
        if not backends_to_compare:
            backends_to_compare = ['aer_simulator', 'qasm_simulator']
            
        comparison_results = {}
        
        # Create test circuit
        test_circuit = self.create_test_circuit()
        
        for backend_name in backends_to_compare:
            if self.verbose:
                print(f"\n🔍 Testing backend: {backend_name}")
                
            # Get properties
            properties = self.manager.get_backend_properties(backend_name)
            
            # Run test job
            start_time = time.time()
            job_result = self.manager.submit_job(test_circuit, backend_name, shots=1000)
            execution_time = time.time() - start_time
            
            if job_result['success']:
                # Analyze results
                counts = job_result['result'].get_counts()
                
                comparison_results[backend_name] = {
                    'properties': properties,
                    'execution_time': execution_time,
                    'counts': counts,
                    'total_shots': sum(counts.values()),
                    'success_rate': job_result['success'],
                    'circuit_depth': job_result['transpiled_circuits'][0].depth(),
                    'circuit_ops': len(job_result['transpiled_circuits'][0])
                }
                
                if self.verbose:
                    print(f"   ✅ Completed in {execution_time:.2f}s")
                    print(f"   📊 Circuit depth: {comparison_results[backend_name]['circuit_depth']}")
                    
            else:
                comparison_results[backend_name] = {
                    'properties': properties,
                    'execution_time': execution_time,
                    'error': job_result.get('error', 'Unknown error'),
                    'success_rate': False
                }
                
                if self.verbose:
                    print(f"   ❌ Failed: {job_result.get('error', 'Unknown error')}")
                    
        return comparison_results
    
    def create_test_circuit(self):
        """Create a test circuit for backend comparison."""
        # Bell state circuit with measurements
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        return qc
    
    def benchmark_performance(self, backend_name, circuits=None, shots_list=[100, 500, 1000]):
        """Benchmark backend performance with different parameters."""
        if not circuits:
            circuits = [
                self.create_test_circuit(),
                self.create_ghz_circuit(3),
                self.create_random_circuit(4, 10)
            ]
            
        benchmark_results = {}
        
        for i, circuit in enumerate(circuits):
            circuit_name = f"circuit_{i+1}"
            benchmark_results[circuit_name] = {}
            
            for shots in shots_list:
                if self.verbose:
                    print(f"🔬 Benchmarking {circuit_name} with {shots} shots...")
                    
                start_time = time.time()
                job_result = self.manager.submit_job(circuit, backend_name, shots=shots)
                execution_time = time.time() - start_time
                
                if job_result['success']:
                    counts = job_result['result'].get_counts()
                    
                    # Calculate metrics
                    total_counts = sum(counts.values())
                    entropy = self.calculate_entropy(counts)
                    
                    benchmark_results[circuit_name][shots] = {
                        'execution_time': execution_time,
                        'counts': counts,
                        'total_shots': total_counts,
                        'entropy': entropy,
                        'success': True,
                        'shots_per_second': total_counts / execution_time if execution_time > 0 else 0
                    }
                else:
                    benchmark_results[circuit_name][shots] = {
                        'execution_time': execution_time,
                        'success': False,
                        'error': job_result.get('error', 'Unknown error')
                    }
                    
        return benchmark_results
    
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
        
        gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']
        
        for _ in range(depth):
            gate = np.random.choice(gates)
            
            if gate in ['h', 'x', 'y', 'z']:
                qubit = np.random.randint(n_qubits)
                getattr(qc, gate)(qubit)
            elif gate in ['rx', 'ry', 'rz']:
                qubit = np.random.randint(n_qubits)
                angle = np.random.uniform(0, 2*np.pi)
                getattr(qc, gate)(angle, qubit)
            elif gate == 'cx':
                control = np.random.randint(n_qubits)
                target = np.random.randint(n_qubits)
                if control != target:
                    qc.cx(control, target)
                    
        qc.measure_all()
        return qc
    
    def calculate_entropy(self, counts):
        """Calculate Shannon entropy of measurement results."""
        total = sum(counts.values())
        if total == 0:
            return 0
            
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
                
        return entropy
    
    def visualize_results(self, comparison_results, benchmark_results=None):
        """Visualize backend comparison and benchmark results."""
        fig = plt.figure(figsize=(16, 12))
        
        # Backend comparison
        ax1 = plt.subplot(2, 3, 1)
        
        backend_names = []
        execution_times = []
        success_rates = []
        
        for backend, results in comparison_results.items():
            if 'execution_time' in results:
                backend_names.append(backend.replace('_', '\n'))
                execution_times.append(results['execution_time'])
                success_rates.append(1.0 if results['success_rate'] else 0.0)
        
        if backend_names:
            bars = ax1.bar(backend_names, execution_times, 
                          color=['green' if sr == 1.0 else 'red' for sr in success_rates])
            
            ax1.set_ylabel('Execution Time (s)')
            ax1.set_title('Backend Execution Time Comparison')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, execution_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.2f}s', ha='center', va='bottom')
        
        # Backend properties comparison
        ax2 = plt.subplot(2, 3, 2)
        
        qubit_counts = []
        quantum_volumes = []
        
        for backend, results in comparison_results.items():
            if 'properties' in results:
                props = results['properties']
                qubit_counts.append(props.get('n_qubits', 0))
                quantum_volumes.append(props.get('quantum_volume', 0) or 0)
        
        if backend_names and qubit_counts:
            x = np.arange(len(backend_names))
            width = 0.35
            
            ax2.bar(x - width/2, qubit_counts, width, label='Qubits', alpha=0.7)
            ax2_twin = ax2.twinx()
            ax2_twin.bar(x + width/2, quantum_volumes, width, 
                        label='Quantum Volume', alpha=0.7, color='orange')
            
            ax2.set_xlabel('Backend')
            ax2.set_ylabel('Number of Qubits')
            ax2_twin.set_ylabel('Quantum Volume')
            ax2.set_title('Backend Specifications')
            ax2.set_xticks(x)
            ax2.set_xticklabels(backend_names)
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        
        # Measurement results
        ax3 = plt.subplot(2, 3, 3)
        
        for i, (backend, results) in enumerate(comparison_results.items()):
            if 'counts' in results:
                counts = results['counts']
                states = list(counts.keys())
                counts_values = list(counts.values())
                
                # Show only top 8 states for clarity
                if len(states) > 8:
                    sorted_items = sorted(zip(counts_values, states), reverse=True)[:8]
                    counts_values, states = zip(*sorted_items)
                
                x_pos = np.arange(len(states)) + i * 0.3
                ax3.bar(x_pos, counts_values, width=0.25, 
                       label=backend, alpha=0.7)
        
        if any('counts' in results for results in comparison_results.values()):
            ax3.set_xlabel('Quantum States')
            ax3.set_ylabel('Counts')
            ax3.set_title('Measurement Results Comparison')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
        
        # Benchmark results (if provided)
        if benchmark_results:
            ax4 = plt.subplot(2, 3, 4)
            
            # Execution time vs shots
            for circuit_name, circuit_results in benchmark_results.items():
                shots_list = []
                times_list = []
                
                for shots, results in circuit_results.items():
                    if results.get('success', False):
                        shots_list.append(shots)
                        times_list.append(results['execution_time'])
                
                if shots_list:
                    ax4.plot(shots_list, times_list, 'o-', label=circuit_name, linewidth=2)
            
            ax4.set_xlabel('Number of Shots')
            ax4.set_ylabel('Execution Time (s)')
            ax4.set_title('Performance Scaling')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Shots per second
            ax5 = plt.subplot(2, 3, 5)
            
            for circuit_name, circuit_results in benchmark_results.items():
                shots_list = []
                sps_list = []
                
                for shots, results in circuit_results.items():
                    if results.get('success', False):
                        shots_list.append(shots)
                        sps_list.append(results.get('shots_per_second', 0))
                
                if shots_list:
                    ax5.plot(shots_list, sps_list, 's-', label=circuit_name, linewidth=2)
            
            ax5.set_xlabel('Number of Shots')
            ax5.set_ylabel('Shots per Second')
            ax5.set_title('Throughput Analysis')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Summary and insights
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "IBM Quantum Platform Summary:\n\n"
        
        if comparison_results:
            fastest_backend = min(comparison_results.items(), 
                                key=lambda x: x[1].get('execution_time', float('inf')))
            summary_text += f"Fastest Backend: {fastest_backend[0]}\n"
            summary_text += f"Execution Time: {fastest_backend[1].get('execution_time', 0):.2f}s\n\n"
        
        successful_backends = [name for name, results in comparison_results.items() 
                             if results.get('success_rate', False)]
        summary_text += f"Successful Backends: {len(successful_backends)}\n"
        summary_text += f"Total Tested: {len(comparison_results)}\n\n"
        
        summary_text += "Key Features:\n"
        summary_text += "• Real hardware access\n"
        summary_text += "• Multiple backend options\n"
        summary_text += "• Job queuing system\n"
        summary_text += "• Runtime optimization\n\n"
        
        summary_text += "Best Practices:\n"
        summary_text += "• Use simulators for development\n"
        summary_text += "• Optimize circuits for hardware\n"
        summary_text += "• Monitor queue times\n"
        summary_text += "• Implement error mitigation"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="IBM Quantum Platform Access")
    parser.add_argument('--token', type=str, help='IBM Quantum token')
    parser.add_argument('--instance', type=str, help='IBM Quantum instance')
    parser.add_argument('--backend', type=str, default='aer_simulator', 
                       help='Backend to use')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots')
    parser.add_argument('--compare-backends', action='store_true',
                       help='Compare multiple backends')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--list-backends', action='store_true',
                       help='List available backends')
    parser.add_argument('--show-visualization', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    print("Quantum Computing 101 - Module 7: Quantum Hardware and Cloud Platforms")
    print("Example 1: IBM Quantum Platform Access")
    print("=" * 56)
    
    try:
        # Check for token in environment if not provided
        if not args.token:
            args.token = os.getenv('QISKIT_IBM_TOKEN')
            
        # Initialize IBM Quantum manager
        manager = IBMQuantumManager(
            token=args.token,
            instance=args.instance,
            verbose=args.verbose
        )
        
        # Authenticate
        auth_success = manager.authenticate()
        
        if auth_success:
            print("✅ IBM Quantum authentication successful")
        else:
            print("⚠️  Using simulation mode (no IBM Quantum access)")
        
        # List available backends
        if args.list_backends:
            print("\n📋 Available Backends:")
            
            simulators = manager.get_available_backends(simulator=True, real_hardware=False)
            hardware = manager.get_available_backends(simulator=False, real_hardware=True)
            
            if simulators:
                print("\n   🔬 Simulators:")
                for backend in simulators:
                    status_icon = "🟢" if backend['status'] == 'operational' else "🔴"
                    print(f"     {status_icon} {backend['name']} ({backend['n_qubits']} qubits)")
                    if backend['pending_jobs'] > 0:
                        print(f"        Queue: {backend['pending_jobs']} jobs")
            
            if hardware:
                print("\n   🖥️  Real Hardware:")
                for backend in hardware:
                    status_icon = "🟢" if backend['status'] == 'operational' else "🔴"
                    qv_info = f", QV: {backend['quantum_volume']}" if backend['quantum_volume'] else ""
                    print(f"     {status_icon} {backend['name']} ({backend['n_qubits']} qubits{qv_info})")
                    if backend['pending_jobs'] > 0:
                        print(f"        Queue: {backend['pending_jobs']} jobs")
        
        # Get backend properties
        print(f"\n🔍 Backend Properties: {args.backend}")
        properties = manager.get_backend_properties(args.backend)
        
        for key, value in properties.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"   {key}: {len(value)} items")
            elif key == 'coupling_map' and value:
                print(f"   {key}: {len(value)} connections")
            else:
                print(f"   {key}: {value}")
        
        # Initialize analyzer
        analyzer = IBMQuantumAnalyzer(manager, verbose=args.verbose)
        
        # Compare backends
        comparison_results = None
        if args.compare_backends:
            print(f"\n🔄 Comparing backends...")
            
            # Select backends to compare
            backends_to_compare = ['aer_simulator']
            
            available = manager.get_available_backends(simulator=True, real_hardware=False)
            if len(available) > 1:
                backends_to_compare.extend([b['name'] for b in available[:2] if b['name'] != 'aer_simulator'])
            
            comparison_results = analyzer.compare_backends(backends_to_compare)
            
            print(f"\n📊 Backend Comparison Results:")
            for backend_name, results in comparison_results.items():
                print(f"\n   {backend_name}:")
                if results.get('success_rate', False):
                    print(f"     ✅ Execution time: {results['execution_time']:.2f}s")
                    print(f"     📏 Circuit depth: {results['circuit_depth']}")
                    print(f"     🔢 Total shots: {results['total_shots']}")
                    
                    if 'properties' in results:
                        props = results['properties']
                        print(f"     💾 Qubits: {props.get('n_qubits', 'Unknown')}")
                        print(f"     🎯 Simulator: {props.get('simulator', 'Unknown')}")
                else:
                    print(f"     ❌ Failed: {results.get('error', 'Unknown error')}")
        
        # Performance benchmark
        benchmark_results = None
        if args.benchmark:
            print(f"\n🏁 Running performance benchmark on {args.backend}...")
            
            benchmark_results = analyzer.benchmark_performance(
                args.backend, 
                shots_list=[100, 500, 1000]
            )
            
            print(f"\n📈 Benchmark Results:")
            for circuit_name, circuit_results in benchmark_results.items():
                print(f"\n   {circuit_name}:")
                for shots, results in circuit_results.items():
                    if results.get('success', False):
                        print(f"     {shots} shots: {results['execution_time']:.2f}s "
                              f"({results['shots_per_second']:.1f} shots/s)")
                        print(f"       Entropy: {results['entropy']:.3f}")
                    else:
                        print(f"     {shots} shots: Failed")
        
        # Single job execution
        if not args.compare_backends and not args.benchmark:
            print(f"\n🚀 Executing test circuit on {args.backend}...")
            
            test_circuit = analyzer.create_test_circuit()
            job_result = manager.submit_job(test_circuit, args.backend, shots=args.shots)
            
            if job_result['success']:
                print(f"✅ Job completed successfully")
                print(f"   Job ID: {job_result['job_id']}")
                print(f"   Backend: {job_result['backend']}")
                print(f"   Shots: {job_result['shots']}")
                
                counts = job_result['result'].get_counts()
                print(f"\n📊 Measurement Results:")
                for state, count in sorted(counts.items()):
                    probability = count / args.shots
                    print(f"   |{state}⟩: {count} ({probability:.3f})")
                    
            else:
                print(f"❌ Job failed: {job_result.get('error', 'Unknown error')}")
        
        # Visualization
        if args.show_visualization and (comparison_results or benchmark_results):
            analyzer.visualize_results(comparison_results or {}, benchmark_results)
        
        print(f"\n📚 Key Insights:")
        print(f"   • IBM Quantum provides access to real quantum hardware")
        print(f"   • Different backends have varying capabilities and performance")
        print(f"   • Simulators are ideal for development and testing")
        print(f"   • Real hardware requires queue management and error mitigation")
        
        print(f"\n🎯 Best Practices:")
        print(f"   • Start development with simulators")
        print(f"   • Optimize circuits before running on hardware")
        print(f"   • Monitor backend status and queue times")
        print(f"   • Use appropriate shot counts for statistical significance")
        
        print(f"\n✅ IBM Quantum platform access demonstration completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
