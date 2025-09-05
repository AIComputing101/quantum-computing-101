# Module 7: Quantum Hardware & Cloud Platforms
*Advanced Tier*

## 7.0 Overview
This module bridges theory and practice by exploring real quantum hardware technologies and the cloud platforms that provide access to them. You will learn architectural principles of leading qubit modalities, hardware performance metrics, access workflows across major cloud providers, execution optimization strategies, and how to responsibly run and benchmark workloads on actual quantum devices.

### Learning Objectives
By the end, you will be able to:
- Distinguish major quantum hardware modalities (superconducting, trapped ions, photonic, neutral atoms, annealing)
- Interpret hardware performance metrics (T1, T2, gate fidelity, SPAM errors, coherence volume)
- Select an appropriate backend for a given algorithmic workload
- Submit, monitor, and retrieve jobs on IBM Quantum, AWS Braket, and Azure Quantum
- Optimize circuits for specific hardware topologies (qubit mapping, transpilation levels, pulse-aware choices)
- Apply error-aware execution strategies (dynamical decoupling, readout mitigation, pulse alignment)
- Benchmark algorithm performance across simulators and hardware
- Estimate cost and queue time impacts; plan batch execution
- Design and run a hardware experiment with reproducible methodology

### Prerequisites
- Module 1–6 completion
- Familiarity with Qiskit, basic Cirq usage, variational circuits, and noise modeling
- Comfortable with Python tooling and asynchronous job handling

---
## 7.1 Hardware Landscape Overview

| Modality | Physical System | Typical Strengths | Trade-offs | Representative Providers |
|----------|-----------------|-------------------|-----------|--------------------------|
| Superconducting | Josephson junction circuits | Fast gates, ecosystem maturity | Short coherence, crosstalk | IBM, Google, Rigetti |
| Trapped Ions | Laser-cooled ions in EM traps | Long coherence, high fidelity | Slower gates, scaling traps | IonQ, Honeywell (Quantinuum) |
| Photonic | Single photons / interferometers | Room temp, low decoherence | Probabilistic gates, sources challenging | Xanadu, PsiQuantum |
| Neutral Atoms | Rydberg atom arrays | Flexible geometry, mid-circuit ops emerging | Laser control complexity | QuEra, Pasqal |
| Quantum Annealing | Flux qubits in annealer topology | Optimization at scale (10k+ qubits) | Limited to Ising/QUBO forms | D-Wave |

### Key Architectural Concepts
- Qubit Connectivity Graph
- Gate Set (native vs synthesized)
- Coherence Times (T1 energy relaxation, T2 dephasing)
- Readout Mechanisms & SPAM Errors
- Calibration Cycles & Drift
- Native Pulse Controls vs Compiled Gates

### Visualizing Connectivity (Example: Mock Lattice)
```python
from qiskit.providers.fake_provider import FakeManila
backend = FakeManila()
print("Coupling map:", backend.configuration().coupling_map)

# Build adjacency graph
import networkx as nx, matplotlib.pyplot as plt
G = nx.Graph()
G.add_nodes_from(range(backend.configuration().num_qubits))
G.add_edges_from(backend.configuration().coupling_map)
plt.figure(figsize=(4,4))
pos = nx.spring_layout(G, seed=2)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.title('Mock Backend Connectivity (FakeManila)')
plt.show()
```

#### Practical: Connectivity → Shortest Path Mapping
```python
# Map a logical CNOT between distant qubits onto physical path
logical_control, logical_target = 0, 4
cmap = backend.configuration().coupling_map

import networkx as nx
g = nx.Graph()
g.add_edges_from([tuple(e) for e in cmap])
path = nx.shortest_path(g, logical_control, logical_target)
print("Physical path for logical CNOT 0→4:", path)

# Insert SWAPs along internal nodes (conceptual example)
from qiskit import QuantumCircuit
qc = QuantumCircuit(backend.configuration().num_qubits)
for i in range(len(path)-2):
    qc.swap(path[i+1], path[i+2])
qc.cx(logical_control, logical_target)
print(qc)
```

---
### 7.1.1 Quick Modality Comparison Helper
```python
modalities = {
    'superconducting': {
        'gate_time_ns': 200, 't1_us': 100, 't2_us': 120,
        'two_q_error': 0.015, 'notes': 'Fast, shorter coherence'
    },
    'trapped_ions': {
        'gate_time_ns': 10000, 't1_us': 2_000_000, 't2_us': 1_000_000,
        'two_q_error': 0.005, 'notes': 'Slow gates, high fidelity'
    },
    'neutral_atoms': {
        'gate_time_ns': 800, 't1_us': 300, 't2_us': 400,
        'two_q_error': 0.02, 'notes': 'Scalable geometry'
    }
}

def rough_depth_budget(mod):
    m = modalities[mod]
    # Very crude: allowable depth ≈ (min(T1,T2) * 1000) / gate_time_ns * (1 - two_q_error*10)
    import math
    base = (min(m['t1_us'], m['t2_us']) * 1000) / m['gate_time_ns']
    penalty = (1 - 10*m['two_q_error'])
    return math.floor(base * penalty)

for mod in modalities:
    print(f"{mod:16}: depth budget ~{rough_depth_budget(mod)} (heuristic) -> {modalities[mod]['notes']}")
```

---
## 7.2 Core Performance Metrics & Interpretation
| Metric | Meaning | Typical Range (Current Gen) | Impact |
|--------|---------|-----------------------------|--------|
| T1 | Energy relaxation time | 50–500 μs (SC), 1–10 s (ions) | Circuit depth survivability |
| T2 | Dephasing time | 50–200 μs (SC), 1–3 s (ions) | Phase-sensitive algorithms |
| Gate Fidelity (1Q) | Error per single-qubit gate | 99.5–99.99% | Accumulated error scaling |
| Gate Fidelity (2Q) | Error per entangling gate | 97–99.5% | Affects entanglement layers |
| Readout Error | Probability of misclassification | 1–5% SC, <1% ions | Final measurement accuracy |
| SPAM Error | State prep & measurement combined | 1–5% | Baseline noise floor |
| Crosstalk | Unintended qubit coupling | Backend-specific | Layout-dependent errors |
| QV / CLOPS | Quantum Volume / circuits per second | QV 16–1024 | Throughput & quality mix |
| Coherence Volume | (T coherence × gate speed × fidelity) heuristic | Relative | Resource planning |

### Metric Retrieval (IBM Qiskit Example)
```python
from qiskit import IBMQ
# IBMQ.load_account()  # Uncomment after storing account
# provider = IBMQ.get_provider(hub='ibm-q')
# backend = provider.get_backend('ibmq_jakarta')
# props = backend.properties()
# for qubit, data in enumerate(props.qubits):
#     t1 = [p for p in data if p.name == 'T1'][0].value
#     t2 = [p for p in data if p.name == 'T2'][0].value
#     print(f"Qubit {qubit}: T1={t1:.1f} μs, T2={t2:.1f} μs")
```

### Reading Calibration Drift
- Capture timestamp of properties snapshot
- Re-query before long experiment batches
- Track moving average of error rates → adapt transpilation strategies

### Practical Heuristics
- If 2Q error > ~2%, aggressively minimize CX count
- Prefer qubits with balanced high T1 & T2, not just one peak
- Re-run layout if queue time was long (calibrations may shift)

#### 7.2.1 Aggregating Best Qubit Subset
```python
# Select k "best" qubits by scoring multiple properties
def score_qubit(qubit_props):
    vals = {p.name: p.value for p in qubit_props}
    t1 = vals.get('T1', 0.0)
    t2 = vals.get('T2', 0.0)
    anh = vals.get('anharmonicity', 0.0)
    # Weighted score (normalize rough ranges)
    return (t1/100) + (t2/100) + 0.1*anh

# props = backend.properties()
# scores = [(i, score_qubit(q)) for i, q in enumerate(props.qubits)]
# best = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
# print("Top 5 qubits:", best)
```

#### 7.2.2 Backend Scoring Function
```python
def backend_score(backend, weight_latency=0.2, weight_error=0.5, weight_queue=0.3):
    status = backend.status()
    props = backend.properties()
    avg_2q = []
    for gate in props.gates:
        if gate.gate == 'cx':
            err = next(p.value for p in gate.parameters if p.name == 'gate_error')
            avg_2q.append(err)
    avg_2q_err = sum(avg_2q)/len(avg_2q) if avg_2q else 0.05
    latency = 1.0  # placeholder static
    queue_penalty = status.pending_jobs
    # Normalize approx (heuristic)
    norm_err = min(avg_2q_err/0.05, 1.0)
    norm_queue = min(queue_penalty/50, 1.0)
    score = (1-weight_error)* (1-norm_err) + (1-weight_queue)*(1-norm_queue) - weight_latency*latency*0.01
    return score, {'2q_err': avg_2q_err, 'queue': queue_penalty}

# for b in provider.backends():
#     s, meta = backend_score(b)
#     print(f"{b.name():15} score={s:.3f} meta={meta}")
```

---
## 7.3 Accessing Cloud Platforms
### 7.3.1 IBM Quantum (Qiskit) - Complete Workflow
```python
from qiskit import IBMQ, QuantumCircuit, transpile, execute
from qiskit.circuit.random import random_circuit
import time

# Account setup (one-time)
# IBMQ.save_account('YOUR_TOKEN')  # Replace with actual token

def ibm_complete_workflow():
    """Complete IBM Quantum workflow with error handling"""
    
    # Load account and get provider
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    
    # Get available backends
    backends = provider.backends(simulator=False, operational=True)
    print("Available backends:")
    for backend in backends:
        status = backend.status()
        config = backend.configuration()
        print(f"  {backend.name()}: {config.n_qubits} qubits, queue: {status.pending_jobs}")
    
    # Select backend intelligently
    def select_best_backend(backends, min_qubits=5):
        """Select backend with good balance of capabilities and queue"""
        candidates = [b for b in backends if b.configuration().n_qubits >= min_qubits]
        
        if not candidates:
            return backends[0]  # fallback
        
        # Score by queue length (lower is better)
        scored = [(b, b.status().pending_jobs) for b in candidates]
        scored.sort(key=lambda x: x[1])
        return scored[0][0]
    
    backend = select_best_backend(backends)
    print(f"Selected backend: {backend.name()}")
    
    # Create test circuit
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    # Analyze pre-transpilation
    print(f"Original circuit: {qc.depth()} depth, {qc.count_ops()} gates")
    
    # Transpile with optimization
    transpiled_qc = transpile(qc, backend, optimization_level=3)
    print(f"Transpiled circuit: {transpiled_qc.depth()} depth, {transpiled_qc.count_ops()} gates")
    
    # Submit job with error handling
    try:
        job = backend.run(transpiled_qc, shots=1024)
        print(f"Job submitted: {job.job_id()}")
        
        # Monitor job progress
        while job.status().name not in ['DONE', 'CANCELLED', 'ERROR']:
            print(f"Status: {job.status().name}")
            time.sleep(30)  # Poll every 30 seconds
        
        if job.status().name == 'DONE':
            result = job.result()
            counts = result.get_counts()
            print(f"Results: {counts}")
            
            # Extract execution metadata
            metadata = result.results[0].header.metadata
            print(f"Execution time: {result.time_taken:.2f}s")
            
        else:
            print(f"Job failed with status: {job.status()}")
            
    except Exception as e:
        print(f"Job submission failed: {e}")
    
    return backend, job

# Run the workflow (uncomment to execute)
# backend, job = ibm_complete_workflow()
```

#### Advanced Backend Analysis
```python
def analyze_backend_properties(backend):
    """Comprehensive backend analysis for decision making"""
    
    properties = backend.properties()
    configuration = backend.configuration()
    
    analysis = {
        'name': backend.name(),
        'qubits': configuration.n_qubits,
        'basis_gates': configuration.basis_gates,
        'coupling_map': configuration.coupling_map,
        'quantum_volume': getattr(configuration, 'quantum_volume', None)
    }
    
    # Analyze qubit quality
    if properties:
        qubit_metrics = []
        for i, qubit in enumerate(properties.qubits):
            metrics = {}
            for prop in qubit:
                metrics[prop.name] = prop.value
            
            qubit_metrics.append({
                'qubit': i,
                'T1': metrics.get('T1', 0),
                'T2': metrics.get('T2', 0),
                'readout_error': metrics.get('readout_error', 0),
                'frequency': metrics.get('frequency', 0)
            })
            
        analysis['qubit_metrics'] = qubit_metrics
        
        # Find best qubits
        best_qubits = sorted(qubit_metrics, 
                           key=lambda x: x['T1'] * x['T2'] / (1 + x['readout_error']), 
                           reverse=True)[:5]
        analysis['best_qubits'] = [q['qubit'] for q in best_qubits]
        
        # Analyze gate errors
        gate_errors = {}
        for gate in properties.gates:
            gate_key = f"{gate.gate}_{gate.qubits}"
            for param in gate.parameters:
                if param.name == 'gate_error':
                    gate_errors[gate_key] = param.value
        
        analysis['gate_errors'] = gate_errors
        
        # Calculate average 2-qubit error
        cx_errors = [err for key, err in gate_errors.items() if 'cx_' in key]
        analysis['avg_cx_error'] = sum(cx_errors) / len(cx_errors) if cx_errors else None
    
    return analysis

# Example usage
# analysis = analyze_backend_properties(backend)
# print(f"Backend analysis for {analysis['name']}:\n  Best qubits: {analysis['best_qubits']}\n  Average CX error: {analysis['avg_cx_error']:.4f}")
```


#### Async Job Polling Utility
```python
import time
def wait_for_job(job, poll=5, timeout=1800):
    start = time.time()
    while True:
        status = job.status()
        if status.name in ('DONE','CANCELLED','ERROR'): break
        if time.time() - start > timeout:
            raise TimeoutError("Job timeout")
        print(f"Status={status.name} elapsed={time.time()-start:.1f}s")
        time.sleep(poll)
    return job.result()
```

### 7.3.2 AWS Braket - Complete Implementation
```python
from braket.aws import AwsDevice, AwsQuantumTask
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import boto3
import time

def braket_complete_workflow():
    """Complete AWS Braket workflow with device selection"""
    
    # List available devices
    def list_braket_devices():
        """Get available Braket devices with status"""
        devices = AwsDevice.get_devices()
        device_info = []
        
        for device in devices:
            info = {
                'name': device.name,
                'arn': device.arn,
                'type': device.type,
                'status': device.status,
                'provider': device.provider_name
            }
            
            # Get device properties if available
            try:
                properties = device.properties
                if hasattr(properties, 'braketSchemaHeader'):
                    if 'GateModelQpuDeviceProperties' in str(properties.braketSchemaHeader):
                        info['qubit_count'] = getattr(properties.paradigm, 'qubitCount', 'N/A')
                        info['connectivity'] = getattr(properties.paradigm, 'connectivity', 'N/A')
            except:
                pass
            
            device_info.append(info)
        
        return device_info
    
    devices = list_braket_devices()
    print("Available Braket devices:")
    for device in devices:
        print(f"  {device['name']} ({device['provider']}): {device['status']}")
        if 'qubit_count' in device:
            print(f"    Qubits: {device['qubit_count']}")
    
    # Create test circuit
    def create_bell_circuit():
        """Create a Bell state circuit in Braket format"""
        circuit = Circuit()
        circuit.h(0)
        circuit.cnot(0, 1)
        # Add explicit measurement instructions for hardware
        circuit.measure(0)
        circuit.measure(1)
        return circuit
    
    bell_circuit = create_bell_circuit()
    print(f"\nCreated circuit with {bell_circuit.depth} depth")
    
    # Run on local simulator first (validation)
    local_sim = LocalSimulator()
    local_task = local_sim.run(bell_circuit, shots=1000)
    local_result = local_task.result()
    print(f"Local simulation results: {local_result.measurement_counts}")
    
    # Select cloud device (simulator for demo)
    sv1_device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    
    try:
        # Submit to cloud simulator
        cloud_task = sv1_device.run(bell_circuit, shots=1000)
        print(f"Cloud task submitted: {cloud_task.id}")
        
        # Wait for completion
        while cloud_task.state() not in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print(f"Task state: {cloud_task.state()}")
            time.sleep(5)
        
        if cloud_task.state() == 'COMPLETED':
            cloud_result = cloud_task.result()
            print(f"Cloud results: {cloud_result.measurement_counts}")
            
            # Compare with local simulation
            local_counts = local_result.measurement_counts
            cloud_counts = cloud_result.measurement_counts
            
            print("\nComparison (local vs cloud):")
            for outcome in ['00', '01', '10', '11']:
                local_prob = local_counts.get(outcome, 0) / 1000
                cloud_prob = cloud_counts.get(outcome, 0) / 1000
                print(f"  {outcome}: {local_prob:.3f} vs {cloud_prob:.3f}")
        
        else:
            print(f"Task failed: {cloud_task.state()}")
    
    except Exception as e:
        print(f"Braket execution failed: {e}")
    
    return bell_circuit, cloud_task

# Example cost estimation for Braket
def estimate_braket_cost(device_arn, shots, estimated_runtime_seconds):
    """Estimate cost for Braket execution"""
    
    # Cost per second for different device types (as of 2025, approximate)
    device_costs = {
        'ionq': 0.00035,           # IonQ devices
        'rigetti': 0.00025,       # Rigetti devices  
        'sv1': 0.000075,          # State vector simulator
        'tn1': 0.000275,          # Tensor network simulator
        'dm1': 0.000375,          # Density matrix simulator
    }
    
    # Extract device type from ARN
    device_type = 'unknown'
    if 'ionq' in device_arn.lower():
        device_type = 'ionq'
    elif 'rigetti' in device_arn.lower():
        device_type = 'rigetti'
    elif 'sv1' in device_arn:
        device_type = 'sv1'
    elif 'tn1' in device_arn:
        device_type = 'tn1'
    elif 'dm1' in device_arn:
        device_type = 'dm1'
    
    cost_per_second = device_costs.get(device_type, 0.001)  # Default fallback
    total_cost = estimated_runtime_seconds * cost_per_second
    
    return {
        'device_type': device_type,
        'cost_per_second': cost_per_second,
        'estimated_runtime': estimated_runtime_seconds,
        'total_cost': total_cost,
        'shots': shots
    }

# Example usage
# circuit, task = braket_complete_workflow()
cost_estimate = estimate_braket_cost('arn:aws:braket:::device/quantum-simulator/amazon/sv1', 
                                   1000, 3.0)
print(f"\nCost estimate: ${cost_estimate['total_cost']:.4f} for {cost_estimate['shots']} shots")
```

### 7.3.3 Azure Quantum - Complete Implementation
```python
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit import QuantumCircuit, transpile
import time

def azure_quantum_workflow():
    """Complete Azure Quantum workflow with provider management"""
    
    # Workspace configuration (replace with your details)
    workspace_config = {
        'subscription_id': 'your-subscription-id',
        'resource_group': 'your-resource-group',
        'name': 'your-workspace-name',
        'location': 'your-location'  # e.g., 'East US'
    }
    
    try:
        # Initialize workspace
        workspace = Workspace(**workspace_config)
        print(f"Connected to workspace: {workspace.name}")
        
        # Get available providers and backends
        def list_azure_backends():
            """List all available backends in Azure Quantum workspace"""
            backends_info = []
            
            # Common providers in Azure Quantum
            providers = ['ionq', 'quantinuum', 'rigetti']
            
            for provider_name in providers:
                try:
                    provider = workspace.get_provider(provider_name)
                    backends = provider.backends()
                    
                    for backend in backends:
                        info = {
                            'provider': provider_name,
                            'name': backend.name(),
                            'status': backend.status().operational,
                            'max_shots': getattr(backend.configuration(), 'max_shots', 'N/A'),
                            'max_experiments': getattr(backend.configuration(), 'max_experiments', 'N/A')
                        }
                        
                        # Get backend-specific properties
                        config = backend.configuration()
                        if hasattr(config, 'n_qubits'):
                            info['qubits'] = config.n_qubits
                        if hasattr(config, 'basis_gates'):
                            info['basis_gates'] = config.basis_gates
                        
                        backends_info.append(info)
                        
                except Exception as e:
                    print(f"Could not access {provider_name}: {e}")
            
            return backends_info
        
        backends = list_azure_backends()
        print("\nAvailable backends:")
        for backend in backends:
            status = "✓" if backend['status'] else "✗"
            print(f"  {status} {backend['provider']}.{backend['name']}: {backend.get('qubits', 'N/A')} qubits")
        
        # Create and prepare circuit
        def create_grover_circuit():
            """Create a simple 2-qubit Grover circuit"""
            qc = QuantumCircuit(2, 2)
            
            # Initialize superposition
            qc.h([0, 1])
            
            # Oracle: mark |11⟩ state
            qc.cz(0, 1)
            
            # Diffusion operator
            qc.h([0, 1])
            qc.x([0, 1])
            qc.cz(0, 1)
            qc.x([0, 1])
            qc.h([0, 1])
            
            qc.measure_all()
            return qc
        
        circuit = create_grover_circuit()
        print(f"\nCreated Grover circuit: {circuit.depth()} depth, {circuit.width()} qubits")
        
        # Select backend (use simulator for demo)
        if backends:
            # Prefer IonQ simulator if available
            ionq_backends = [b for b in backends if b['provider'] == 'ionq' and 'simulator' in b['name']]
            if ionq_backends:
                target_backend = ionq_backends[0]
                provider = workspace.get_provider(target_backend['provider'])
                backend = provider.get_backend(target_backend['name'])
                
                print(f"Selected backend: {target_backend['provider']}.{target_backend['name']}")
                
                # Transpile circuit
                transpiled = transpile(circuit, backend)
                print(f"Transpiled circuit: {transpiled.depth()} depth")
                
                # Submit job (uncomment to run)
                # job = backend.run(transpiled, shots=1024)
                # print(f"Job submitted: {job.job_id()}")
                
                # Monitor job progress
                # while job.status().name not in ['DONE', 'CANCELLED', 'ERROR']:
                #     print(f"Status: {job.status().name}")
                #     time.sleep(10)
                
                # if job.status().name == 'DONE':
                #     result = job.result()
                #     counts = result.get_counts()
                #     print(f"Results: {counts}")
                
        else:
            print("No backends available")
    
    except Exception as e:
        print(f"Azure Quantum workflow failed: {e}")
        return None
    
    return workspace

# Azure Quantum cost estimation
def estimate_azure_cost(provider, backend_name, shots):
    """Estimate cost for Azure Quantum execution"""
    
    # Pricing as of 2025 (approximate, varies by region/plan)
    pricing_table = {
        'ionq': {
            'simulator': {'per_shot': 0.0},           # Often free tier
            'qpu': {'per_shot': 0.00003}              # IonQ QPU pricing
        },
        'quantinuum': {
            'simulator': {'per_shot': 0.0},           # H-Series simulator
            'qpu': {'per_shot': 0.00005}              # H-Series QPU
        },
        'rigetti': {
            'simulator': {'per_shot': 0.0},           # Rigetti simulator
            'qpu': {'per_shot': 0.00001}              # Rigetti QPU
        }
    }
    
    device_type = 'simulator' if 'simulator' in backend_name.lower() else 'qpu'
    
    if provider in pricing_table and device_type in pricing_table[provider]:
        cost_per_shot = pricing_table[provider][device_type]['per_shot']
        total_cost = shots * cost_per_shot
        
        return {
            'provider': provider,
            'backend': backend_name,
            'device_type': device_type,
            'shots': shots,
            'cost_per_shot': cost_per_shot,
            'total_cost': total_cost
        }
    else:
        return {'error': f'Pricing not available for {provider}.{backend_name}'}

# Example usage
# workspace = azure_quantum_workflow()
cost_example = estimate_azure_cost('ionq', 'ionq.simulator', 1024)
print(f"\nAzure cost estimate: ${cost_example['total_cost']:.4f} for {cost_example['shots']} shots")
```

### 7.3.4 Platform Selection Matrix
| Need | Prefer | Rationale |
|------|--------|-----------|
| Fast iteration on SC qubits | IBM | Rich transpiler, mature docs |
| Long coherence experiments | IonQ / Quantinuum | Ion trap stability |
| Hybrid classical orchestration | Braket | Managed hybrid jobs |
| Photonic ML workflows | Xanadu (via API) | Native photonic toolchain |
| Large QUBO optimization | D-Wave | Annealing scale |

---
## 7.4 Circuit Optimization & Hardware-Aware Transpilation
### Objectives
- Reduce depth & 2Q gate count
- Align with backend native gate set
- Exploit dynamical decoupling (DD) & pulse optimizations

### Baseline vs Optimized Example
```python
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from qiskit import transpile

qc = QuantumCircuit(5)
qc.h(range(5))
for i in range(4):
    qc.cx(i, i+1)
qc.barrier()
qc.cx(0,2)
qc.cx(1,3)
qc.measure_all()

# backend = provider.get_backend('ibmq_jakarta')
# default = transpile(qc, backend, optimization_level=0)
# optimized = transpile(qc, backend, optimization_level=3)
# print("CX count (L0 vs L3):", default.count_ops().get('cx',0), optimized.count_ops().get('cx',0))
```

### Advanced: Smart Backend Selection Algorithm
```python
import numpy as np
from typing import List, Dict, Tuple

def intelligent_backend_selection(backends, circuit, requirements=None):
    """Intelligent backend selection based on circuit requirements and hardware metrics"""
    
    if requirements is None:
        requirements = {
            'min_qubits': circuit.num_qubits,
            'max_queue_time': 3600,  # 1 hour max
            'max_error_rate': 0.1,   # 10% max error
            'prefer_fidelity': True,
            'cost_weight': 0.3,
            'speed_weight': 0.4,
            'fidelity_weight': 0.3
        }
    
    def score_backend(backend_info):
        """Score a backend based on multiple criteria"""
        score = 0.0
        penalties = []
        
        # Hard constraints
        if backend_info.get('qubits', 0) < requirements['min_qubits']:
            return -1, ['Insufficient qubits']
        
        if backend_info.get('queue_time', 0) > requirements['max_queue_time']:
            penalties.append('Queue too long')
            score -= 0.3
        
        # Fidelity scoring
        error_rate = backend_info.get('avg_error_rate', 0.05)
        if error_rate < requirements['max_error_rate']:
            fidelity_score = (1 - error_rate) * requirements['fidelity_weight']
            score += fidelity_score
        else:
            return -1, ['Error rate too high']
        
        # Speed scoring (inverse of queue time)
        queue_time = backend_info.get('queue_time', 1800)
        speed_score = (3600 - min(queue_time, 3600)) / 3600 * requirements['speed_weight']
        score += speed_score
        
        # Cost scoring (inverse of estimated cost)
        estimated_cost = backend_info.get('estimated_cost', 1.0)
        cost_score = (2.0 - min(estimated_cost, 2.0)) / 2.0 * requirements['cost_weight']
        score += cost_score
        
        # Bonus for specific features
        if backend_info.get('supports_mid_circuit_measurement', False):
            score += 0.1
        
        if backend_info.get('connectivity_score', 0.5) > 0.8:  # High connectivity
            score += 0.05
        
        return score, penalties
    
    # Score all backends
    scored_backends = []
    for backend in backends:
        score, penalties = score_backend(backend)
        if score >= 0:  # Only include viable backends
            scored_backends.append({
                'backend': backend,
                'score': score,
                'penalties': penalties
            })
    
    # Sort by score (highest first)
    scored_backends.sort(key=lambda x: x['score'], reverse=True)
    
    return scored_backends

# Example backend data structure
example_backends = [
    {
        'name': 'ibmq_jakarta',
        'provider': 'ibm',
        'qubits': 7,
        'queue_time': 1200,  # 20 minutes
        'avg_error_rate': 0.02,
        'estimated_cost': 0.0,  # Free tier
        'supports_mid_circuit_measurement': False,
        'connectivity_score': 0.6
    },
    {
        'name': 'ionq_qpu',
        'provider': 'aws_braket',
        'qubits': 32,
        'queue_time': 300,   # 5 minutes
        'avg_error_rate': 0.005,
        'estimated_cost': 0.5,
        'supports_mid_circuit_measurement': True,
        'connectivity_score': 1.0  # All-to-all connectivity
    },
    {
        'name': 'rigetti_aspen',
        'provider': 'aws_braket',
        'qubits': 80,
        'queue_time': 2400,  # 40 minutes
        'avg_error_rate': 0.03,
        'estimated_cost': 0.2,
        'supports_mid_circuit_measurement': False,
        'connectivity_score': 0.4
    }
]

# Create a dummy circuit for testing
test_circuit = QuantumCircuit(5)
test_circuit.h(range(5))
for i in range(4):
    test_circuit.cx(i, i+1)
test_circuit.measure_all()

# Run backend selection
ranked_backends = intelligent_backend_selection(example_backends, test_circuit)

print("Backend Selection Results:")
print("=" * 50)
for i, result in enumerate(ranked_backends):
    backend = result['backend']
    print(f"{i+1}. {backend['name']} ({backend['provider']})")
    print(f"   Score: {result['score']:.3f}")
    print(f"   Qubits: {backend['qubits']}, Error: {backend['avg_error_rate']:.3%}")
    print(f"   Queue: {backend['queue_time']/60:.1f}min, Cost: ${backend['estimated_cost']:.2f}")
    if result['penalties']:
        print(f"   Penalties: {', '.join(result['penalties'])}")
    print()
```

#### Advanced Transpilation with Custom Optimization
```python
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import (
    SabreLayout, SabreSwap, Optimize1qGates, CommutativeCancellation,
    Unroller, Depth, Size, RemoveBarriers, ConsolidateBlocks
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def advanced_transpilation_pipeline(circuit, backend, custom_optimization=True):
    """Advanced transpilation with custom optimization strategies"""
    
    coupling_map = CouplingMap(backend.configuration().coupling_map)
    basis_gates = backend.configuration().basis_gates
    backend_properties = backend.properties()
    
    print(f"Original circuit: {circuit.depth()} depth, {circuit.size()} gates")
    
    # Strategy 1: Default optimization levels comparison
    optimization_results = {}
    for level in range(4):
        transpiled = transpile(circuit, backend, optimization_level=level)
        optimization_results[f'level_{level}'] = {
            'depth': transpiled.depth(),
            'size': transpiled.size(),
            'cx_count': transpiled.count_ops().get('cx', 0)
        }
    
    print("\nOptimization level comparison:")
    for level, metrics in optimization_results.items():
        print(f"  {level}: depth={metrics['depth']}, size={metrics['size']}, cx={metrics['cx_count']}")
    
    # Strategy 2: Custom pass manager with noise-aware optimization
    if custom_optimization:
        # Build custom pass manager
        pass_manager = PassManager()
        
        # Initial layout using SABRE (noise-aware)
        pass_manager.append(SabreLayout(coupling_map, backend_properties))
        
        # Route with SABRE swapping
        pass_manager.append(SabreSwap(coupling_map))
        
        # Optimize single-qubit gates
        pass_manager.append(Optimize1qGates(basis=basis_gates))
        
        # Commutative cancellation
        pass_manager.append(CommutativeCancellation())
        
        # Consolidate consecutive blocks
        pass_manager.append(ConsolidateBlocks(basis_gates=basis_gates))
        
        # Final cleanup
        pass_manager.append(RemoveBarriers())
        pass_manager.append(Optimize1qGates(basis=basis_gates))
        
        custom_transpiled = pass_manager.run(circuit)
        
        custom_metrics = {
            'depth': custom_transpiled.depth(),
            'size': custom_transpiled.size(),
            'cx_count': custom_transpiled.count_ops().get('cx', 0)
        }
        
        print(f"\nCustom optimization: depth={custom_metrics['depth']}, size={custom_metrics['size']}, cx={custom_metrics['cx_count']}")
        
        return custom_transpiled, optimization_results, custom_metrics
    
    else:
        # Return best default optimization
        best_level = min(optimization_results.keys(), 
                        key=lambda k: optimization_results[k]['depth'])
        best_transpiled = transpile(circuit, backend, 
                                  optimization_level=int(best_level.split('_')[1]))
        
        return best_transpiled, optimization_results, optimization_results[best_level]

# Strategy 3: Layout-aware optimization for specific algorithms
def algorithm_aware_transpilation(circuit, backend, algorithm_type='general'):
    """Transpilation optimized for specific algorithm types"""
    
    strategies = {
        'variational': {
            'optimization_level': 3,
            'layout_method': 'sabre',  # Good for parameterized circuits
            'routing_method': 'sabre'
        },
        'qaoa': {
            'optimization_level': 2,
            'layout_method': 'noise_adaptive',  # Prefer low-error qubits
            'routing_method': 'basic'
        },
        'simulation': {
            'optimization_level': 1,  # Preserve structure
            'layout_method': 'trivial',
            'routing_method': 'lookahead'
        },
        'general': {
            'optimization_level': 3,
            'layout_method': 'sabre',
            'routing_method': 'sabre'
        }
    }
    
    strategy = strategies.get(algorithm_type, strategies['general'])
    
    # Create preset pass manager with strategy
    pass_manager = generate_preset_pass_manager(
        optimization_level=strategy['optimization_level'],
        backend=backend,
        layout_method=strategy['layout_method'],
        routing_method=strategy['routing_method']
    )
    
    optimized_circuit = pass_manager.run(circuit)
    
    return optimized_circuit, strategy

# Example usage with analysis
def transpilation_benchmark(circuit, backend):
    """Comprehensive transpilation benchmarking"""
    
    print("Transpilation Benchmark Results")
    print("=" * 40)
    
    # Test custom optimization
    custom_result, level_results, custom_metrics = advanced_transpilation_pipeline(
        circuit, backend, custom_optimization=True)
    
    # Test algorithm-specific strategies
    algorithm_types = ['variational', 'qaoa', 'simulation', 'general']
    algorithm_results = {}
    
    print("\nAlgorithm-specific optimization:")
    for alg_type in algorithm_types:
        optimized, strategy = algorithm_aware_transpilation(circuit, backend, alg_type)
        metrics = {
            'depth': optimized.depth(),
            'size': optimized.size(),
            'cx_count': optimized.count_ops().get('cx', 0),
            'strategy': strategy
        }
        algorithm_results[alg_type] = metrics
        print(f"  {alg_type}: depth={metrics['depth']}, cx={metrics['cx_count']}")
    
    # Find best overall strategy
    all_results = {**level_results, 'custom': custom_metrics, **algorithm_results}
    
    # Score by weighted combination of depth and cx count
    def transpilation_score(metrics):
        return metrics['depth'] * 0.6 + metrics['cx_count'] * 0.4
    
    best_strategy = min(all_results.keys(), 
                       key=lambda k: transpilation_score(all_results[k]))
    
    print(f"\nBest strategy: {best_strategy}")
    print(f"Metrics: {all_results[best_strategy]}")
    
    return all_results

# Example usage (uncomment to run with real backend)
# benchmark_results = transpilation_benchmark(test_circuit, backend)
```

### Layout Strategies
- SABRE heuristic for dynamic mapping
- Noise-adaptive layout (choose low-error qubits)
- Manual pinning for reproducibility

### Dynamical Decoupling (Conceptual)
Insert echo sequences (X–I–X or XY4) into idle windows to preserve coherence.

### Pulse-Level Awareness (Advanced)
- Align phase calibration windows
- Avoid back-to-back cross-resonance saturation
- Calibrate custom schedules (requires pulse backend access)

---
## 7.5 Error Mitigation & Readout Strategies on Hardware
| Technique | Purpose | When to Use |
|-----------|---------|-------------|
| Readout Mitigation | Correct measurement bias | Any mid/high readout error |
| Zero-Noise Extrapolation | Infer zero-noise observable | Variational expectation eval |
| Probabilistic Error Cancellation | High-fidelity obs (costly) | Small circuits, critical accuracy |
| Measurement Error Mitigation + Subspace Projection | Stabilizer / structured states | Post-QEC / encoded states |
| Dynamical Decoupling | Preserve idle qubits | Deep circuits w/ idle spans |

### Comprehensive Error Mitigation Pipeline
```python
from qiskit import QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
import numpy as np

class ErrorMitigationPipeline:
    """Complete error mitigation pipeline for hardware execution"""
    
    def __init__(self, backend):
        self.backend = backend
        self.calibration_data = {}
        self.mitigation_methods = []
    
    def calibrate_readout_errors(self, qubits):
        """Calibrate readout error mitigation matrix"""
        print(f"Calibrating readout errors for qubits {qubits}")
        
        # Create calibration circuits
        cal_circuits = []
        state_labels = []
        
        # Generate all computational basis states for given qubits
        n_qubits = len(qubits)
        for state in range(2**n_qubits):
            qc = QuantumCircuit(max(qubits) + 1, n_qubits)
            
            # Prepare computational basis state
            for i, qubit in enumerate(qubits):
                if (state >> i) & 1:
                    qc.x(qubit)
            
            # Measure only the specified qubits
            for i, qubit in enumerate(qubits):
                qc.measure(qubit, i)
            
            cal_circuits.append(qc)
            state_labels.append(format(state, f'0{n_qubits}b'))
        
        # Execute calibration circuits
        transpiled_cals = transpile(cal_circuits, self.backend)
        cal_job = self.backend.run(transpiled_cals, shots=8192)
        
        # Wait for results (in practice, implement proper polling)
        # cal_results = cal_job.result()
        
        # Build confusion matrix (simplified for demo)
        confusion_matrix = np.eye(2**n_qubits) * 0.95  # 95% fidelity diagonal
        
        # Add off-diagonal errors
        for i in range(2**n_qubits):
            error_prob = 0.05 / (2**n_qubits - 1)  # Distribute 5% error
            for j in range(2**n_qubits):
                if i != j:
                    confusion_matrix[i, j] = error_prob
        
        self.calibration_data['readout_matrix'] = confusion_matrix
        self.calibration_data['qubits'] = qubits
        print(f"Readout calibration completed. Average fidelity: {np.trace(confusion_matrix)/len(confusion_matrix):.3f}")
        
        return confusion_matrix
    
    def apply_readout_mitigation(self, counts):
        """Apply readout error mitigation to measurement counts"""
        if 'readout_matrix' not in self.calibration_data:
            print("Warning: No readout calibration available")
            return counts
        
        confusion_matrix = self.calibration_data['readout_matrix']
        
        # Convert counts to probability vector
        total_shots = sum(counts.values())
        n_states = len(confusion_matrix)
        prob_vector = np.zeros(n_states)
        
        for bitstring, count in counts.items():
            state_index = int(bitstring, 2)
            prob_vector[state_index] = count / total_shots
        
        # Invert confusion matrix and apply correction
        try:
            corrected_probs = np.linalg.solve(confusion_matrix.T, prob_vector)
            # Ensure probabilities are non-negative and normalized
            corrected_probs = np.maximum(corrected_probs, 0)
            corrected_probs /= np.sum(corrected_probs)
            
            # Convert back to counts
            corrected_counts = {}
            for i, prob in enumerate(corrected_probs):
                if prob > 1e-6:  # Only include non-negligible probabilities
                    bitstring = format(i, f'0{len(self.calibration_data["qubits"])}b')
                    corrected_counts[bitstring] = int(prob * total_shots)
            
            return corrected_counts
            
        except np.linalg.LinAlgError:
            print("Warning: Readout matrix inversion failed")
            return counts
    
    def zero_noise_extrapolation(self, circuit, executor_func, noise_factors=[1, 2, 3]):
        """Implement zero-noise extrapolation"""
        print("Performing zero-noise extrapolation...")
        
        def fold_circuit(qc, factor):
            """Fold circuit by repeating gate pairs"""
            if factor == 1:
                return qc
            
            folded = QuantumCircuit(qc.num_qubits, qc.num_clbits)
            
            for instruction in qc.data:
                folded.append(instruction[0], instruction[1], instruction[2])
                
                # Add folding: G -> G*G†*G for each gate
                if factor > 1 and instruction[0].name in ['cx', 'x', 'y', 'z', 'h']:
                    for _ in range(2 * (factor - 1)):
                        if instruction[0].name == 'cx':
                            folded.cx(instruction[1][0], instruction[1][1])
                        elif instruction[0].name in ['x', 'y', 'z', 'h']:
                            getattr(folded, instruction[0].name)(instruction[1][0])
            
            return folded
        
        # Execute circuits with different noise levels
        results = []
        for factor in noise_factors:
            folded_circuit = fold_circuit(circuit, factor)
            result = executor_func(folded_circuit)
            results.append((factor, result))
        
        # Extrapolate to zero noise (linear extrapolation for demo)
        x_vals = [r[0] for r in results]
        y_vals = [r[1] for r in results]
        
        # Linear fit: y = mx + b, extrapolate to x=0
        if len(x_vals) >= 2:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            zero_noise_value = coeffs[1]  # y-intercept
            print(f"Zero-noise extrapolated value: {zero_noise_value:.4f}")
            return zero_noise_value
        else:
            return y_vals[0]
    
    def run_with_mitigation(self, circuit, shots=2048, methods=['readout']):
        """Run circuit with comprehensive error mitigation"""
        
        print(f"Running circuit with mitigation methods: {methods}")
        
        # Transpile circuit
        transpiled = transpile(circuit, self.backend, optimization_level=3)
        
        # Raw execution
        raw_job = self.backend.run(transpiled, shots=shots)
        # raw_result = raw_job.result()
        # raw_counts = raw_result.get_counts()
        
        # Simulate raw counts for demo
        raw_counts = {'000': 400, '001': 50, '010': 50, '011': 24, 
                     '100': 50, '101': 24, '110': 24, '111': 400}
        
        results = {'raw': raw_counts}
        
        # Apply requested mitigation methods
        if 'readout' in methods:
            if 'readout_matrix' not in self.calibration_data:
                self.calibrate_readout_errors(list(range(circuit.num_qubits)))
            
            mitigated_counts = self.apply_readout_mitigation(raw_counts)
            results['readout_mitigated'] = mitigated_counts
        
        # Add other mitigation methods as needed
        
        return results

# Example usage and comparison
def mitigation_comparison_demo():
    """Demonstrate error mitigation effectiveness"""
    
    # Create test circuit (GHZ state)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    # Simulate with different backends
    backends = {
        'ideal': AerSimulator(method='statevector'),
        'noisy': AerSimulator(noise_model=NoiseModel())  # Placeholder noise model
    }
    
    print("Error Mitigation Comparison")
    print("=" * 40)
    
    for backend_name, backend in backends.items():
        print(f"\n{backend_name.upper()} BACKEND:")
        
        if backend_name == 'ideal':
            # Ideal execution
            job = backend.run(qc, shots=2048)
            # result = job.result()
            # counts = result.get_counts()
            counts = {'000': 1024, '111': 1024}  # Perfect GHZ
            print(f"  Ideal counts: {counts}")
            
        else:
            # Noisy execution with mitigation
            pipeline = ErrorMitigationPipeline(backend)
            results = pipeline.run_with_mitigation(qc, methods=['readout'])
            
            print(f"  Raw counts: {results['raw']}")
            print(f"  Mitigated: {results['readout_mitigated']}")
            
            # Calculate fidelity improvement
            def state_fidelity_estimate(counts1, counts2):
                """Rough fidelity estimate from counts"""
                total1 = sum(counts1.values())
                total2 = sum(counts2.values())
                
                overlap = 0
                all_states = set(counts1.keys()) | set(counts2.keys())
                for state in all_states:
                    p1 = counts1.get(state, 0) / total1
                    p2 = counts2.get(state, 0) / total2
                    overlap += np.sqrt(p1 * p2)
                
                return overlap
            
            ideal_counts = {'000': 1024, '111': 1024}
            raw_fidelity = state_fidelity_estimate(results['raw'], ideal_counts)
            mitigated_fidelity = state_fidelity_estimate(results['readout_mitigated'], ideal_counts)
            
            print(f"  Fidelity improvement: {raw_fidelity:.3f} → {mitigated_fidelity:.3f}")

# Run demonstration
mitigation_comparison_demo()
```

### Scalable Strategy
1. Batch calibration circuits early (cache matrix)
2. Tag experiments with calibration snapshot ID
3. Recalibrate after threshold drift (e.g., >20% error change)

---
## 7.6 Benchmarking & Cross-Platform Evaluation
### Goals
- Compare simulator ideal vs noisy vs hardware
- Quantify overhead of mitigation
- Track temporal drift

### Benchmark Framework Skeleton
```python
import time, statistics
from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class BenchmarkResult:
    name: str
    depth: int
    width: int
    shots: int
    exec_time: float
    raw_counts: dict
    metadata: dict

class HardwareBenchmarker:
    def __init__(self, backend):
        self.backend = backend
        self.results = []

    def run(self, circuit_factory: Callable, name: str, shots=1024):
        qc = circuit_factory()
        # tqc = transpile(qc, self.backend, optimization_level=3)
        start = time.time()
        # job = self.backend.run(tqc, shots=shots)
        # result = job.result()
        end = time.time()
        self.results.append(
            BenchmarkResult(
                name=name,
                depth=qc.depth(),
                width=qc.num_qubits,
                shots=shots,
                exec_time=end-start,
                raw_counts={}, # result.get_counts() placeholder
                metadata={'approx_queue_delay': None}
            )
        )

    def summarize(self):
        for r in self.results:
            print(f"{r.name}: depth={r.depth}, width={r.width}, time={r.exec_time:.2f}s")
```

#### Adding Fidelity & Noise Model Comparison
```python
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

def ideal_vs_noisy(qc, backend):
    sim = AerSimulator(method='statevector')
    ideal = sim.run(qc).result().get_statevector()

    # noise_model = NoiseModel.from_backend(backend)  # real backend properties
    # noisy_sim = AerSimulator(noise_model=noise_model)
    # noisy_counts = noisy_sim.run(qc, shots=4096).result().get_counts()
    # (Simplified) fidelity placeholder
    # from qiskit.quantum_info import state_fidelity
    # est_fid = state_fidelity(ideal, noisy_statevector)
    # print("Estimated fidelity:", est_fid)
```

### Cross-Platform Comparison Dimensions
| Dimension | Example Metric |
|----------|----------------|
| Fidelity | Overlap vs ideal statevector |
| Throughput | Circuits / hour |
| Cost | $ per 1000 shots |
| Queue Delay | Avg wait time |
| Stability | Variance over repeated runs |

---
## 7.7 Cost, Queue & Resource Management
### Best Practices
- Group small circuits into a single job bundle when supported
- Avoid excessive shots early—progressively refine
- Use simulator pre-validation to prune failing circuits
- Monitor queue length & dynamically switch target backend
- Track cost per successful experiment (include failed reruns)

### AWS Braket Cost Estimation (Conceptual)
```python
# estimated_seconds = 2.5
# rate_per_sec = 0.00035  # example placeholder
# est_cost = estimated_seconds * rate_per_sec
# print(f"Estimated cost: ${est_cost:.4f}")
```

### Queue Time Heuristic
- If queue > threshold (e.g., 45 min) & alternative backend within 5% fidelity → switch
- Maintain rolling log of historical queue times per backend & hour of day

### Advanced Cost & Resource Optimization
```python
import time
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple

class QuantumResourceOptimizer:
    """Comprehensive resource optimization for quantum computing workflows"""
    
    def __init__(self):
        self.execution_history = []
        self.cost_tracking = {}
        self.queue_predictions = {}
    
    def estimate_comprehensive_cost(self, platform, backend_name, circuit, shots, 
                                  include_development=True):
        """Comprehensive cost estimation including hidden costs"""
        
        # Base execution costs (per platform)
        base_costs = {
            'ibm': {'per_shot': 0.0, 'per_second': 0.0, 'queue_multiplier': 1.0},
            'aws_braket_ionq': {'per_shot': 0.00003, 'per_second': 0.0001, 'queue_multiplier': 1.2},
            'aws_braket_rigetti': {'per_shot': 0.00001, 'per_second': 0.00005, 'queue_multiplier': 1.1},
            'azure_ionq': {'per_shot': 0.00003, 'per_second': 0.0001, 'queue_multiplier': 1.1},
            'dwave': {'per_second': 0.0002, 'per_shot': 0.0, 'queue_multiplier': 1.0}
        }
        
        # Estimate execution time based on circuit complexity
        estimated_time = self._estimate_execution_time(circuit, backend_name)
        
        # Base costs
        base_cost = base_costs.get(platform, base_costs['ibm'])
        execution_cost = (shots * base_cost['per_shot'] + 
                         estimated_time * base_cost['per_second'])
        
        # Queue time penalty (opportunity cost)
        queue_cost = estimated_time * 0.1 * base_cost['queue_multiplier']
        
        # Development costs (if requested)
        development_cost = 0
        if include_development:
            # Estimate iterations needed for circuit development
            complexity_factor = circuit.depth() * circuit.num_qubits / 100
            estimated_iterations = max(5, int(10 * complexity_factor))
            development_cost = execution_cost * estimated_iterations * 0.1  # 10% of execution per iteration
        
        total_cost = execution_cost + queue_cost + development_cost
        
        return {
            'execution_cost': execution_cost,
            'queue_cost': queue_cost,
            'development_cost': development_cost,
            'total_cost': total_cost,
            'estimated_time': estimated_time,
            'shots': shots,
            'cost_per_shot': total_cost / shots if shots > 0 else 0
        }
    
    def _estimate_execution_time(self, circuit, backend_name):
        """Estimate circuit execution time based on hardware characteristics"""
        
        # Hardware timing characteristics (approximate)
        timing_profiles = {
            'ibm': {'gate_time': 0.1e-6, 'readout_time': 1e-6, 'overhead': 10e-3},
            'ionq': {'gate_time': 10e-6, 'readout_time': 0.1e-6, 'overhead': 5e-3},
            'rigetti': {'gate_time': 0.05e-6, 'readout_time': 1e-6, 'overhead': 8e-3},
            'dwave': {'gate_time': 0, 'readout_time': 0, 'overhead': 20e-6}  # Annealing time
        }
        
        # Determine profile from backend name
        profile_key = 'ibm'  # default
        for key in timing_profiles:
            if key in backend_name.lower():
                profile_key = key
                break
        
        profile = timing_profiles[profile_key]
        
        # Calculate execution time
        gate_count = circuit.size()
        execution_time = (gate_count * profile['gate_time'] + 
                         circuit.num_qubits * profile['readout_time'] + 
                         profile['overhead'])
        
        return execution_time
    
    def optimize_shot_allocation(self, target_precision, max_budget, cost_per_shot):
        """Optimize shot count for target precision within budget"""
        
        # Statistical error scales as 1/sqrt(shots)
        # For probability estimation: σ = sqrt(p(1-p)/n) ≈ 0.5/sqrt(n) worst case
        
        def precision_for_shots(shots):
            return 0.5 / np.sqrt(shots)  # Worst-case standard error
        
        def shots_for_precision(precision):
            return int(np.ceil((0.5 / precision)**2))
        
        # Find shot count for target precision
        shots_needed = shots_for_precision(target_precision)
        cost_needed = shots_needed * cost_per_shot
        
        if cost_needed <= max_budget:
            return {
                'recommended_shots': shots_needed,
                'cost': cost_needed,
                'achieved_precision': target_precision,
                'budget_utilization': cost_needed / max_budget
            }
        else:
            # Budget-constrained: find best precision within budget
            max_shots = int(max_budget / cost_per_shot)
            achieved_precision = precision_for_shots(max_shots)
            
            return {
                'recommended_shots': max_shots,
                'cost': max_budget,
                'achieved_precision': achieved_precision,
                'budget_utilization': 1.0,
                'precision_shortfall': target_precision - achieved_precision
            }
    
    def queue_time_optimization(self, backends_info, target_completion_time=None):
        """Optimize backend selection based on queue times and completion targets"""
        
        current_time = datetime.now()
        
        recommendations = []
        
        for backend in backends_info:
            queue_time = backend.get('queue_time', 0)  # seconds
            estimated_completion = current_time + timedelta(seconds=queue_time)
            
            score = 0
            notes = []
            
            # Score based on queue time (lower is better)
            if queue_time < 300:  # < 5 minutes
                score += 10
                notes.append("Very short queue")
            elif queue_time < 1800:  # < 30 minutes
                score += 7
                notes.append("Acceptable queue")
            elif queue_time < 3600:  # < 1 hour
                score += 4
                notes.append("Long queue")
            else:
                score += 1
                notes.append("Very long queue")
            
            # Check completion time constraint
            if target_completion_time:
                target_dt = datetime.fromisoformat(target_completion_time) if isinstance(target_completion_time, str) else target_completion_time
                if estimated_completion <= target_dt:
                    score += 5
                    notes.append("Meets deadline")
                else:
                    score -= 3
                    notes.append("Misses deadline")
            
            # Factor in backend quality
            error_rate = backend.get('avg_error_rate', 0.05)
            if error_rate < 0.01:
                score += 3
                notes.append("Low error rate")
            elif error_rate > 0.05:
                score -= 2
                notes.append("High error rate")
            
            recommendations.append({
                'backend': backend,
                'score': score,
                'estimated_completion': estimated_completion,
                'notes': notes
            })
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
    
    def batch_optimization(self, circuits, max_batch_size=20):
        """Optimize circuit batching for efficient execution"""
        
        # Group circuits by similarity for efficient batching
        def circuit_signature(qc):
            return (qc.num_qubits, qc.depth(), tuple(sorted(qc.count_ops().items())))
        
        # Group similar circuits
        circuit_groups = {}
        for i, circuit in enumerate(circuits):
            sig = circuit_signature(circuit)
            if sig not in circuit_groups:
                circuit_groups[sig] = []
            circuit_groups[sig].append((i, circuit))
        
        # Create batches
        batches = []
        for group_circuits in circuit_groups.values():
            # Split large groups into batches
            for i in range(0, len(group_circuits), max_batch_size):
                batch = group_circuits[i:i+max_batch_size]
                batches.append({
                    'circuits': batch,
                    'signature': circuit_signature(batch[0][1]),
                    'size': len(batch)
                })
        
        return batches

# Usage examples
optimizer = QuantumResourceOptimizer()

# Example 1: Cost optimization
test_circuit = QuantumCircuit(5)
test_circuit.h(range(5))
for i in range(4):
    test_circuit.cx(i, i+1)
test_circuit.measure_all()

cost_analysis = optimizer.estimate_comprehensive_cost(
    'aws_braket_ionq', 'ionq_qpu', test_circuit, 2048)

print("Cost Analysis:")
print(f"  Execution: ${cost_analysis['execution_cost']:.4f}")
print(f"  Development: ${cost_analysis['development_cost']:.4f}")
print(f"  Total: ${cost_analysis['total_cost']:.4f}")
print(f"  Cost per shot: ${cost_analysis['cost_per_shot']:.6f}")

# Example 2: Shot optimization
shot_optimization = optimizer.optimize_shot_allocation(
    target_precision=0.01, max_budget=10.0, cost_per_shot=0.00003)

print(f"\nShot Optimization:")
print(f"  Recommended shots: {shot_optimization['recommended_shots']}")
print(f"  Achieved precision: {shot_optimization['achieved_precision']:.4f}")
print(f"  Budget utilization: {shot_optimization['budget_utilization']:.1%}")

# Example 3: Queue optimization
sample_backends = [
    {'name': 'ibm_hanoi', 'queue_time': 1800, 'avg_error_rate': 0.02},
    {'name': 'ionq_qpu', 'queue_time': 300, 'avg_error_rate': 0.005},
    {'name': 'rigetti_aspen', 'queue_time': 2400, 'avg_error_rate': 0.03}
]

queue_recommendations = optimizer.queue_time_optimization(sample_backends)

print(f"\nQueue Optimization:")
for i, rec in enumerate(queue_recommendations[:2]):
    backend = rec['backend']
    print(f"  {i+1}. {backend['name']}: score={rec['score']}, notes={rec['notes']}")
```

---
## 7.8 Project: Run & Benchmark on Real Hardware
### Goal
Design, execute, and analyze a small variational circuit on at least two different hardware platforms (or one hardware + one simulator + one noisy simulation), applying mitigation and optimization techniques.

### Deliverables
1. Experiment Design Document (objective, circuits, metrics, hypotheses)
2. Execution Script (with backend selection + retries)
3. Benchmark Report (tables + plots: fidelity, depth, error rates)
4. Cost & Queue Analysis (shots, time, $ estimate if applicable)
5. Improvement Log (what changed between iterations & why)

### Suggested Steps
| Step | Action | Output |
|------|--------|--------|
| 1 | Define target circuit family (e.g., hardware-efficient ansatz) | Spec sheet |
| 2 | Simulate ideal & noisy models | Baseline metrics |
| 3 | Select 2 backends & record properties | Backend table |
| 4 | Transpile & record gate counts | Optimization delta |
| 5 | Run raw execution | Counts snapshot |
| 6 | Apply mitigation & re-run | Improved counts |
| 7 | Compare results & compute fidelity | Report section |
| 8 | Document cost & queue | Cost table |

### Evaluation Rubric (Excerpt)
| Criterion | Exemplary | Adequate | Needs Work |
|----------|----------|----------|------------|
| Experimental Design | Clear hypotheses & metrics | Basic description | Unclear goals |
| Data Quality | Mitigation + drift handling | Partial mitigation | Raw only |
| Analysis Depth | Cross-platform + cost insights | One platform basic stats | Minimal commentary |
| Reproducibility | Script + config + seed control | Some automation | Manual steps only |

#### Starter Experiment Script Skeleton
```python
import json, time
from qiskit import QuantumCircuit, transpile

def build_ansatz(n):
    qc = QuantumCircuit(n)
    for i in range(n): qc.ry(0.2*i, i)
    for i in range(n-1): qc.cx(i, i+1)
    qc.measure_all()
    return qc

def run_experiment(backend_name='ibmq_lima', shots=2048):
    # provider = IBMQ.get_provider()
    # backend = provider.get_backend(backend_name)
    qc = build_ansatz(5)
    # compiled = transpile(qc, backend, optimization_level=3)
    meta = {
        'backend': backend_name,
        'shots': shots,
        'circuit_depth': qc.depth(),
        'timestamp': time.time()
    }
    # job = backend.run(compiled, shots=shots)
    # result = job.result()
    # meta['job_id'] = job.job_id()
    # counts = result.get_counts()
    counts = {}
    print(json.dumps(meta, indent=2))
    return counts, meta

if __name__ == "__main__":
    run_experiment()
```

---
## 7.9 Checklist & Recap
- Understood modality differences & trade-offs
- Retrieved and interpreted hardware calibration metrics
- Performed backend-aware transpilation & layout selection
- Applied targeted error mitigation (readout, DD, extrapolation)
- Benchmarked vs simulators & across platforms
- Managed cost, queue times, and batching strategy
- Designed a reproducible hardware experiment

### Key Takeaways
- There is no universally "best" hardware—align architecture with workload profile
- Transpilation + layout choices materially affect success probability
- Calibration drift is an operational reality—design monitoring loops
- Mitigation adds overhead; quantify ROI before scaling
- Platform diversity enables comparative advantage—exploit it strategically

### Further Exploration
- Pulse-level calibration & custom scheduling
- Mid-circuit measurement & conditional branching (emerging features)
- Advanced neutral atom programmability (geometry reconfiguration)
- Cross-entropy benchmarking & randomized compiling

---
## 7.10 References & Resources
- IBM Quantum Documentation: https://quantum-computing.ibm.com/docs/
- AWS Braket Developer Guide: https://docs.aws.amazon.com/braket/
- Azure Quantum Docs: https://learn.microsoft.com/azure/quantum/
- Xanadu Photonic Hardware: https://xanadu.ai/
- IonQ Technical Papers: https://ionq.com/resources
- D-Wave System Overview: https://www.dwavesys.com/
- Randomized Benchmarking Primer
- NIST Quantum Computation Metrics Working Group Reports

---
End of Module 7
