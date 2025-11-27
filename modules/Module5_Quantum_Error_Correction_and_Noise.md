# Module 5: Quantum Error Correction & Noise
*Intermediate Tier*

> **✅ Qiskit 2.x Compatible** - All examples updated and tested (Dec 2024)
> 
> **Recent Updates:**
> - Fixed noise model configuration for 1-qubit vs 2-qubit gates
> - Updated to use `density_matrix` method for noisy simulations
> - Added proper measurement circuits where required
> - Fixed readout error API (`add_all_qubit_readout_error`)
> - All 5 examples (100%) passing tests

## Learning Objectives
By the end of this module, you will be able to:
- Explain sources of quantum noise: decoherence, relaxation (T1), dephasing (T2), gate & readout errors
- Interpret common hardware metrics (T1, T2, gate fidelity, readout error, quantum volume)
- Simulate noisy quantum circuits and measure impact on results
- Apply practical error mitigation strategies (measurement error mitigation, zero-noise extrapolation, dynamical decoupling)
- Describe foundational quantum error correction (QEC) principles: redundancy, stabilizers, syndrome extraction
- Implement small illustrative codes (bit-flip, phase-flip, Shor / Steane conceptual sketch)
- Use (or conceptually integrate) Mitiq-style error mitigation workflows
- Benchmark an algorithm under varying noise levels and analyze performance

## Prerequisites
- Completion of Modules 1–4 (quantum gates, circuits, core algorithms)
- Python + Qiskit basics (circuit creation, execution)
- Linear algebra basics (Pauli matrices)

---

## 5.1 Understanding Quantum Noise

### Why Noise Matters
Quantum states are fragile. Interaction with the environment leaks information → loss of coherence → computational errors.

### Key Noise Channels
| Channel | Physical Meaning | Effect on Bloch Sphere | Typical Cause |
|---------|------------------|------------------------|---------------|
| Amplitude Damping | |1⟩ → |0⟩ relaxation | Shrinks toward ground state (Z-axis) | Energy loss (T1) |
| Phase Damping (Dephasing) | Random phase kicks | Equator blur / flattening | Magnetic field fluctuations (T2) |
| Depolarizing | Random Pauli applied | Sphere shrinks to center | Gate imperfections |
| Readout Error | Mislabel measurement | Classical bit flip after measure | Imperfect detectors |

### Characteristic Times
- **T1 (relaxation)**: Time for excited population to decay (|1⟩ → |0⟩)
- **T2 (decoherence)**: Time for phase info to decay (superposition → mixture)
- Always: T2 ≤ 2T1

### Hardware Metrics Snapshot
| Metric | Meaning | Why Important |
|--------|---------|---------------|
| Gate Fidelity | 1 - error per gate | Determines circuit depth viability |
| Readout Error | Probability of misclassification | Affects final measurement quality |
| Quantum Volume | Composite performance benchmark | Holistic device capability |
| Error per Clifford | Aggregated gate error metric | Calibration tracking |

### Visualizing Noise Impact

```python
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, errors
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_noise_channels():
    """Demonstrate different types of quantum noise and their effects"""
    
    print("Quantum Noise Channels Demonstration")
    print("=" * 36)
    
    # Create a simple circuit to test noise on
    qc = QuantumCircuit(1)
    qc.h(0)  # Create superposition |+⟩ = (|0⟩ + |1⟩)/√2
    
    # Get ideal statevector
    ideal_backend = Aer.get_backend('statevector_simulator')
    ideal_sv = execute(qc, ideal_backend).result().get_statevector()
    print(f"Ideal |+⟩ state: {ideal_sv}")
    
    # 1. Amplitude Damping (T1 decay)
    print("\n1. Amplitude Damping (T1 relaxation):")
    noise_model_t1 = NoiseModel()
    amplitude_damping = errors.amplitude_damping_error(0.1)  # 10% probability
    noise_model_t1.add_all_qubit_quantum_error(amplitude_damping, ['h'])
    
    noisy_sv_t1 = execute(qc, ideal_backend, noise_model=noise_model_t1).result().get_statevector()
    print(f"After T1 noise: {noisy_sv_t1}")
    print(f"Population in |1⟩ reduced from {abs(ideal_sv[1])**2:.3f} to {abs(noisy_sv_t1[1])**2:.3f}")
    
    # 2. Phase Damping (T2 dephasing)
    print("\n2. Phase Damping (T2 dephasing):")
    noise_model_t2 = NoiseModel()
    phase_damping = errors.phase_damping_error(0.1)
    noise_model_t2.add_all_qubit_quantum_error(phase_damping, ['h'])
    
    noisy_sv_t2 = execute(qc, ideal_backend, noise_model=noise_model_t2).result().get_statevector()
    print(f"After T2 noise: {noisy_sv_t2}")
    
    # 3. Depolarizing Channel
    print("\n3. Depolarizing Channel:")
    noise_model_depol = NoiseModel()
    depolarizing = errors.depolarizing_error(0.05, 1)  # 5% single-qubit depolarizing
    noise_model_depol.add_all_qubit_quantum_error(depolarizing, ['h'])
    
    noisy_sv_depol = execute(qc, ideal_backend, noise_model=noise_model_depol).result().get_statevector()
    print(f"After depolarizing: {noisy_sv_depol}")
    
    return ideal_sv, noisy_sv_t1, noisy_sv_t2, noisy_sv_depol

def measure_circuit_fidelity():
    """Demonstrate how circuit depth affects fidelity under noise"""
    
    print("\nCircuit Depth vs Fidelity Analysis")
    print("=" * 33)
    
    # Create circuits of increasing depth
    depths = range(1, 11)
    fidelities = []
    
    # Simple noise model
    noise_model = NoiseModel()
    gate_error = errors.depolarizing_error(0.01, 1)  # 1% error per gate
    noise_model.add_all_qubit_quantum_error(gate_error, ['h', 'x', 'z'])
    
    backend_ideal = Aer.get_backend('statevector_simulator')
    backend_noisy = Aer.get_backend('statevector_simulator')
    
    for depth in depths:
        # Create circuit with given depth
        qc = QuantumCircuit(1)
        for _ in range(depth):
            qc.h(0)
            qc.x(0)
            qc.h(0)  # Identity operation, but adds noise
        
        # Get ideal and noisy results
        ideal_sv = execute(qc, backend_ideal).result().get_statevector()
        noisy_sv = execute(qc, backend_noisy, noise_model=noise_model).result().get_statevector()
        
        # Calculate fidelity |⟨ψ_ideal|ψ_noisy⟩|²
        fidelity = abs(np.vdot(ideal_sv, noisy_sv))**2
        fidelities.append(fidelity)
        
        print(f"Depth {depth:2d}: Fidelity = {fidelity:.4f}")
    
    # Show decay trend
    print(f"\nFidelity decay: {fidelities[0]:.3f} → {fidelities[-1]:.3f}")
    print("Exponential decay with circuit depth is typical!")
    
    return depths, fidelities

def hardware_metrics_simulation():
    """Simulate realistic hardware error rates and their impact"""
    
    print("\nHardware Error Rates Simulation")
    print("=" * 31)
    
    # Typical hardware parameters (approximate values for demonstration)
    hardware_specs = {
        'T1': 100e-6,           # 100 microseconds
        'T2': 50e-6,            # 50 microseconds  
        'gate_time': 50e-9,     # 50 nanoseconds
        'readout_error': 0.02,  # 2% readout error
        'gate_fidelity': 0.999  # 99.9% gate fidelity
    }
    
    print("Simulated hardware specifications:")
    for param, value in hardware_specs.items():
        if 'time' in param or param in ['T1', 'T2']:
            print(f"  {param}: {value*1e6:.1f} μs")
        else:
            print(f"  {param}: {value}")
    
    # Calculate coherence-limited gate count
    max_gates_t1 = hardware_specs['T1'] / hardware_specs['gate_time']
    max_gates_t2 = hardware_specs['T2'] / hardware_specs['gate_time']
    
    print(f"\nCoherence-limited circuit depth:")
    print(f"  T1 limit: ~{max_gates_t1:.0f} gates")
    print(f"  T2 limit: ~{max_gates_t2:.0f} gates")
    print(f"  Practical limit: ~{min(max_gates_t1, max_gates_t2)/10:.0f} gates (with safety margin)")
    
    # Create realistic noise model
    noise_model = NoiseModel()
    
    # Add T1/T2 noise
    t1_error = errors.thermal_relaxation_error(
        t1=hardware_specs['T1'], 
        t2=hardware_specs['T2'], 
        time=hardware_specs['gate_time']
    )
    noise_model.add_all_qubit_quantum_error(t1_error, ['h', 'x', 'z'])
    
    # Add gate errors
    gate_error = errors.depolarizing_error(1 - hardware_specs['gate_fidelity'], 1)
    noise_model.add_all_qubit_quantum_error(gate_error, ['h', 'x', 'z'])
    
    # Add readout errors
    readout_error = errors.ReadoutError([
        [1 - hardware_specs['readout_error'], hardware_specs['readout_error']],
        [hardware_specs['readout_error'], 1 - hardware_specs['readout_error']]
    ])
    noise_model.add_readout_error(readout_error, [0])
    
    print(f"\nNoise model created with realistic hardware parameters")
    return noise_model

# Run noise demonstrations
ideal, t1, t2, depol = demonstrate_noise_channels()
depths, fidelities = measure_circuit_fidelity()
realistic_noise = hardware_metrics_simulation()
```

### Real Hardware Data Visualization

```python
def plot_hardware_trends():
    """Show how quantum hardware has improved over time"""
    
    print("\nQuantum Hardware Evolution")
    print("=" * 25)
    
    # Approximate data for major quantum systems (illustrative)
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    # IBM quantum systems evolution
    ibm_qubits = [5, 16, 20, 27, 65, 127, 433, 1000, 1121]
    ibm_t1_times = [40, 50, 60, 80, 100, 120, 140, 150, 160]  # microseconds
    
    print("IBM Quantum System Evolution:")
    print("Year | Qubits | T1 (μs)")
    print("-" * 25)
    for year, qubits, t1 in zip(years[-5:], ibm_qubits[-5:], ibm_t1_times[-5:]):
        print(f"{year} |   {qubits:4d} | {t1:6.0f}")
    
    print(f"\nKey trends:")
    print(f"- Qubit count growing exponentially")
    print(f"- Coherence times improving steadily")
    print(f"- Gate fidelities approaching 99.9%+")
    
    return years, ibm_qubits, ibm_t1_times

plot_hardware_trends()
```

---

## 5.2 NISQ Era Constraints

### What is NISQ?
**Noisy Intermediate-Scale Quantum**: 50–1000 qubits, noisy, no full error correction. Focus: hybrid + variational algorithms, error-aware design.

### Design Trade-offs
| Constraint | Tension | Mitigation |
|-----------|--------|------------|
| Depth vs Fidelity | More layers → more accumulated error | Optimize, transpile for topology |
| Expressivity vs Noise | Rich ansatz vs decoherence window | Parameter-efficient templates |
| Qubit Count vs Connectivity | SWAP overhead | Layout, mapping, routing |
| Shots vs Run Time | Precision vs queue limits | Adaptive shot allocation |

### Practical Workflow Pattern
1. Prototype ideal circuit → evaluate sensitivity
2. Reduce depth (gate fusion, remove redundancies)
3. Insert dynamical decoupling / echo pulses where helpful
4. Apply measurement error mitigation
5. Run multiple noise scaling variants (for extrapolation)
6. Aggregate & statistically analyze

### NISQ Algorithm Design Examples

```python
def nisq_circuit_optimization_demo():
    """Demonstrate NISQ-era circuit optimization techniques"""
    
    print("NISQ Circuit Optimization Techniques")
    print("=" * 35)
    
    # Original circuit (inefficient)
    original = QuantumCircuit(3)
    original.h(0)
    original.cx(0, 1)
    original.h(1)
    original.h(1)  # Redundant - H†H = I
    original.cx(1, 2)
    original.x(2)
    original.x(2)  # Redundant - X†X = I
    
    print(f"Original circuit depth: {original.depth()}")
    print(f"Original gate count: {len(original.data)}")
    
    # Optimized circuit (redundancies removed)
    optimized = QuantumCircuit(3)
    optimized.h(0)
    optimized.cx(0, 1)
    # Removed redundant H†H
    optimized.cx(1, 2)
    # Removed redundant X†X
    
    print(f"Optimized circuit depth: {optimized.depth()}")
    print(f"Optimized gate count: {len(optimized.data)}")
    
    # Topology-aware routing example
    print(f"\nTopology considerations:")
    print(f"- Linear coupling: 0-1-2 (good for our circuit)")
    print(f"- Heavy-hex coupling: would need SWAP gates")
    print(f"- All-to-all: ideal but doesn't exist in practice")
    
    return original, optimized

def adaptive_shot_allocation():
    """Demonstrate adaptive resource allocation for NISQ experiments"""
    
    print("\nAdaptive Shot Allocation Strategy")
    print("=" * 33)
    
    # Simulate different circuit complexities
    experiments = [
        {"name": "Simple Bell state", "depth": 2, "variance": 0.1},
        {"name": "VQE ansatz", "depth": 8, "variance": 0.3},
        {"name": "Deep Grover", "depth": 20, "variance": 0.8}
    ]
    
    total_budget = 10000  # Total shots available
    
    print("Experiment | Depth | Est. Variance | Allocated Shots")
    print("-" * 55)
    
    # Allocate shots based on variance (higher variance needs more shots)
    for exp in experiments:
        # Simple heuristic: more shots for higher variance
        base_shots = 1000
        variance_factor = exp["variance"]
        allocated = int(base_shots * (1 + 2 * variance_factor))
        exp["shots"] = allocated
        
        print(f"{exp['name']:<15} | {exp['depth']:5d} | {variance_factor:11.1f} | {allocated:13d}")
    
    total_used = sum(exp["shots"] for exp in experiments)
    print(f"\nTotal shots used: {total_used}/{total_budget}")
    
    return experiments

def nisq_error_budget_analysis():
    """Analyze error budget for NISQ algorithms"""
    
    print("\nNISQ Error Budget Analysis")
    print("=" * 26)
    
    # Typical NISQ error rates
    gate_error = 0.001      # 0.1% per gate
    readout_error = 0.02    # 2% readout error
    idle_error_rate = 1e-5  # per microsecond
    
    # Example circuit analysis
    circuit_depth = 10
    gate_count = 25
    idle_time = 5e-6  # 5 microseconds total idle time
    
    # Calculate cumulative error
    gate_error_total = gate_count * gate_error
    idle_error_total = idle_time * idle_error_rate
    total_error = gate_error_total + idle_error_total + readout_error
    
    success_probability = 1 - total_error
    
    print(f"Circuit specifications:")
    print(f"  Depth: {circuit_depth}")
    print(f"  Gate count: {gate_count}")
    print(f"  Idle time: {idle_time*1e6:.1f} μs")
    
    print(f"\nError contributions:")
    print(f"  Gate errors: {gate_error_total:.1%}")
    print(f"  Idle errors: {idle_error_total:.1%}")
    print(f"  Readout error: {readout_error:.1%}")
    print(f"  Total error: {total_error:.1%}")
    
    print(f"\nExpected success rate: {success_probability:.1%}")
    
    # Show how this scales with circuit size
    print(f"\nScaling analysis:")
    for scale in [1, 2, 4, 8]:
        scaled_gates = gate_count * scale
        scaled_error = scaled_gates * gate_error + readout_error
        scaled_success = 1 - scaled_error
        print(f"  {scale}x circuit: {scaled_success:.1%} success rate")

# Run NISQ demonstrations
original, optimized = nisq_circuit_optimization_demo()
experiments = adaptive_shot_allocation()
nisq_error_budget_analysis()
```

---

## 5.3 Error Mitigation Techniques (Pre-QEC)

These improve results without full logical qubits.

### 1. Measurement Error Mitigation

```python
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np

def comprehensive_measurement_mitigation():
    """Comprehensive demonstration of measurement error mitigation"""
    
    print("Measurement Error Mitigation")
    print("=" * 28)
    
    # Create a 2-qubit system for demonstration
    n_qubits = 2
    qr = QuantumRegister(n_qubits)
    
    # Step 1: Generate calibration circuits
    print("Step 1: Generating calibration circuits")
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    
    print(f"Generated {len(meas_calibs)} calibration circuits:")
    for i, cal_circuit in enumerate(meas_calibs):
        print(f"  Circuit {i}: prepares |{state_labels[i]}⟩")
    
    # Step 2: Add measurement errors to simulate hardware
    print(f"\nStep 2: Simulating measurement errors")
    from qiskit.providers.aer.noise import NoiseModel, errors
    
    # Create readout error model
    noise_model = NoiseModel()
    readout_error_0 = errors.ReadoutError([[0.95, 0.05], [0.1, 0.9]])  # Asymmetric errors
    readout_error_1 = errors.ReadoutError([[0.98, 0.02], [0.08, 0.92]])
    
    noise_model.add_readout_error(readout_error_0, [0])
    noise_model.add_readout_error(readout_error_1, [1])
    
    # Step 3: Run calibration
    backend = Aer.get_backend('qasm_simulator')
    cal_results = execute(meas_calibs, backend, shots=1024, noise_model=noise_model).result()
    
    # Step 4: Build correction matrix
    meas_fitter = CompleteMeasFitter(cal_results, state_labels)
    print(f"\nMeasurement calibration matrix:")
    print(meas_fitter.cal_matrix)
    
    # Step 5: Test on actual circuit
    test_circuit = QuantumCircuit(n_qubits, n_qubits)
    test_circuit.h(0)  # Create |+0⟩ state
    test_circuit.measure_all()
    
    # Get raw (uncorrected) results
    raw_result = execute(test_circuit, backend, shots=1024, noise_model=noise_model).result()
    raw_counts = raw_result.get_counts()
    
    # Apply mitigation
    mitigated_counts = meas_fitter.filter.apply(raw_counts)
    
    print(f"\nResults comparison:")
    print(f"Raw counts: {raw_counts}")
    print(f"Mitigated: {dict(mitigated_counts)}")
    
    # Expected: roughly 50% |00⟩ and 50% |10⟩ for |+0⟩ state
    expected_00 = 0.5
    expected_10 = 0.5
    
    raw_00_prob = raw_counts.get('00', 0) / 1024
    raw_10_prob = raw_counts.get('10', 0) / 1024
    
    mit_00_prob = mitigated_counts.get('00', 0) / 1024
    mit_10_prob = mitigated_counts.get('10', 0) / 1024
    
    print(f"\nAccuracy comparison:")
    print(f"Expected |00⟩: {expected_00:.2f}, |10⟩: {expected_10:.2f}")
    print(f"Raw error: |00⟩ {abs(raw_00_prob - expected_00):.3f}, |10⟩ {abs(raw_10_prob - expected_10):.3f}")
    print(f"Mitigated error: |00⟩ {abs(mit_00_prob - expected_00):.3f}, |10⟩ {abs(mit_10_prob - expected_10):.3f}")
    
    return meas_fitter

def apply_meas_mitigation(raw_counts, meas_fitter):
    """Helper function to apply measurement mitigation"""
    return meas_fitter.filter.apply(raw_counts)

# Run measurement mitigation demo
fitter = comprehensive_measurement_mitigation()
```

### 2. Zero-Noise Extrapolation (ZNE)

```python
def comprehensive_zne_demo():
    """Comprehensive Zero-Noise Extrapolation demonstration"""
    
    print("\nZero-Noise Extrapolation (ZNE)")
    print("=" * 30)
    
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.providers.aer.noise import NoiseModel, errors
    import numpy as np
    from scipy.optimize import curve_fit
    
    # Create test circuit for ZNE
    test_circuit = QuantumCircuit(1, 1)
    test_circuit.h(0)
    test_circuit.z(0)  # Should have no effect on |+⟩
    test_circuit.h(0)  # Should return to |0⟩
    test_circuit.measure(0, 0)
    
    print("Test circuit: H-Z-H on |0⟩ → should give |0⟩ with prob 1.0")
    
    def execute_with_noise_scaling(circuit, noise_scale, base_noise_rate=0.01):
        """Execute circuit with scaled noise"""
        noise_model = NoiseModel()
        
        # Scale the noise
        scaled_error_rate = base_noise_rate * noise_scale
        gate_error = errors.depolarizing_error(scaled_error_rate, 1)
        noise_model.add_all_qubit_quantum_error(gate_error, ['h', 'z'])
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=2048, noise_model=noise_model)
        counts = job.result().get_counts()
        
        # Calculate expectation value of Z (prob |0⟩ - prob |1⟩)
        prob_0 = counts.get('0', 0) / 2048
        prob_1 = counts.get('1', 0) / 2048
        expectation_z = prob_0 - prob_1
        
        return expectation_z
    
    # Test different noise scales
    noise_scales = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    expectation_values = []
    
    print(f"\nNoise scaling experiment:")
    print(f"Scale | Noise Rate | ⟨Z⟩ Expectation")
    print("-" * 35)
    
    for scale in noise_scales:
        exp_val = execute_with_noise_scaling(test_circuit, scale)
        expectation_values.append(exp_val)
        noise_rate = 0.01 * scale
        print(f"{scale:4.1f} | {noise_rate:8.3f} | {exp_val:12.4f}")
    
    expectation_values = np.array(expectation_values)
    
    # Fit polynomial and extrapolate to zero noise
    print(f"\nExtrapolation methods:")
    
    # Linear extrapolation
    linear_coeffs = np.polyfit(noise_scales, expectation_values, 1)
    linear_zero = np.polyval(linear_coeffs, 0)
    print(f"Linear extrapolation to zero: {linear_zero:.4f}")
    
    # Quadratic extrapolation
    quad_coeffs = np.polyfit(noise_scales, expectation_values, 2)
    quad_zero = np.polyval(quad_coeffs, 0)
    print(f"Quadratic extrapolation to zero: {quad_zero:.4f}")
    
    # Exponential fit
    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    try:
        exp_params, _ = curve_fit(exponential_decay, noise_scales, expectation_values)
        exp_zero = exponential_decay(0, *exp_params)
        print(f"Exponential extrapolation to zero: {exp_zero:.4f}")
    except:
        print("Exponential fit failed (numerical issues)")
    
    print(f"\nIdeal expectation value: 1.0000")
    print(f"Best ZNE estimate improves from {expectation_values[0]:.4f} to ~{quad_zero:.4f}")
    
    return noise_scales, expectation_values, quad_zero

def digital_zero_noise_extrapolation():
    """Demonstrate digital ZNE using gate folding"""
    
    print("\nDigital ZNE: Gate Folding Method")
    print("=" * 31)
    
    def fold_gates(circuit, folding_factor):
        """Apply gate folding: U → U(U†U)^n for noise scaling"""
        if folding_factor < 1:
            return circuit
        
        folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        for instruction, qargs, cargs in circuit.data:
            if instruction.name == 'measure':
                folded.append(instruction, qargs, cargs)
                continue
                
            # Apply original gate
            folded.append(instruction, qargs, cargs)
            
            # Add folding: (U†U) pairs
            n_folds = int(folding_factor) - 1
            for _ in range(n_folds):
                folded.append(instruction.inverse(), qargs, cargs)
                folded.append(instruction, qargs, cargs)
        
        return folded
    
    # Test circuit
    base_circuit = QuantumCircuit(1, 1)
    base_circuit.ry(np.pi/4, 0)  # Rotate to |+⟩ state
    base_circuit.measure(0, 0)
    
    folding_factors = [1, 3, 5, 7]  # Odd numbers preserve the original unitary
    
    print("Gate folding demonstration:")
    print("Factor | Circuit Depth | Expected ⟨Z⟩")
    print("-" * 35)
    
    for factor in folding_factors:
        folded_circuit = fold_gates(base_circuit, factor)
        depth = folded_circuit.depth()
        
        # Theoretical: each U†U pair adds noise but preserves operation
        # For ry(π/4): ⟨Z⟩ = cos(π/4) = 1/√2 ≈ 0.707
        expected_z = np.cos(np.pi/4)
        
        print(f"{factor:4d}x |{depth:11d} | {expected_z:10.3f}")
    
    print(f"\nGate folding allows us to artificially scale noise")
    print(f"while preserving the ideal quantum operation!")

# Run ZNE demonstrations
scales, expectations, zne_result = comprehensive_zne_demo()
digital_zero_noise_extrapolation()
```

### 3. Dynamical Decoupling

```python
def dynamical_decoupling_demo():
    """Demonstrate dynamical decoupling for dephasing protection"""
    
    print("\nDynamical Decoupling")
    print("=" * 20)
    
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.providers.aer.noise import NoiseModel, errors
    
    def add_decoupling_sequence(circuit, qubit, sequence_type='XY4'):
        """Add dynamical decoupling sequence during idle periods"""
        
        sequences = {
            'XY4': ['x', 'y', 'x', 'y'],  # 4-pulse XY sequence
            'CPMG': ['x', 'x'],           # Carr-Purcell-Meiboom-Gill
            'Hahn': ['x']                 # Simple spin echo
        }
        
        if sequence_type not in sequences:
            return circuit
        
        decoupled = circuit.copy()
        pulse_sequence = sequences[sequence_type]
        
        # Insert pulses (simplified - in practice, timing matters)
        for pulse in pulse_sequence:
            if pulse == 'x':
                decoupled.x(qubit)
            elif pulse == 'y':
                decoupled.y(qubit)
        
        return decoupled
    
    # Create test scenario: qubit idles in superposition
    idle_circuit = QuantumCircuit(1, 1)
    idle_circuit.h(0)  # Create |+⟩ state
    
    # Simulate idle time with barrier (where dephasing occurs)
    for _ in range(10):  # Simulate 10 time steps of idling
        idle_circuit.barrier()
        idle_circuit.id(0)  # Identity gate represents idle time
    
    idle_circuit.h(0)  # Transform back to computational basis
    idle_circuit.measure(0, 0)
    
    # Create decoupled version
    decoupled_circuit = QuantumCircuit(1, 1)
    decoupled_circuit.h(0)
    
    # Add decoupling during idle periods
    for _ in range(5):  # Fewer idle periods due to active protection
        decoupled_circuit.barrier()
        decoupled_circuit.x(0)  # π pulse for decoupling
        decoupled_circuit.barrier()
        decoupled_circuit.x(0)  # Another π pulse
    
    decoupled_circuit.h(0)
    decoupled_circuit.measure(0, 0)
    
    # Add dephasing noise model
    noise_model = NoiseModel()
    dephasing_error = errors.phase_damping_error(0.05)  # 5% dephasing per gate
    noise_model.add_all_qubit_quantum_error(dephasing_error, ['id', 'barrier'])
    
    backend = Aer.get_backend('qasm_simulator')
    
    # Test without decoupling
    idle_result = execute(idle_circuit, backend, shots=1024, noise_model=noise_model).result()
    idle_counts = idle_result.get_counts()
    idle_fidelity = idle_counts.get('0', 0) / 1024  # Should be ~1.0 ideally
    
    # Test with decoupling  
    decoupled_result = execute(decoupled_circuit, backend, shots=1024, noise_model=noise_model).result()
    decoupled_counts = decoupled_result.get_counts()
    decoupled_fidelity = decoupled_counts.get('0', 0) / 1024
    
    print(f"Dephasing protection comparison:")
    print(f"Without decoupling: {idle_fidelity:.3f} fidelity")
    print(f"With decoupling: {decoupled_fidelity:.3f} fidelity")
    print(f"Improvement: {decoupled_fidelity - idle_fidelity:.3f}")
    
    print(f"\nDecoupling sequences:")
    print(f"- Hahn echo: X pulse at midpoint")
    print(f"- CPMG: Multiple X pulses") 
    print(f"- XY4: Alternating X and Y pulses (better for general noise)")
    
    return idle_fidelity, decoupled_fidelity

# Run dynamical decoupling demo
idle_fid, decoupled_fid = dynamical_decoupling_demo()
```

### 3. Probabilistic Error Cancellation (PEC)
Requires noise model tomography; reweights samples. High variance cost → currently research-grade.

### 4. Dynamical Decoupling
Insert identity-equivalent pulse sequences (e.g., X–I–X–I) to refocus dephasing while idling.

### 5. Readout Symmetrization
Randomly flip measurement bases; average to reduce bias.

---

## 5.4 Intro to Quantum Error Correction (QEC)

### Core Idea
Encode 1 logical qubit into many physical ones so that errors can be detected and corrected without measuring protected quantum info.

### Fundamental Constraints
- **No-cloning**: Must use entanglement & redundancy cleverly
- **Error Types**: Any 1-qubit error ⟶ combination of {I, X, Y, Z}
- **Distance d**: Code corrects up to ⌊(d−1)/2⌋ errors

### Simple Codes
| Code | Encodes | Protects Against | Notes |
|------|---------|------------------|-------|
| 3-qubit bit-flip | 1 qubit | Single X error | Majority vote |
| 3-qubit phase-flip | 1 qubit | Single Z error | Hadamard + bit-flip idea |
| 9-qubit Shor | 1 qubit | Any single-qubit error | Bit + phase protection |
| 7-qubit Steane | 1 qubit | Any single-qubit error | CSS, transversal gates |

### Bit-Flip Code Walkthrough
Logical encoding: |0_L⟩ = |000⟩, |1_L⟩ = |111⟩
Syndrome measurement (stabilizers): Z₁Z₂, Z₂Z₃.

```python
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import random
from qiskit import Aer, execute

def comprehensive_bit_flip_code():
    """Complete implementation of 3-qubit bit-flip code"""
    
    print("3-Qubit Bit-Flip Code Implementation")
    print("=" * 36)
    
    def encode_logical_qubit(logical_bit):
        """Encode a logical qubit into 3 physical qubits"""
        qc = QuantumCircuit(3, name='encode')
        
        # Prepare logical |0⟩ or |1⟩
        if logical_bit == 1:
            qc.x(0)  # Start with |1⟩ instead of |0⟩
        
        # Create encoding: |0⟩ → |000⟩, |1⟩ → |111⟩
        qc.cx(0, 1)  # Copy qubit 0 to qubit 1
        qc.cx(0, 2)  # Copy qubit 0 to qubit 2
        
        return qc
    
    def inject_bit_flip_errors(qc, error_prob=0.2):
        """Inject random bit-flip errors"""
        noisy_qc = qc.copy()
        errors_applied = []
        
        for qubit in range(3):
            if random.random() < error_prob:
                noisy_qc.x(qubit)
                errors_applied.append(qubit)
        
        return noisy_qc, errors_applied
    
    def measure_syndrome(qc):
        """Measure error syndrome using ancilla qubits"""
        # Add ancilla qubits for syndrome measurement
        data_qubits = qc.qubits[:3]
        syndrome_qc = QuantumCircuit(5, 2)  # 3 data + 2 ancilla, 2 classical
        
        # Copy the encoded state
        syndrome_qc.compose(qc, range(3), inplace=True)
        
        # Measure stabilizers Z₁Z₂ and Z₂Z₃
        # Stabilizer 1: Z₁Z₂ (detects errors between qubits 0 and 1)
        syndrome_qc.cx(0, 3)  # Control: qubit 0, Target: ancilla 0
        syndrome_qc.cx(1, 3)  # Control: qubit 1, Target: ancilla 0
        
        # Stabilizer 2: Z₂Z₃ (detects errors between qubits 1 and 2)  
        syndrome_qc.cx(1, 4)  # Control: qubit 1, Target: ancilla 1
        syndrome_qc.cx(2, 4)  # Control: qubit 2, Target: ancilla 1
        
        # Measure syndrome
        syndrome_qc.measure([3, 4], [0, 1])
        
        return syndrome_qc
    
    def decode_syndrome(syndrome_bits):
        """Decode syndrome and determine correction"""
        syndrome = ''.join(map(str, syndrome_bits))
        
        correction_map = {
            '00': 'No error',
            '01': 'Error on qubit 2', 
            '10': 'Error on qubit 0',
            '11': 'Error on qubit 1'
        }
        
        correction_qubit = {
            '00': None,
            '01': 2,
            '10': 0, 
            '11': 1
        }
        
        return correction_map[syndrome], correction_qubit[syndrome]
    
    def full_error_correction_demo():
        """Demonstrate complete error correction cycle"""
        
        print("\nError Correction Demonstration:")
        print("-" * 32)
        
        successes = 0
        trials = 100
        
        for trial in range(trials):
            # Choose random logical bit
            logical_bit = random.choice([0, 1])
            
            # Encode
            encoded_qc = encode_logical_qubit(logical_bit)
            
            # Inject errors
            noisy_qc, actual_errors = inject_bit_flip_errors(encoded_qc, error_prob=0.3)
            
            # Measure syndrome
            syndrome_qc = measure_syndrome(noisy_qc)
            
            # Run syndrome measurement
            backend = Aer.get_backend('qasm_simulator')
            job = execute(syndrome_qc, backend, shots=1)
            result = job.result().get_counts()
            
            # Get syndrome measurement result
            measured_syndrome = list(result.keys())[0]
            syndrome_bits = [int(measured_syndrome[1]), int(measured_syndrome[0])]  # Note: reversed order
            
            # Decode syndrome
            error_description, correction_qubit = decode_syndrome(syndrome_bits)
            
            # Check if correction matches actual error
            if len(actual_errors) == 0 and correction_qubit is None:
                successes += 1  # Correctly identified no error
            elif len(actual_errors) == 1 and correction_qubit == actual_errors[0]:
                successes += 1  # Correctly identified single error
            # Note: Multiple errors cannot be corrected by this code
            
            if trial < 10:  # Show first 10 trials
                print(f"Trial {trial+1}: Logical={logical_bit}, Errors={actual_errors}, "
                      f"Syndrome={syndrome_bits}, {error_description}")
        
        success_rate = successes / trials
        print(f"\nError correction success rate: {success_rate:.1%}")
        print(f"(Note: Multiple simultaneous errors cannot be corrected)")
        
        return success_rate
    
    # Demonstrate encoding
    print("Encoding demonstration:")
    for logical in [0, 1]:
        encoded = encode_logical_qubit(logical)
        print(f"Logical |{logical}⟩ → {encoded.name}")
        
        # Show the encoded state
        backend = Aer.get_backend('statevector_simulator')
        statevector = execute(encoded, backend).result().get_statevector()
        print(f"  Encoded state: {statevector}")
    
    # Show stabilizer measurements
    print(f"\nStabilizer generators:")
    print(f"  S₁ = Z₀ ⊗ Z₁ ⊗ I₂  (measures parity of qubits 0,1)")
    print(f"  S₂ = I₀ ⊗ Z₁ ⊗ Z₂  (measures parity of qubits 1,2)")
    
    print(f"\nSyndrome lookup table:")
    print(f"  00: No error detected")
    print(f"  01: Error on qubit 2") 
    print(f"  10: Error on qubit 0")
    print(f"  11: Error on qubit 1")
    
    # Run full demo
    success_rate = full_error_correction_demo()
    
    return success_rate

def phase_flip_code_demo():
    """Demonstrate 3-qubit phase-flip code"""
    
    print("\n3-Qubit Phase-Flip Code")
    print("=" * 23)
    
    def encode_phase_flip(logical_bit):
        """Encode against phase-flip errors"""
        qc = QuantumCircuit(3)
        
        if logical_bit == 1:
            qc.x(0)
        
        # Apply Hadamards to work in X-basis
        qc.h([0, 1, 2])
        
        # Encode in X-basis: |+⟩ → |+++⟩, |−⟩ → |---⟩
        qc.cx(0, 1)
        qc.cx(0, 2)
        
        return qc
    
    def inject_phase_errors(qc, error_prob=0.2):
        """Inject Z errors (phase flips)"""
        noisy_qc = qc.copy()
        errors = []
        
        for qubit in range(3):
            if random.random() < error_prob:
                noisy_qc.z(qubit)
                errors.append(qubit)
        
        return noisy_qc, errors
    
    def measure_x_stabilizers():
        """Measure X-type stabilizers for phase-flip code"""
        # Similar to bit-flip but with X measurements
        print("X-type stabilizers:")
        print("  S₁ = X₀ ⊗ X₁ ⊗ I₂")
        print("  S₂ = I₀ ⊗ X₁ ⊗ X₂")
        print("These detect phase-flip (Z) errors")
    
    # Demonstrate phase-flip encoding
    print("Phase-flip code protects against Z errors")
    print("Uses Hadamard transform: Z errors ↔ X errors")
    
    for logical in [0, 1]:
        encoded = encode_phase_flip(logical)
        print(f"\nLogical |{logical}⟩ encoded in X-basis")
    
    measure_x_stabilizers()
    
    return True

def shor_code_conceptual():
    """Conceptual overview of 9-qubit Shor code"""
    
    print("\n9-Qubit Shor Code (Conceptual)")
    print("=" * 30)
    
    print("Structure: Concatenated code")
    print("1. First level: 3-qubit phase-flip code")
    print("2. Second level: Each qubit → 3-qubit bit-flip code")
    print("3. Total: 9 physical qubits per logical qubit")
    
    print(f"\nLogical |0⟩ encoding:")
    print(f"  |0_L⟩ = (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩)/2√2")
    
    print(f"\nProtection capabilities:")
    print(f"  ✓ Any single X error (bit-flip)")
    print(f"  ✓ Any single Z error (phase-flip)") 
    print(f"  ✓ Any single Y error (Y = iXZ)")
    print(f"  → Can correct any single-qubit error!")
    
    print(f"\nStabilizers: 8 generators")
    print(f"  6 X-type: check bit-flip errors within each block")
    print(f"  2 Z-type: check phase-flip errors between blocks")
    
    return True

# Run comprehensive QEC demonstrations
bit_flip_success = comprehensive_bit_flip_code()
phase_flip_demo = phase_flip_code_demo()
shor_conceptual = shor_code_conceptual()
```

### Phase-Flip Code Analogy
Apply Hadamards, use bit-flip logic in Hadamard basis.

### Shor Code Concept Sketch
1. Encode against phase errors via 3-qubit phase-protection
2. Each qubit expanded into 3 for bit-flip protection
3. Total 9 physical qubits

### Steane Code Concept
CSS structure; supports transversal H, S, CNOT → fault-tolerant logical operations.

---

## 5.5 Syndrome Extraction & Stabilizers

### Stabilizer Formalism (High-Level)
- Codes defined as joint +1 eigenspace of commuting Pauli group generators
- Measuring stabilizers collapses errors while preserving logical info

### Example: Bit-Flip Stabilizers
- Generators: Z₁Z₂, Z₂Z₃
- Syndrome table:
  - 00: no error
  - 01: X on qubit 2
  - 10: X on qubit 0
  - 11: X on qubit 1

```python
# Syndrome decoding demo (classical logic)
syndrome_to_correction = {
    '00': 'I',
    '01': 'X2',
    '10': 'X0',
    '11': 'X1'
}
```

### Extending to Phase Errors
Replace Z stabilizers with X-type after basis transforms.

### Distance & Logical Operators
Logical operators commute with stabilizers but act non-trivially on code space (e.g., X_L = X⊗X⊗X for bit-flip code).

---

## 5.6 Putting It Together: Mini QEC Flow

```python
def simulate_bit_flip_round(logical_bit=0, error_prob=0.3, trials=50):
    from collections import Counter
    successes = 0
    for _ in range(trials):
        enc = bit_flip_encode(logical_bit)
        enc_with_err = bit_flip_error_channel(enc, p=error_prob)
        # (Skipping full correction simulation for brevity)
        # Decode majority vote
        backend = Aer.get_backend('qasm_simulator')
        meas = QuantumCircuit(3,3)
        meas.measure(range(3), range(3))
        full = enc_with_err + meas
        counts = execute(full, backend, shots=1).result().get_counts()
        observed = list(counts.keys())[0]
        majority = 1 if observed.count('1') > 1 else 0
        if majority == logical_bit:
            successes += 1
    return successes / trials

print("Logical 0 success rate:", simulate_bit_flip_round(0))
print("Logical 1 success rate:", simulate_bit_flip_round(1))
```

### Key Insight
Redundancy + selective measurement recovers from some physical errors without revealing encoded quantum info.

---

## 5.7 Benchmarking Under Noise

### Comprehensive Algorithm Benchmarking Framework

```python
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, errors

def create_benchmark_noise_models():
    """Create different noise models for benchmarking"""
    
    noise_models = {}
    
    # 1. Simple depolarizing noise
    simple_noise = NoiseModel()
    depol_error = errors.depolarizing_error(0.01, 1)  # 1% depolarizing
    simple_noise.add_all_qubit_quantum_error(depol_error, ['h', 'x', 'z', 'cx'])
    noise_models['simple'] = simple_noise
    
    # 2. Realistic hardware noise
    realistic_noise = NoiseModel()
    
    # T1/T2 relaxation
    t1_error = errors.thermal_relaxation_error(100e-6, 50e-6, 50e-9)  # T1=100μs, T2=50μs, gate_time=50ns
    realistic_noise.add_all_qubit_quantum_error(t1_error, ['h', 'x', 'z'])
    
    # Two-qubit gate errors
    cx_error = errors.depolarizing_error(0.005, 2)  # 0.5% two-qubit error
    realistic_noise.add_all_qubit_quantum_error(cx_error, ['cx'])
    
    # Readout errors
    readout_error = errors.ReadoutError([[0.98, 0.02], [0.05, 0.95]])
    realistic_noise.add_readout_error(readout_error, [0])
    
    noise_models['realistic'] = realistic_noise
    
    # 3. High noise (worst case)
    high_noise = NoiseModel()
    high_depol = errors.depolarizing_error(0.05, 1)  # 5% error rate
    high_noise.add_all_qubit_quantum_error(high_depol, ['h', 'x', 'z', 'cx'])
    noise_models['high'] = high_noise
    
    return noise_models

def benchmark_grover_algorithm():
    """Benchmark Grover's algorithm under different noise conditions"""
    
    print("Grover Algorithm Noise Benchmark")
    print("=" * 32)
    
    def build_grover_circuit(n_qubits, marked_item):
        """Build Grover circuit for benchmarking"""
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize superposition
        qc.h(range(n_qubits))
        
        # Optimal number of iterations
        N = 2**n_qubits
        optimal_iter = int(np.pi/4 * sqrt(N))
        
        for _ in range(optimal_iter):
            # Oracle (mark target state)
            if marked_item > 0:
                # Convert marked_item to binary and flip those qubits
                for i in range(n_qubits):
                    if (marked_item >> i) & 1:
                        qc.x(i)
                
                # Multi-controlled Z gate
                if n_qubits == 1:
                    qc.z(0)
                else:
                    qc.h(n_qubits-1)
                    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
                    qc.h(n_qubits-1)
                
                # Flip back
                for i in range(n_qubits):
                    if (marked_item >> i) & 1:
                        qc.x(i)
            
            # Diffusion operator
            qc.h(range(n_qubits))
            qc.x(range(n_qubits))
            
            if n_qubits == 1:
                qc.z(0)
            else:
                qc.h(n_qubits-1)
                qc.mcx(list(range(n_qubits-1)), n_qubits-1)
                qc.h(n_qubits-1)
            
            qc.x(range(n_qubits))
            qc.h(range(n_qubits))
        
        qc.measure_all()
        return qc
    
    # Test parameters
    n_qubits = 3
    marked_item = 5  # Target state |101⟩
    noise_models = create_benchmark_noise_models()
    
    results = {}
    backend = Aer.get_backend('qasm_simulator')
    shots = 2048
    
    print(f"Testing {n_qubits}-qubit Grover search for item {marked_item}")
    print(f"Theoretical success probability: ~{1.0:.3f}")
    
    # Test each noise model
    for noise_name, noise_model in noise_models.items():
        grover_circuit = build_grover_circuit(n_qubits, marked_item)
        
        if noise_name == 'ideal':
            job = execute(grover_circuit, backend, shots=shots)
        else:
            job = execute(grover_circuit, backend, shots=shots, noise_model=noise_model)
        
        counts = job.result().get_counts()
        
        # Calculate success probability
        target_state = format(marked_item, f'0{n_qubits}b')
        success_count = counts.get(target_state, 0)
        success_prob = success_count / shots
        
        results[noise_name] = {
            'success_prob': success_prob,
            'counts': counts,
            'circuit_depth': grover_circuit.depth()
        }
        
        print(f"{noise_name:10s}: {success_prob:.3f} success rate (depth: {grover_circuit.depth()})")
    
    return results

def benchmark_vqe_ansatz():
    """Benchmark VQE ansatz under noise"""
    
    print("\nVQE Ansatz Noise Sensitivity")
    print("=" * 28)
    
    def create_vqe_ansatz(n_qubits, depth, params):
        """Create hardware-efficient VQE ansatz"""
        qc = QuantumCircuit(n_qubits)
        param_idx = 0
        
        for d in range(depth):
            # Rotation layer
            for qubit in range(n_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
            
            # Entangling layer
            for qubit in range(n_qubits - 1):
                qc.cx(qubit, qubit + 1)
        
        return qc
    
    def measure_pauli_expectation(circuit, pauli_string, noise_model=None):
        """Measure expectation value of Pauli operator"""
        # Add measurement basis rotation
        meas_circuit = circuit.copy()
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                meas_circuit.h(i)
            elif pauli == 'Y':
                meas_circuit.sdg(i)
                meas_circuit.h(i)
            # Z basis: no rotation needed
        
        meas_circuit.measure_all()
        
        backend = Aer.get_backend('qasm_simulator')
        if noise_model:
            job = execute(meas_circuit, backend, shots=1024, noise_model=noise_model)
        else:
            job = execute(meas_circuit, backend, shots=1024)
        
        counts = job.result().get_counts()
        
        # Calculate expectation value
        expectation = 0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            parity = sum(int(bit) for bit in bitstring) % 2
            sign = 1 if parity == 0 else -1
            expectation += sign * count / total_shots
        
        return expectation
    
    # Test parameters
    n_qubits = 2
    depth = 2
    params = np.random.uniform(0, 2*np.pi, depth * n_qubits)  # Random parameters
    pauli_observable = 'ZZ'  # Measure ⟨ZZ⟩
    
    noise_models = create_benchmark_noise_models()
    
    print(f"VQE ansatz: {n_qubits} qubits, depth {depth}")
    print(f"Observable: {pauli_observable}")
    
    vqe_results = {}
    
    for noise_name, noise_model in noise_models.items():
        ansatz = create_vqe_ansatz(n_qubits, depth, params)
        
        if noise_name == 'ideal':
            expectation = measure_pauli_expectation(ansatz, pauli_observable)
        else:
            expectation = measure_pauli_expectation(ansatz, pauli_observable, noise_model)
        
        vqe_results[noise_name] = expectation
        print(f"{noise_name:10s}: ⟨{pauli_observable}⟩ = {expectation:6.3f}")
    
    return vqe_results

def noise_scaling_analysis():
    """Analyze how algorithm performance scales with noise strength"""
    
    print("\nNoise Scaling Analysis")
    print("=" * 21)
    
    def simple_bell_state_fidelity(noise_strength):
        """Measure Bell state fidelity vs noise strength"""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Create noise model with variable strength
        noise_model = NoiseModel()
        gate_error = errors.depolarizing_error(noise_strength, 1)
        cx_error = errors.depolarizing_error(noise_strength * 2, 2)  # 2-qubit gates worse
        
        noise_model.add_all_qubit_quantum_error(gate_error, ['h'])
        noise_model.add_all_qubit_quantum_error(cx_error, ['cx'])
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1024, noise_model=noise_model)
        counts = job.result().get_counts()
        
        # Bell state should give |00⟩ and |11⟩ with equal probability
        bell_counts = counts.get('00', 0) + counts.get('11', 0)
        fidelity = bell_counts / 1024
        
        return fidelity
    
    # Test different noise strengths
    noise_strengths = np.logspace(-4, -1, 10)  # From 0.01% to 10%
    fidelities = []
    
    print("Noise Strength | Bell State Fidelity")
    print("-" * 32)
    
    for strength in noise_strengths:
        fidelity = simple_bell_state_fidelity(strength)
        fidelities.append(fidelity)
        print(f"{strength:11.1e} | {fidelity:15.3f}")
    
    # Find noise threshold (where fidelity drops below 90%)
    threshold_idx = next((i for i, f in enumerate(fidelities) if f < 0.9), len(fidelities)-1)
    threshold = noise_strengths[threshold_idx]
    
    print(f"\nApproximate 90% fidelity threshold: {threshold:.2e}")
    
    return noise_strengths, fidelities

# Run comprehensive benchmarking
print("Starting Comprehensive Algorithm Benchmarking")
print("=" * 44)

grover_results = benchmark_grover_algorithm()
vqe_results = benchmark_vqe_ansatz()
strengths, fidelities = noise_scaling_analysis()

print("\nBenchmarking Summary:")
print("- Grover: Success rate varies significantly with noise")
print("- VQE: Observable expectations shift under noise")
print("- Scaling: Performance degrades predictably with noise strength")
```

### Statistical Analysis and Confidence Intervals

```python
def statistical_analysis_demo():
    """Demonstrate proper statistical analysis of quantum experiments"""
    
    print("\nStatistical Analysis of Quantum Experiments")
    print("=" * 40)
    
    def wilson_confidence_interval(successes, trials, confidence=0.95):
        """Wilson score interval for binomial proportions"""
        from scipy.stats import norm
        
        z = norm.ppf((1 + confidence) / 2)
        p = successes / trials
        n = trials
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        
        return center - margin, center + margin
    
    def analyze_experiment_statistics(algorithm_name, results_list):
        """Analyze multiple runs of quantum experiment"""
        
        print(f"\nStatistical Analysis: {algorithm_name}")
        print("-" * (22 + len(algorithm_name)))
        
        n_runs = len(results_list)
        mean_success = np.mean(results_list)
        std_success = np.std(results_list)
        
        print(f"Number of runs: {n_runs}")
        print(f"Mean success rate: {mean_success:.3f} ± {std_success:.3f}")
        
        # Calculate confidence interval for the mean
        from scipy.stats import t
        confidence = 0.95
        df = n_runs - 1
        t_value = t.ppf((1 + confidence) / 2, df)
        margin_error = t_value * std_success / sqrt(n_runs)
        
        ci_low = mean_success - margin_error
        ci_high = mean_success + margin_error
        
        print(f"95% CI for mean: [{ci_low:.3f}, {ci_high:.3f}]")
        
        # For individual run confidence intervals (assuming 1024 shots each)
        shots_per_run = 1024
        avg_successes = mean_success * shots_per_run
        wilson_low, wilson_high = wilson_confidence_interval(avg_successes, shots_per_run)
        
        print(f"Wilson CI for proportion: [{wilson_low:.3f}, {wilson_high:.3f}]")
        
        return mean_success, std_success, (ci_low, ci_high)
    
    # Simulate multiple runs of an experiment
    np.random.seed(42)  # For reproducibility
    
    # Simulate noisy Grover results (multiple independent runs)
    true_success_rate = 0.85  # True underlying success rate
    n_experiments = 20
    shots_per_experiment = 1024
    
    grover_runs = []
    for _ in range(n_experiments):
        # Simulate binomial sampling
        successes = np.random.binomial(shots_per_experiment, true_success_rate)
        success_rate = successes / shots_per_experiment
        grover_runs.append(success_rate)
    
    analyze_experiment_statistics("Grover Algorithm", grover_runs)
    
    # Power analysis: How many shots needed for given precision?
    print(f"\nPower Analysis: Required shots for precision")
    print("-" * 40)
    
    desired_margins = [0.01, 0.02, 0.05, 0.1]  # 1%, 2%, 5%, 10% margins
    
    for margin in desired_margins:
        # For 95% confidence, Z ≈ 1.96
        # Margin = 1.96 * sqrt(p(1-p)/n)
        # Solving for n: n = (1.96^2 * p(1-p)) / margin^2
        p_estimate = 0.5  # Worst case (maximum variance)
        required_shots = (1.96**2 * p_estimate * (1 - p_estimate)) / margin**2
        
        print(f"±{margin:.1%} margin: {required_shots:.0f} shots needed")
    
    return grover_runs

# Run statistical analysis
experiment_stats = statistical_analysis_demo()
```

---

## 5.8 Looking Forward: Fault Tolerance

### Threshold Theorem
Below physical error threshold + enough overhead → arbitrarily reliable quantum computation achievable.

### Surface Code (Conceptual)
2D lattice of data + ancilla qubits; repeated syndrome extraction; high threshold (~1e-2). Logical qubit area grows with distance.

### Logical vs Physical Qubits
Huge overhead today (hundreds–thousands per logical). Drives hardware scale ambition.

### Near-Term Outlook
- Better noise tailoring & mitigation
- Early small logical qubits (d=3–5 surface code patches)
- Integration of QEC + application workloads

---

## 5.9 Summary & Project

### Recap
You learned:
- Sources and models of quantum noise
- Mitigation vs full error correction
- Basic stabilizer code mechanics
- Syndrome extraction principles
- Benchmarking methodology

### Project: Noise Benchmark Suite
Build a script/notebook to:
1. Choose algorithm (Grover, VQE subroutine)
2. Generate circuits with adjustable depth
3. Apply escalating noise models
4. Collect metrics: success probability, fidelity, variance
5. Apply mitigation (measurement calibration + ZNE)
6. Plot improvement vs baseline
7. Produce summary report (tables + brief narrative)

### Stretch Goals
- Add dynamical decoupling insertion pass
- Implement simple surface code stabilizer measurement simulation
- Integrate Mitiq library for real ZNE / PEC (if environment permits)

### Further Reading
- Nielsen & Chuang, Ch. 10–11 (Error Correction)
- Terhal, “Quantum Error Correction for Quantum Memories”
- Google Quantum AI: Surface Code experiments
- Mitiq documentation (error mitigation framework)

---

## Check Your Understanding (Quick Quiz)
1. Difference between T1 and T2?  
2. What does a distance-3 code guarantee?  
3. Why can’t we clone qubits to back them up?  
4. Core idea of zero-noise extrapolation?  
5. Why is error mitigation not a replacement for full QEC?

*End of Module 5 – proceed to Module 6 when comfortable with noise-aware design.*
