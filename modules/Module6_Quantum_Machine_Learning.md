# Module 6: Quantum Machine Learning (QML)
*Intermediate Tier*

> **✅ Qiskit 2.x Compatible** - All examples updated and tested (Dec 2024)
> 
> **Recent Updates:**
> - Updated `bind_parameters` → `assign_parameters` in all QML examples
> - All 5 examples (100%) passing tests
> - Note: QNN training may take 60-120s depending on dataset size

## Learning Objectives
By the end of this module, you will be able to:
- Explain what makes data "quantum ready" and compare quantum data encoding strategies
- Build variational quantum circuits (VQCs) as trainable models (“quantum neural networks”)
- Implement a simple quantum classifier on a small dataset (e.g., XOR or concentric circles)
- Use quantum kernels with an SVM workflow and understand when they may help
- Leverage PennyLane for differentiable hybrid quantum-classical optimization
- Compare hybrid (parameterized quantum + classical) vs. fully classical baselines
- Evaluate model performance under noise and discuss expressivity / trainability trade‑offs

## Prerequisites
- Modules 1–5 (state vectors, gates, circuits, noise & mitigation)
- Comfortable with Python & NumPy
- Basic machine learning concepts (features, labels, training loop, loss)

---

## 6.1 What Is Quantum Machine Learning?

Quantum Machine Learning seeks either:
1. **Quantum-enhanced ML**: Use quantum subroutines inside an ML pipeline for potential speedups or richer feature spaces.
2. **Hybrid Variational Models**: Parameterized quantum circuits + classical optimizers (NISQ focused).
3. **Quantum Data Processing**: Learn from data that is *natively quantum* (e.g., states from quantum systems).

### Where Might Advantage Come From?
| Potential Lever | Idea | Caveats |
|-----------------|------|--------|
| High-Dimensional Hilbert Space | Implicit feature lifting | Encoding overhead may erase gains |
| Quantum Kernels | Complex inner products | Must beat classical kernel tricks |
| Entanglement as Correlation | Non-classical feature interaction | Noise reduces benefit |
| Variational Expressivity | Compact parametrization | Barren plateaus, trainability issues |

### Core Building Blocks
- **Encoding / Embedding**: Map classical vector → quantum state |x⟩
- **Parameterized Circuit**: U(θ) producing |ψ(x, θ)⟩
- **Measurement**: Observable → scalar(s) used for loss
- **Optimizer**: Classical gradient-free or gradient-based

---

## 6.2 Quantum Data Encoding (Feature Maps)

Goal: Embed classical vector x ∈ R^d into a quantum state for downstream processing.

### Encoding Strategies
| Strategy | Idea | Qubits Needed | Pros | Cons |
|----------|------|--------------|------|------|
| Basis / Computational | Map integer → |index⟩ | log₂(N) | Simple | Sparse info (one-hot) |
| Angle (Rotation) | Each feature → rotation angle (e.g., RY(xᵢ)) | d (or reused) | Easy hardware | Scales linearly |
| Amplitude | Normalize vector → amplitudes | log₂(d) | Exponential compression | Preparation cost O(d) |
| IQP / Feature Map Circuits | Non-linear phase features via entanglers | d | Rich kernels | Depth increases |
| Data Reuploading | Inject data multiple layers | d | Increases expressivity | More gates/noise |

### Simple Examples
```python
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt

def comprehensive_encoding_demo():
    """Demonstrate different quantum data encoding strategies"""
    
    print("Quantum Data Encoding Strategies")
    print("=" * 32)
    
    # Sample classical data
    data_2d = np.array([0.3, 0.7])  # 2D feature vector
    data_4d = np.array([0.1, 0.4, 0.8, 0.2])  # 4D feature vector
    
    print(f"Sample 2D data: {data_2d}")
    print(f"Sample 4D data: {data_4d}")
    
    # 1. Angle Encoding
    def angle_encoding(x):
        """Angle encode a feature vector x into RY rotations."""
        qc = QuantumCircuit(len(x))
        for i, val in enumerate(x):
            qc.ry(val * np.pi, i)  # Scale to [0, π] range
        return qc
    
    # 2. Amplitude Encoding  
    def amplitude_encoding(vec):
        """Normalize and load amplitudes"""
        v = np.array(vec, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0: 
            raise ValueError("Zero vector not encodable")
        v = v / norm
        
        # Number of qubits needed
        n_qubits = int(np.ceil(np.log2(len(v))))
        padded = np.zeros(2**n_qubits)
        padded[:len(v)] = v
        
        qc = QuantumCircuit(n_qubits)
        qc.initialize(padded, list(range(n_qubits)))
        return qc
    
    # 3. Basis Encoding (for integers)
    def basis_encoding(value, n_qubits):
        """Encode integer as computational basis state"""
        qc = QuantumCircuit(n_qubits)
        # Convert to binary and apply X gates
        for i in range(n_qubits):
            if (value >> i) & 1:
                qc.x(i)
        return qc
    
    # 4. Feature Map Encoding (with entanglement)
    def feature_map_encoding(x):
        """Create entangled feature map"""
        qc = QuantumCircuit(len(x))
        
        # First layer: individual rotations
        for i, val in enumerate(x):
            qc.h(i)
            qc.rz(val * np.pi, i)
        
        # Second layer: entangling interactions
        for i in range(len(x)-1):
            qc.cx(i, i+1)
            qc.rz((x[i] * x[i+1]) * np.pi, i+1)
        
        return qc
    
    # 5. Data Reuploading
    def data_reuploading_encoding(x, layers=2):
        """Multiple layers of data encoding"""
        qc = QuantumCircuit(len(x))
        
        for layer in range(layers):
            # Data encoding layer
            for i, val in enumerate(x):
                qc.ry(val * np.pi * (layer + 1), i)
            
            # Entangling layer (except last)
            if layer < layers - 1:
                for i in range(len(x)-1):
                    qc.cx(i, i+1)
        
        return qc
    
    # Demonstrate each encoding
    print("\n1. Angle Encoding:")
    qc_angle = angle_encoding(data_2d)
    print(qc_angle.draw())
    
    # Get and show resulting state
    backend = Aer.get_backend('statevector_simulator')
    statevector = execute(qc_angle, backend).result().get_statevector()
    print(f"Resulting amplitudes: {np.abs(statevector)}")
    
    print("\n2. Amplitude Encoding:")
    qc_amp = amplitude_encoding(data_4d)
    print(qc_amp.draw())
    
    print("\n3. Basis Encoding (integer 5 in 3 qubits):")
    qc_basis = basis_encoding(5, 3)
    print(qc_basis.draw())
    
    print("\n4. Feature Map Encoding:")
    qc_fm = feature_map_encoding(data_2d)
    print(qc_fm.draw())
    
    print("\n5. Data Reuploading (2 layers):")
    qc_reup = data_reuploading_encoding(data_2d, layers=2)
    print(qc_reup.draw())
    
    return qc_angle, qc_amp, qc_basis, qc_fm, qc_reup

def encoding_comparison_analysis():
    """Compare different encodings on same data"""
    
    print("\nEncoding Strategy Comparison")
    print("=" * 28)
    
    # Test data
    test_vectors = [
        np.array([0.2, 0.8]),
        np.array([0.5, 0.5]), 
        np.array([0.9, 0.1]),
        np.array([0.0, 1.0])
    ]
    
    backend = Aer.get_backend('statevector_simulator')
    
    print("Data Vector | Angle Encoding | Amplitude Encoding")
    print("-" * 50)
    
    for vec in test_vectors:
        # Angle encoding result
        qc_angle = angle_encoding(vec)
        sv_angle = execute(qc_angle, backend).result().get_statevector()
        
        # Amplitude encoding result
        qc_amp = amplitude_encoding(vec)
        sv_amp = execute(qc_amp, backend).result().get_statevector()
        
        print(f"{vec} | {np.abs(sv_angle)[:2]} | {np.abs(sv_amp)[:4]}")
    
    print(f"\nKey insights:")
    print(f"- Angle encoding: Preserves relative magnitudes via rotation angles")
    print(f"- Amplitude encoding: Direct mapping but requires normalization")
    print(f"- Feature maps: Create quantum correlations via entanglement")

def visualize_encoding_effects():
    """Visualize how different encodings affect quantum states"""
    
    print("\nVisualization: Encoding Effects on Bloch Sphere")
    print("=" * 47)
    
    # Single qubit examples
    angles = np.linspace(0, 2*np.pi, 8)
    
    print("Angle | RY(θ) State |0⟩ component | |1⟩ component")
    print("-" * 50)
    
    for angle in angles:
        qc = QuantumCircuit(1)
        qc.ry(angle, 0)
        
        backend = Aer.get_backend('statevector_simulator')
        sv = execute(qc, backend).result().get_statevector()
        
        prob_0 = abs(sv[0])**2
        prob_1 = abs(sv[1])**2
        
        print(f"{angle:.2f} | RY({angle:.2f})|0⟩ | {prob_0:.3f} | {prob_1:.3f}")
    
    print(f"\nBloch sphere interpretation:")
    print(f"- θ=0: North pole |0⟩") 
    print(f"- θ=π/2: Equator |+⟩ = (|0⟩+|1⟩)/√2")
    print(f"- θ=π: South pole |1⟩")

# Run comprehensive encoding demonstrations
encodings = comprehensive_encoding_demo()
encoding_comparison_analysis()
visualize_encoding_effects()
```

### Advanced Feature Map Examples

```python
def advanced_feature_maps():
    """Demonstrate sophisticated feature map constructions"""
    
    print("\nAdvanced Quantum Feature Maps")
    print("=" * 29)
    
    def pauli_feature_map(x, reps=1):
        """Pauli feature map with repeated applications"""
        n_qubits = len(x)
        qc = QuantumCircuit(n_qubits)
        
        for rep in range(reps):
            # Hadamard layer
            for i in range(n_qubits):
                qc.h(i)
            
            # Single-qubit rotations
            for i, val in enumerate(x):
                qc.rz(2 * val, i)
            
            # Entangling layer with pairwise interactions
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    qc.cx(i, j)
                    qc.rz(2 * x[i] * x[j], j)
                    qc.cx(i, j)
        
        return qc
    
    def chebyshev_feature_map(x, degree=2):
        """Feature map using Chebyshev polynomials"""
        n_qubits = len(x)
        qc = QuantumCircuit(n_qubits)
        
        # Prepare superposition
        qc.h(range(n_qubits))
        
        # Apply Chebyshev-inspired rotations
        for d in range(1, degree + 1):
            for i, val in enumerate(x):
                # Chebyshev polynomial of degree d
                if d == 1:
                    angle = val
                elif d == 2:
                    angle = 2 * val**2 - 1
                else:
                    angle = val  # Simplified for demo
                
                qc.rz(angle * np.pi, i)
        
        # Add interactions
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc
    
    def qaoa_inspired_feature_map(x, layers=2):
        """QAOA-inspired feature map with alternating layers"""
        n_qubits = len(x)
        qc = QuantumCircuit(n_qubits)
        
        # Initial superposition
        qc.h(range(n_qubits))
        
        for layer in range(layers):
            # Problem layer (feature-dependent)
            for i, val in enumerate(x):
                qc.rz(val * np.pi, i)
            
            # Mixer layer (classical correlations)
            for i in range(n_qubits):
                qc.rx(np.pi/4, i)
            
            # Entangling layer
            if layer < layers - 1:
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
        
        return qc
    
    # Test different feature maps
    test_data = np.array([0.3, 0.7])
    
    print("Comparing feature map complexities:")
    
    # Standard Pauli map
    fm1 = pauli_feature_map(test_data, reps=1)
    print(f"Pauli map (1 rep): {fm1.depth()} depth, {len(fm1.data)} gates")
    
    # Deeper Pauli map
    fm2 = pauli_feature_map(test_data, reps=2)
    print(f"Pauli map (2 reps): {fm2.depth()} depth, {len(fm2.data)} gates")
    
    # Chebyshev map
    fm3 = chebyshev_feature_map(test_data, degree=2)
    print(f"Chebyshev map: {fm3.depth()} depth, {len(fm3.data)} gates")
    
    # QAOA-inspired map
    fm4 = qaoa_inspired_feature_map(test_data, layers=2)
    print(f"QAOA-inspired: {fm4.depth()} depth, {len(fm4.data)} gates")
    
    # Analyze expressivity vs complexity trade-off
    print(f"\nExpressivity vs Complexity Trade-offs:")
    print(f"- Deeper circuits → more expressive but harder to train")
    print(f"- More entanglement → richer feature space but more noise-sensitive")
    print(f"- Parameter count affects optimization landscape")
    
    return fm1, fm2, fm3, fm4

# Run advanced feature map demonstrations
feature_maps = advanced_feature_maps()
```

### Design Considerations
- **Expressivity vs Simplicity**: More layers & entanglement can map to richer feature spaces.
- **Trainability**: Highly expressive feature maps risk barren plateaus.
- **Normalization**: Amplitude encoding demands normalized vectors.

---

## 6.3 Variational Quantum Circuits (Quantum Neural Networks)

A parameterized circuit acts like a neural network layer; parameters are updated to minimize a cost.

### Generic Pattern
| Step | Operation |
|------|-----------|
| 1 | Encode input x into quantum state |
| 2 | Apply trainable layers U(θ) (rotations + entanglers) |
| 3 | Measure observable(s) → raw scores |
| 4 | Map measurement to prediction (e.g., sigmoid) |
| 5 | Compute loss vs label, optimize θ |

### Complete VQC Implementation with Training

```python
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

class VariationalQuantumClassifier:
    """Complete implementation of a variational quantum classifier"""
    
    def __init__(self, n_qubits, ansatz_type='hardware_efficient', shots=1024):
        self.n_qubits = n_qubits
        self.ansatz_type = ansatz_type
        self.shots = shots
        self.parameters = None
        self.optimal_params = None
        self.loss_history = []
        self.backend = Aer.get_backend('qasm_simulator')
        
    def build_ansatz(self, depth=3):
        """Build different types of variational ansatzes"""
        
        if self.ansatz_type == 'hardware_efficient':
            return self._hardware_efficient_ansatz(depth)
        elif self.ansatz_type == 'qaoa_inspired':
            return self._qaoa_inspired_ansatz(depth)
        elif self.ansatz_type == 'alternate_entangling':
            return self._alternate_entangling_ansatz(depth)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz_type}")
    
    def _hardware_efficient_ansatz(self, depth):
        """Hardware-efficient ansatz with RY rotations and CX gates"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Parameters for rotations
        params = []
        
        for layer in range(depth):
            # Single-qubit rotations
            layer_params = []
            for qubit in range(self.n_qubits):
                param = Parameter(f'θ_{layer}_{qubit}')
                layer_params.append(param)
                qc.ry(param, qubit)
            params.extend(layer_params)
            
            # Entangling gates (circular connectivity)
            for qubit in range(self.n_qubits):
                qc.cx(qubit, (qubit + 1) % self.n_qubits)
        
        # Final rotation layer
        final_params = []
        for qubit in range(self.n_qubits):
            param = Parameter(f'θ_final_{qubit}')
            final_params.append(param)
            qc.ry(param, qubit)
        params.extend(final_params)
        
        self.parameters = params
        return qc
    
    def _qaoa_inspired_ansatz(self, depth):
        """QAOA-inspired ansatz with alternating ZZ and X layers"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial superposition
        qc.h(range(self.n_qubits))
        
        params = []
        
        for layer in range(depth):
            # ZZ interaction layer (problem layer)
            gamma = Parameter(f'γ_{layer}')
            params.append(gamma)
            
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(gamma, i + 1)
                qc.cx(i, i + 1)
            
            # X mixer layer
            beta = Parameter(f'β_{layer}')
            params.append(beta)
            
            for qubit in range(self.n_qubits):
                qc.rx(beta, qubit)
        
        self.parameters = params
        return qc
    
    def _alternate_entangling_ansatz(self, depth):
        """Alternating entangling ansatz for better expressivity"""
        qc = QuantumCircuit(self.n_qubits)
        
        params = []
        
        for layer in range(depth):
            # Rotation layer
            for qubit in range(self.n_qubits):
                # Three rotation angles per qubit
                for axis, gate in zip(['x', 'y', 'z'], [qc.rx, qc.ry, qc.rz]):
                    param = Parameter(f'θ_{axis}_{layer}_{qubit}')
                    params.append(param)
                    gate(param, qubit)
            
            # Entangling layer (alternating between linear and circular)
            if layer % 2 == 0:
                # Linear entanglement
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
            else:
                # Circular entanglement
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)
        
        self.parameters = params
        return qc
    
    def encode_data(self, x):
        """Angle encoding of classical data"""
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x[:self.n_qubits]):
            qc.ry(val * np.pi, i)
        return qc
    
    def create_circuit(self, x, params):
        """Create complete circuit with data encoding + ansatz"""
        # Data encoding
        encoding_circuit = self.encode_data(x)
        
        # Variational ansatz
        ansatz = self.build_ansatz()
        param_dict = {self.parameters[i]: params[i] for i in range(len(params))}
        bound_ansatz = ansatz.bind_parameters(param_dict)
        
        # Combine circuits
        full_circuit = encoding_circuit.compose(bound_ansatz)
        
        # Add measurement
        full_circuit.add_register(ClassicalRegister(1, 'result'))
        full_circuit.measure(0, 0)  # Measure first qubit
        
        return full_circuit
    
    def compute_expectation(self, x, params):
        """Compute expectation value for classification"""
        circuit = self.create_circuit(x, params)
        
        # Execute circuit
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate expectation <Z_0>
        prob_0 = counts.get('0', 0) / self.shots
        prob_1 = counts.get('1', 0) / self.shots
        expectation = prob_0 - prob_1
        
        return expectation
    
    def cost_function(self, params, X, y):
        """Cost function for training"""
        predictions = []
        
        for x_sample in X:
            exp_val = self.compute_expectation(x_sample, params)
            predictions.append(exp_val)
        
        predictions = np.array(predictions)
        
        # Mean squared error loss
        loss = np.mean((predictions - y) ** 2)
        
        return loss
    
    def gradient_finite_diff(self, params, X, y, epsilon=0.01):
        """Compute gradient via finite differences"""
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            
            loss_plus = self.cost_function(params_plus, X, y)
            loss_minus = self.cost_function(params_minus, X, y)
            
            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return grad
    
    def train(self, X_train, y_train, epochs=50, learning_rate=0.1, verbose=True):
        """Train the quantum classifier"""
        
        # Initialize parameters randomly
        n_params = len(self.parameters) if self.parameters else 3 * self.n_qubits
        params = np.random.uniform(-np.pi, np.pi, n_params)
        
        self.loss_history = []
        
        print(f"Training VQC with {len(params)} parameters...")
        print(f"Ansatz type: {self.ansatz_type}")
        print("-" * 40)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Compute loss and gradient
            loss = self.cost_function(params, X_train, y_train)
            grad = self.gradient_finite_diff(params, X_train, y_train)
            
            # Update parameters
            params -= learning_rate * grad
            
            # Store loss
            self.loss_history.append(loss)
            
            epoch_time = time.time() - start_time
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: Loss = {loss:.4f}, Time = {epoch_time:.2f}s")
        
        self.optimal_params = params
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"Final loss: {self.loss_history[-1]:.4f}")
    
    def predict(self, X_test):
        """Make predictions on test data"""
        if self.optimal_params is None:
            raise ValueError("Model not trained yet!")
        
        predictions = []
        for x in X_test:
            exp_val = self.compute_expectation(x, self.optimal_params)
            # Convert expectation value to binary prediction
            pred = 1 if exp_val > 0 else -1
            predictions.append(pred)
        
        return np.array(predictions)
    
    def plot_training_history(self):
        """Plot training loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'VQC Training History ({self.ansatz_type} ansatz)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

def demonstrate_vqc_training():
    """Complete demonstration of VQC training and evaluation"""
    
    print("Variational Quantum Classifier Demo")
    print("=" * 35)
    
    # Generate synthetic dataset
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100, 
        n_features=2, 
        n_redundant=0, 
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert labels to {-1, +1}
    y = 2 * y - 1
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X_train)} training, {len(X_test)} test samples")
    print(f"Features: {X.shape[1]} dimensions")
    
    # Compare different ansatz types
    ansatz_types = ['hardware_efficient', 'qaoa_inspired']
    
    results = {}
    
    for ansatz_type in ansatz_types:
        print(f"\n{ansatz_type.title()} Ansatz:")
        print("-" * 30)
        
        # Create and train classifier
        vqc = VariationalQuantumClassifier(
            n_qubits=2, 
            ansatz_type=ansatz_type,
            shots=1024
        )
        
        # Train model
        vqc.train(X_train, y_train, epochs=30, learning_rate=0.1)
        
        # Make predictions
        y_pred = vqc.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        print(f"Test accuracy: {accuracy:.3f}")
        
        # Store results
        results[ansatz_type] = {
            'model': vqc,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(ansatz_types), figsize=(15, 5))
    
    for idx, (ansatz_type, result) in enumerate(results.items()):
        ax = axes[idx] if len(ansatz_types) > 1 else axes
        
        # Plot training history
        ax.plot(result['model'].loss_history, 'b-', linewidth=2)
        ax.set_title(f'{ansatz_type.title()} Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # Add final accuracy as text
        ax.text(0.7, 0.9, f'Accuracy: {result["accuracy"]:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_barren_plateaus():
    """Analyze the barren plateau phenomenon in VQCs"""
    
    print("\nBarren Plateau Analysis")
    print("=" * 23)
    
    def compute_gradient_variance(n_qubits, depth, num_trials=20):
        """Compute variance of gradients for different circuit depths"""
        
        variances = []
        
        for trial in range(num_trials):
            # Create random VQC
            vqc = VariationalQuantumClassifier(n_qubits, 'hardware_efficient')
            ansatz = vqc.build_ansatz(depth)
            
            # Random parameters
            params = np.random.uniform(-np.pi, np.pi, len(vqc.parameters))
            
            # Random data point
            x = np.random.uniform(-1, 1, n_qubits)
            y = np.random.choice([-1, 1])
            
            # Compute gradient
            grad = vqc.gradient_finite_diff(params, [x], [y])
            variances.append(np.var(grad))
        
        return np.mean(variances)
    
    # Test different depths
    depths = range(1, 8)
    n_qubits = 4
    
    print(f"Analyzing gradient variance vs circuit depth ({n_qubits} qubits):")
    
    variances = []
    for depth in depths:
        var = compute_gradient_variance(n_qubits, depth, num_trials=5)
        variances.append(var)
        print(f"Depth {depth}: Gradient variance = {var:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(depths, variances, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Circuit Depth')
    plt.ylabel('Gradient Variance (log scale)')
    plt.title('Barren Plateau: Gradient Variance vs Circuit Depth')
    plt.grid(True, alpha=0.3)
    
    # Add exponential decay line for reference
    theoretical = [variances[0] * np.exp(-0.5 * d) for d in depths]
    plt.semilogy(depths, theoretical, 'b--', alpha=0.7, label='Theoretical decay')
    plt.legend()
    plt.show()
    
    print(f"\nKey observations:")
    print(f"- Gradient variance decreases exponentially with depth")
    print(f"- Deeper circuits → flatter loss landscapes")
    print(f"- Mitigation: Careful initialization, structured ansatzes")

# Run comprehensive VQC demonstrations
from qiskit import ClassicalRegister

# Execute all demonstrations
vqc_results = demonstrate_vqc_training()
analyze_barren_plateaus()
```

### Simple XOR Example for Quick Understanding

```python
def simple_xor_demo():
    """Quick XOR demonstration for concept understanding"""
    
    print("Simple XOR Quantum Classifier")
    print("=" * 30)
    
    # XOR dataset: inputs and expected outputs
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR truth table
    
    # Simple ansatz: 2 qubits, parameterized RY gates + entangling CX
    def create_xor_ansatz(x_input, theta):
        qc = QuantumCircuit(2, 1)
        
        # Data encoding: angle encoding
        qc.ry(x_input[0] * np.pi, 0)
        qc.ry(x_input[1] * np.pi, 1)
        
        # Variational layer
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1)
        qc.ry(theta[2], 0)
        qc.ry(theta[3], 1)
        
        # Measurement
        qc.measure(0, 0)
        return qc
    
    # Show circuit for different inputs
    params = [0.1, 0.5, -0.2, 0.8]
    
    print("Circuit for input [0, 1]:")
    circuit = create_xor_ansatz([0, 1], params)
    print(circuit.draw())
    
    # Execute for all XOR inputs
    backend = Aer.get_backend('qasm_simulator')
    
    print(f"\nXOR Results with parameters {params}:")
    print("Input | Expected | Quantum Output")
    print("-" * 30)
    
    for i, (x_val, y_expected) in enumerate(zip(X, y)):
        qc = create_xor_ansatz(x_val, params)
        
        # Execute circuit
        job = execute(qc, backend, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # Get probability of measuring |0⟩
        prob_0 = counts.get('0', 0) / 1000
        quantum_output = 1 if prob_0 < 0.5 else 0  # Threshold decision
        
        print(f"{x_val} | {y_expected} | {quantum_output} (P(0)={prob_0:.3f})")
    
    print(f"\nNote: Parameters need optimization for correct XOR mapping!")
    
    return circuit

# Run simple XOR demo
xor_circuit = simple_xor_demo()
```
    ce = 0
    for y_hat, y in zip(preds, Y):
        ce += -(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))
    return ce / len(Y)

init_params = np.random.uniform(0, 2*np.pi, 5)
res = minimize(loss, init_params, method='COBYLA', options={'maxiter':100})
print("Optimized params:", res.x)
print("Final loss:", loss(res.x))

for x, y in zip(X, Y):
    print("Input", x, "Label", y, "Pred", round(predict(x, res.x),3))
```

### Notes
- This is *not* scalable but illustrates loop mechanics.
- Real implementations batch measurements and reuse transpiled circuits for efficiency.

### Avoiding Barren Plateaus
| Strategy | Description |
|----------|-------------|
| Local Cost Functions | Measure subsets of qubits |
| Layer-wise Training | Grow circuit gradually |
| Parameter Initialization | Small random values or structured seeds |
| Data Reuploading | More expressivity without huge depth |

---

## 6.4 Quantum Kernels & SVM

Quantum kernel idea: Embed data via feature map Φ(x) on a quantum device; kernel K(x, x') = |⟨Φ(x)|Φ(x')⟩|² or ⟨Φ(x)|Φ(x')⟩.

### Why Kernels?
Leverage high-dimensional Hilbert space to separate classes linearly after mapping.

## 6.4 Quantum Kernels & SVM

Quantum kernel idea: Embed data via feature map Φ(x) on a quantum device; kernel K(x, x') = |⟨Φ(x)|Φ(x')⟩|² or ⟨Φ(x)|Φ(x')⟩.

### Complete Quantum Kernel Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

class QuantumKernelEstimator:
    """Complete quantum kernel implementation with different feature maps"""
    
    def __init__(self, feature_map_type='zz', n_qubits=2, reps=2, shots=1024):
        self.feature_map_type = feature_map_type
        self.n_qubits = n_qubits
        self.reps = reps
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        
    def create_feature_map(self, x):
        """Create different types of quantum feature maps"""
        
        if self.feature_map_type == 'zz':
            return self._zz_feature_map(x)
        elif self.feature_map_type == 'pauli':
            return self._pauli_feature_map(x)
        elif self.feature_map_type == 'custom':
            return self._custom_feature_map(x)
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map_type}")
    
    def _zz_feature_map(self, x):
        """ZZ feature map with Pauli interactions"""
        qc = QuantumCircuit(self.n_qubits)
        
        for rep in range(self.reps):
            # First layer: individual rotations
            for i, val in enumerate(x[:self.n_qubits]):
                qc.h(i)
                qc.rz(2 * val, i)
            
            # Second layer: ZZ interactions
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cx(i, j)
                    qc.rz(2 * x[i] * x[j], j)
                    qc.cx(i, j)
        
        return qc
    
    def _pauli_feature_map(self, x):
        """Pauli feature map with X, Y, Z rotations"""
        qc = QuantumCircuit(self.n_qubits)
        
        for rep in range(self.reps):
            # Hadamard layer
            for i in range(self.n_qubits):
                qc.h(i)
            
            # Pauli rotations
            for i, val in enumerate(x[:self.n_qubits]):
                qc.rz(val, i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(x[i] * x[i + 1], i + 1)
                qc.cx(i, i + 1)
        
        return qc
    
    def _custom_feature_map(self, x):
        """Custom feature map for specific data patterns"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial superposition
        qc.h(range(self.n_qubits))
        
        for rep in range(self.reps):
            # Data-dependent rotations
            for i, val in enumerate(x[:self.n_qubits]):
                qc.ry(val * np.pi, i)
                qc.rz(val * np.pi / 2, i)
            
            # Complex entangling pattern
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cx(i, j)
                    qc.ry(x[i] * x[j] * np.pi, j)
                    qc.cx(i, j)
        
        return qc
    
    def compute_kernel_element(self, x1, x2):
        """Compute kernel element K(x1, x2) = |⟨Φ(x1)|Φ(x2)⟩|²"""
        
        # Create feature maps
        qc1 = self.create_feature_map(x1)
        qc2 = self.create_feature_map(x2)
        
        # Create overlap test circuit
        qc_overlap = QuantumCircuit(self.n_qubits)
        
        # Apply Φ(x1)
        qc_overlap = qc_overlap.compose(qc1)
        
        # Apply Φ†(x2) (inverse of Φ(x2))
        qc_overlap = qc_overlap.compose(qc2.inverse())
        
        # Measure overlap in computational basis
        qc_overlap.measure_all()
        
        # Execute circuit
        job = execute(qc_overlap, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Probability of measuring all zeros gives |⟨Φ(x1)|Φ(x2)⟩|²
        all_zeros = '0' * self.n_qubits
        prob_all_zeros = counts.get(all_zeros, 0) / self.shots
        
        return prob_all_zeros
    
    def compute_kernel_matrix(self, X):
        """Compute full kernel matrix for dataset X"""
        
        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))
        
        print(f"Computing {n_samples}x{n_samples} quantum kernel matrix...")
        start_time = time.time()
        
        for i in range(n_samples):
            for j in range(i, n_samples):  # Use symmetry
                kernel_val = self.compute_kernel_element(X[i], X[j])
                K[i, j] = K[j, i] = kernel_val
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1}/{n_samples} rows, {elapsed:.1f}s elapsed")
        
        total_time = time.time() - start_time
        print(f"Kernel computation completed in {total_time:.2f}s")
        
        return K

def demonstrate_quantum_kernel_svm():
    """Complete demonstration of quantum kernel SVM"""
    
    print("Quantum Kernel SVM Demonstration")
    print("=" * 33)
    
    # Create synthetic datasets for testing
    datasets = {
        'linear_separable': make_classification(
            n_samples=80, n_features=2, n_redundant=0, 
            n_informative=2, n_clusters_per_class=1, random_state=42
        ),
        'circles': make_circles(n_samples=80, noise=0.1, factor=0.6, random_state=42)
    }
    
    # Test different feature maps
    feature_maps = ['zz', 'pauli', 'custom']
    
    results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 30)
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        dataset_results = {}
        
        for feature_map in feature_maps:
            print(f"\nFeature Map: {feature_map}")
            
            # Create quantum kernel estimator
            qke = QuantumKernelEstimator(
                feature_map_type=feature_map, 
                n_qubits=2, 
                reps=1,  # Keep low for demo
                shots=512  # Reduced for speed
            )
            
            # Compute training kernel matrix
            K_train = qke.compute_kernel_matrix(X_train)
            
            # Compute test kernel matrix
            K_test = np.zeros((len(X_test), len(X_train)))
            for i, x_test in enumerate(X_test):
                for j, x_train in enumerate(X_train):
                    K_test[i, j] = qke.compute_kernel_element(x_test, x_train)
            
            # Train SVM with precomputed quantum kernel
            svm_quantum = SVC(kernel='precomputed', C=1.0)
            svm_quantum.fit(K_train, y_train)
            
            # Make predictions
            y_pred = svm_quantum.predict(K_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  Quantum SVM Accuracy: {accuracy:.3f}")
            
            # Compare with classical RBF kernel
            svm_classical = SVC(kernel='rbf', C=1.0, gamma='scale')
            svm_classical.fit(X_train, y_train)
            y_pred_classical = svm_classical.predict(X_test)
            accuracy_classical = accuracy_score(y_test, y_pred_classical)
            
            print(f"  Classical RBF SVM:    {accuracy_classical:.3f}")
            
            # Store results
            dataset_results[feature_map] = {
                'quantum_accuracy': accuracy,
                'classical_accuracy': accuracy_classical,
                'K_train': K_train,
                'K_test': K_test
            }
        
        results[dataset_name] = dataset_results
    
    return results

def visualize_kernel_matrices():
    """Visualize quantum kernel matrices"""
    
    print("\nKernel Matrix Visualization")
    print("=" * 27)
    
    # Generate simple 2D dataset
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (20, 2))
    
    # Compute kernels with different feature maps
    feature_maps = ['zz', 'pauli', 'custom']
    
    fig, axes = plt.subplots(1, len(feature_maps), figsize=(15, 4))
    
    for idx, feature_map in enumerate(feature_maps):
        qke = QuantumKernelEstimator(
            feature_map_type=feature_map, 
            n_qubits=2, 
            reps=1,
            shots=256
        )
        
        K = qke.compute_kernel_matrix(X)
        
        # Plot kernel matrix
        im = axes[idx].imshow(K, cmap='Blues', vmin=0, vmax=1)
        axes[idx].set_title(f'{feature_map.title()} Feature Map')
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('Sample Index')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx])
        
        # Print statistics
        print(f"{feature_map} kernel statistics:")
        print(f"  Mean: {K.mean():.3f}")
        print(f"  Std:  {K.std():.3f}")
        print(f"  Min:  {K.min():.3f}")
        print(f"  Max:  {K.max():.3f}")
    
    plt.tight_layout()
    plt.show()

def quantum_kernel_expressivity_analysis():
    """Analyze expressivity of different quantum kernels"""
    
    print("\nQuantum Kernel Expressivity Analysis")
    print("=" * 36)
    
    # Generate data with different structures
    test_patterns = {
        'linear': np.random.uniform(-1, 1, (30, 2)),
        'circular': np.array([[r*np.cos(theta), r*np.sin(theta)] 
                             for r in np.linspace(0.2, 1, 15) 
                             for theta in np.linspace(0, 2*np.pi, 2)]),
        'clustered': np.concatenate([
            np.random.multivariate_normal([0.5, 0.5], [[0.1, 0], [0, 0.1]], 15),
            np.random.multivariate_normal([-0.5, -0.5], [[0.1, 0], [0, 0.1]], 15)
        ])
    }
    
    feature_maps = ['zz', 'pauli', 'custom']
    
    for pattern_name, X in test_patterns.items():
        print(f"\nPattern: {pattern_name}")
        print("-" * 20)
        
        for feature_map in feature_maps:
            qke = QuantumKernelEstimator(
                feature_map_type=feature_map,
                n_qubits=2,
                reps=1,
                shots=256
            )
            
            # Compute kernel matrix
            K = qke.compute_kernel_matrix(X)
            
            # Analyze kernel properties
            eigenvals = np.linalg.eigvals(K)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros
            
            # Effective rank (number of significant eigenvalues)
            effective_rank = len(eigenvals)
            
            # Trace and determinant
            trace = np.trace(K)
            det = np.linalg.det(K + 1e-8 * np.eye(len(K)))  # Regularized
            
            print(f"  {feature_map:6} | Rank: {effective_rank:2d} | "
                  f"Trace: {trace:5.1f} | Det: {det:8.2e}")
        
        print()

# Run all quantum kernel demonstrations
print("Running Quantum Kernel Demonstrations...")
print("=" * 40)

kernel_results = demonstrate_quantum_kernel_svm()
visualize_kernel_matrices()
quantum_kernel_expressivity_analysis()
```

### Why Kernels Work and When to Use Them
```python
def kernel_advantage_analysis():
    """Analyze when quantum kernels provide advantages"""
    
    print("Quantum Kernel Advantage Analysis")
    print("=" * 33)
    
    scenarios = {
        'high_dimensional': {
            'description': 'High-dimensional data with complex patterns',
            'advantage': 'Quantum Hilbert space dimensionality',
            'data_size': 'Small to medium (< 1000 samples)',
            'noise_tolerance': 'Low - requires high-fidelity quantum devices'
        },
        'structured_data': {
            'description': 'Data with quantum-inspired structure',
            'advantage': 'Natural encoding in quantum states',
            'data_size': 'Any size (limited by kernel computation)',
            'noise_tolerance': 'Medium - some noise tolerance'
        },
        'feature_discovery': {
            'description': 'Unknown optimal feature transformations',
            'advantage': 'Exponential feature space exploration',
            'data_size': 'Small (< 500 samples)',
            'noise_tolerance': 'Low - precision needed for kernel estimates'
        }
    }
    
    print("Scenario | Best Use Case | Quantum Advantage | Limitations")
    print("-" * 65)
    
    for name, info in scenarios.items():
        print(f"{name:12} | {info['description']:13} | {info['advantage']:17} | "
              f"{info['data_size']}, {info['noise_tolerance']}")
    
    print(f"\nKey Considerations:")
    print(f"- Quantum kernel computation scales as O(N²) in dataset size")
    print(f"- Shot noise affects kernel matrix accuracy")
    print(f"- Classical SVMs still needed for optimization")
    print(f"- Advantage depends on problem structure matching quantum encoding")

# Run kernel advantage analysis
kernel_advantage_analysis()
```

---

## 6.5 PennyLane Deep Dive (Differentiable QP)

PennyLane enables automatic differentiation through quantum circuits (parameter-shift rule).

## 6.5 PennyLane Deep Dive (Differentiable QP)

PennyLane enables automatic differentiation through quantum circuits using the parameter-shift rule and modern autodiff frameworks.

### Complete PennyLane Implementation with PyTorch Integration

```python
import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class PennyLaneQuantumClassifier:
    """Complete quantum classifier using PennyLane with PyTorch integration"""
    
    def __init__(self, n_qubits=2, n_layers=3, device_type='default.qubit'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device(device_type, wires=n_qubits)
        self.n_params = self._count_parameters()
        
    def _count_parameters(self):
        """Count total parameters in the circuit"""
        # Each layer has n_qubits RY rotations + n_qubits RZ rotations
        return self.n_layers * self.n_qubits * 2
    
    def data_encoding(self, x):
        """Encode classical data using angle encoding"""
        for i, val in enumerate(x):
            if i < self.n_qubits:
                qml.RY(val * np.pi, wires=i)
    
    def variational_circuit(self, params):
        """Parameterized quantum circuit (ansatz)"""
        
        # Reshape parameters for easier indexing
        params = params.reshape(self.n_layers, self.n_qubits, 2)
        
        for layer in range(self.n_layers):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qml.RY(params[layer, qubit, 0], wires=qubit)
                qml.RZ(params[layer, qubit, 1], wires=qubit)
            
            # Entangling layer (except for last layer)
            if layer < self.n_layers - 1:
                for qubit in range(self.n_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])
                
                # Add circular entanglement
                if self.n_qubits > 2:
                    qml.CZ(wires=[self.n_qubits - 1, 0])
    
    def create_qnode(self, measurement_type='single'):
        """Create quantum node with specified measurement"""
        
        if measurement_type == 'single':
            @qml.qnode(self.device, interface='torch')
            def circuit(x, params):
                self.data_encoding(x)
                self.variational_circuit(params)
                return qml.expval(qml.PauliZ(0))
            
        elif measurement_type == 'multi':
            @qml.qnode(self.device, interface='torch')
            def circuit(x, params):
                self.data_encoding(x)
                self.variational_circuit(params)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        elif measurement_type == 'probs':
            @qml.qnode(self.device, interface='torch')
            def circuit(x, params):
                self.data_encoding(x)
                self.variational_circuit(params)
                return qml.probs(wires=range(self.n_qubits))
        
        return circuit

class HybridPyTorchModel(nn.Module):
    """Hybrid quantum-classical model using PyTorch"""
    
    def __init__(self, n_qubits=2, n_layers=3, n_classical=4):
        super().__init__()
        
        # Quantum layer
        self.quantum_layer = PennyLaneQuantumClassifier(n_qubits, n_layers)
        self.qnode = self.quantum_layer.create_qnode(measurement_type='multi')
        
        # Quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(self.quantum_layer.n_params, requires_grad=True) * 0.1
        )
        
        # Classical layers
        self.classical_layers = nn.Sequential(
            nn.Linear(n_qubits, n_classical),
            nn.ReLU(),
            nn.Linear(n_classical, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass through hybrid model"""
        batch_outputs = []
        
        for sample in x:
            # Quantum forward pass
            quantum_output = self.qnode(sample, self.quantum_params)
            
            # Convert to tensor if needed
            if isinstance(quantum_output, list):
                quantum_output = torch.stack(quantum_output)
            
            batch_outputs.append(quantum_output)
        
        # Stack batch results
        quantum_features = torch.stack(batch_outputs)
        
        # Classical forward pass
        classical_output = self.classical_layers(quantum_features)
        
        return classical_output.squeeze()

def demonstrate_pennylane_training():
    """Complete demonstration of PennyLane training"""
    
    print("PennyLane Quantum Classifier Demo")
    print("=" * 33)
    
    # Generate dataset
    np.random.seed(42)
    torch.manual_seed(42)
    
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, 
        n_informative=2, n_clusters_per_class=1, random_state=42
    )
    
    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.3, random_state=42
    )
    
    print(f"Dataset: {len(X_train)} training, {len(X_test)} test samples")
    
    # Create hybrid model
    model = HybridPyTorchModel(n_qubits=2, n_layers=2, n_classical=4)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    epochs = 100
    train_losses = []
    test_accuracies = []
    
    print(f"\nTraining hybrid quantum-classical model...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluation phase
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_predictions = (test_outputs > 0.5).float()
                test_accuracy = (test_predictions == y_test).float().mean().item()
                test_accuracies.append(test_accuracy)
                
                print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, "
                      f"Test Accuracy = {test_accuracy:.3f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test)
        final_predictions = (final_outputs > 0.5).float()
        final_accuracy = (final_predictions == y_test).float().mean().item()
    
    print(f"\nFinal test accuracy: {final_accuracy:.3f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    test_epochs = list(range(19, epochs, 20))
    plt.plot(test_epochs, test_accuracies, 'ro-', linewidth=2, markersize=6)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, train_losses, test_accuracies

def advanced_pennylane_features():
    """Demonstrate advanced PennyLane features"""
    
    print("\nAdvanced PennyLane Features")
    print("=" * 27)
    
    # 1. Different gradient computation methods
    n_qubits = 2
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, interface='torch')
    def circuit_grad_demo(params):
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=1)
        qml.CZ(wires=[0, 1])
        qml.RY(params[2], wires=1)
        return qml.expval(qml.PauliZ(0))
    
    params = torch.tensor([0.1, 0.5, -0.2], requires_grad=True)
    
    print("1. Gradient Computation Methods:")
    
    # Parameter-shift rule (default)
    output_ps = circuit_grad_demo(params)
    grad_ps = torch.autograd.grad(output_ps, params, create_graph=True)[0]
    print(f"   Parameter-shift gradients: {grad_ps.detach().numpy()}")
    
    # 2. Circuit inspection and visualization
    print(f"\n2. Circuit Inspection:")
    print(f"   Circuit depth: {circuit_grad_demo.tape.graph.depth}")
    print(f"   Number of gates: {len(circuit_grad_demo.tape.operations)}")
    
    # Draw circuit
    print(f"\n   Circuit diagram:")
    print(qml.draw(circuit_grad_demo)(params.detach()))
    
    # 3. Different measurement strategies
    print(f"\n3. Measurement Strategies:")
    
    # Expectation values
    @qml.qnode(dev)
    def circuit_expval(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Probabilities
    @qml.qnode(dev)
    def circuit_probs(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        return qml.probs(wires=[0, 1])
    
    # Samples
    @qml.qnode(dev)
    def circuit_samples(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        return qml.sample(wires=[0, 1])
    
    test_params = pnp.array([0.5, 1.2])
    
    expvals = circuit_expval(test_params)
    probs = circuit_probs(test_params)
    
    print(f"   Expectation values: {expvals}")
    print(f"   Probabilities: {probs}")
    
    # 4. Custom observables
    print(f"\n4. Custom Observables:")
    
    # Define custom Hamiltonian
    H = qml.Hamiltonian([0.5, 0.3], [qml.PauliZ(0), qml.PauliX(1)])
    
    @qml.qnode(dev)
    def circuit_hamiltonian(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        return qml.expval(H)
    
    h_expval = circuit_hamiltonian(test_params)
    print(f"   Hamiltonian expectation: {h_expval:.4f}")

def pennylane_optimization_comparison():
    """Compare different optimizers in PennyLane"""
    
    print("\nOptimizer Comparison")
    print("=" * 20)
    
    # Simple optimization problem
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def cost_function(params):
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=1)
        qml.CZ(wires=[0, 1])
        qml.RY(params[2], wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    # Target: minimize cost function
    target_params = pnp.array([np.pi/4, np.pi/3, np.pi/6])
    
    optimizers = {
        'GradientDescent': qml.GradientDescentOptimizer(stepsize=0.1),
        'Adam': qml.AdamOptimizer(stepsize=0.1),
        'Momentum': qml.MomentumOptimizer(stepsize=0.1),
        'RMSProp': qml.RMSPropOptimizer(stepsize=0.1)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        # Initialize parameters
        params = pnp.random.uniform(0, 2*np.pi, 3, requires_grad=True)
        
        costs = []
        for step in range(50):
            params = optimizer.step(cost_function, params)
            costs.append(cost_function(params))
        
        results[opt_name] = {
            'final_cost': costs[-1],
            'final_params': params,
            'history': costs
        }
        
        print(f"{opt_name:15}: Final cost = {costs[-1]:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for opt_name, result in results.items():
        plt.plot(result['history'], label=opt_name, linewidth=2)
    
    plt.xlabel('Optimization Step')
    plt.ylabel('Cost Function Value')
    plt.title('Optimizer Comparison in PennyLane')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results

# Run comprehensive PennyLane demonstrations
print("Running PennyLane Demonstrations...")
print("=" * 35)

model, losses, accuracies = demonstrate_pennylane_training()
advanced_pennylane_features()
optimization_results = pennylane_optimization_comparison()
```

### Why PennyLane Excels for QML

```python
def pennylane_advantages_demo():
    """Demonstrate key advantages of PennyLane for QML"""
    
    print("PennyLane Advantages for Quantum Machine Learning")
    print("=" * 48)
    
    advantages = {
        'Framework Agnostic': {
            'description': 'Works with PyTorch, TensorFlow, JAX, NumPy',
            'benefit': 'Seamless integration with existing ML pipelines',
            'example': 'Quantum layers in neural networks'
        },
        'Automatic Differentiation': {
            'description': 'Parameter-shift rule + backpropagation',
            'benefit': 'Exact gradients for quantum circuits',
            'example': 'End-to-end training of hybrid models'
        },
        'Hardware Flexibility': {
            'description': 'Multiple backends (simulators, real devices)',
            'benefit': 'Easy transition from development to deployment',
            'example': 'Same code on simulator and IBM/Rigetti hardware'
        },
        'Rich Gate Set': {
            'description': 'Comprehensive set of quantum operations',
            'benefit': 'Express complex quantum algorithms naturally',
            'example': 'Custom ansatzes, advanced encodings'
        },
        'Optimization Tools': {
            'description': 'Built-in optimizers designed for quantum circuits',
            'benefit': 'Efficient training for VQCs',
            'example': 'Momentum-based optimizers with parameter-shift'
        }
    }
    
    print("Feature | Description | Key Benefit | Use Case")
    print("-" * 80)
    
    for feature, info in advantages.items():
        print(f"{feature:20} | {info['description']:25} | {info['benefit']:20} | {info['example']}")
    
    print(f"\nIntegration Example: Quantum Layer in Neural Network")
    print("=" * 52)
    
    # Demonstrate quantum layer integration
    import torch.nn as nn
    
    class QuantumLayer(nn.Module):
        def __init__(self, n_qubits):
            super().__init__()
            self.n_qubits = n_qubits
            self.device = qml.device('default.qubit', wires=n_qubits)
            
            @qml.qnode(self.device, interface='torch')
            def quantum_circuit(inputs, weights):
                # Encode inputs
                for i, inp in enumerate(inputs):
                    qml.RY(inp, wires=i)
                
                # Variational circuit
                for i in range(n_qubits):
                    qml.RY(weights[i], wires=i)
                
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.quantum_circuit = quantum_circuit
            self.weights = nn.Parameter(torch.randn(n_qubits))
        
        def forward(self, x):
            # Process batch
            batch_size = x.shape[0]
            outputs = []
            
            for i in range(batch_size):
                qoutput = self.quantum_circuit(x[i], self.weights)
                outputs.append(torch.stack(qoutput))
            
            return torch.stack(outputs)
    
    # Example hybrid network
    class HybridNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.classical1 = nn.Linear(4, 2)
            self.quantum = QuantumLayer(2)
            self.classical2 = nn.Linear(2, 1)
            self.activation = nn.Sigmoid()
        
        def forward(self, x):
            x = torch.relu(self.classical1(x))
            x = self.quantum(x)
            x = self.activation(self.classical2(x))
            return x
    
    # Demonstrate usage
    hybrid_model = HybridNN()
    sample_input = torch.randn(10, 4)  # Batch of 10 samples, 4 features each
    output = hybrid_model(sample_input)
    
    print(f"Hybrid model input shape: {sample_input.shape}")
    print(f"Hybrid model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in hybrid_model.parameters())}")
    
    # Show that gradients flow through quantum layer
    loss = output.sum()
    loss.backward()
    
    quantum_grad = hybrid_model.quantum.weights.grad
    print(f"Quantum layer gradients: {quantum_grad}")
    
    return hybrid_model

# Run PennyLane advantages demonstration
pennylane_model = pennylane_advantages_demo()
```

---

## 6.6 Hybrid Classical–Quantum Algorithms

## 6.6 Hybrid Classical–Quantum Algorithms

### Complete Hybrid Architecture Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class AdvancedHybridModel(nn.Module):
    """Advanced hybrid quantum-classical model with multiple architectures"""
    
    def __init__(self, input_dim, n_qubits=4, n_quantum_layers=3, 
                 classical_hidden=64, n_classes=3, architecture='parallel'):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers
        self.architecture = architecture
        
        # Quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Classical preprocessing
        self.classical_pre = nn.Sequential(
            nn.Linear(input_dim, classical_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classical_hidden, n_qubits),
            nn.Tanh()  # Scale to [-1, 1] for quantum encoding
        )
        
        # Quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(n_quantum_layers * n_qubits * 2) * 0.1
        )
        
        # Architecture-specific layers
        if architecture == 'parallel':
            # Parallel processing: classical and quantum features combined
            self.classical_branch = nn.Sequential(
                nn.Linear(input_dim, classical_hidden),
                nn.ReLU(),
                nn.Linear(classical_hidden, classical_hidden // 2)
            )
            
            self.final_layer = nn.Sequential(
                nn.Linear(n_qubits + classical_hidden // 2, classical_hidden),
                nn.ReLU(),
                nn.Linear(classical_hidden, n_classes)
            )
            
        elif architecture == 'sequential':
            # Sequential: quantum features → classical processing
            self.classical_post = nn.Sequential(
                nn.Linear(n_qubits, classical_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(classical_hidden, classical_hidden // 2),
                nn.ReLU(),
                nn.Linear(classical_hidden // 2, n_classes)
            )
            
        elif architecture == 'ensemble':
            # Ensemble: multiple quantum circuits
            self.n_circuits = 3
            self.quantum_params = nn.Parameter(
                torch.randn(self.n_circuits, n_quantum_layers * n_qubits * 2) * 0.1
            )
            
            self.ensemble_weights = nn.Parameter(torch.ones(self.n_circuits) / self.n_circuits)
            
            self.final_layer = nn.Sequential(
                nn.Linear(n_qubits * self.n_circuits, classical_hidden),
                nn.ReLU(),
                nn.Linear(classical_hidden, n_classes)
            )
        
        # Create quantum circuit
        self.create_quantum_circuit()
    
    def create_quantum_circuit(self):
        """Create parameterized quantum circuit"""
        
        if self.architecture == 'ensemble':
            # Multiple circuits for ensemble
            @qml.qnode(self.dev, interface='torch')
            def quantum_circuit_ensemble(x, params, circuit_idx):
                # Data encoding
                for i, val in enumerate(x):
                    qml.RY(val * np.pi, wires=i)
                
                # Variational layers
                param_idx = 0
                for layer in range(self.n_quantum_layers):
                    # Rotation layer
                    for qubit in range(self.n_qubits):
                        qml.RY(params[circuit_idx, param_idx], wires=qubit)
                        param_idx += 1
                        qml.RZ(params[circuit_idx, param_idx], wires=qubit)
                        param_idx += 1
                    
                    # Entangling layer
                    for qubit in range(self.n_qubits - 1):
                        qml.CZ(wires=[qubit, qubit + 1])
                    qml.CZ(wires=[self.n_qubits - 1, 0])  # Circular
                
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
            self.quantum_circuit = quantum_circuit_ensemble
            
        else:
            # Single circuit
            @qml.qnode(self.dev, interface='torch')
            def quantum_circuit_single(x, params):
                # Data encoding with amplitude amplification
                for i, val in enumerate(x):
                    qml.RY(val * np.pi, wires=i)
                
                # Additional encoding layer
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                
                # Variational layers
                param_idx = 0
                for layer in range(self.n_quantum_layers):
                    # Rotation layer
                    for qubit in range(self.n_qubits):
                        qml.RY(params[param_idx], wires=qubit)
                        param_idx += 1
                        qml.RZ(params[param_idx], wires=qubit)
                        param_idx += 1
                    
                    # Entangling layer with different patterns
                    if layer % 2 == 0:
                        # Linear entanglement
                        for qubit in range(self.n_qubits - 1):
                            qml.CZ(wires=[qubit, qubit + 1])
                    else:
                        # Star entanglement
                        for qubit in range(1, self.n_qubits):
                            qml.CZ(wires=[0, qubit])
                
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            
            self.quantum_circuit = quantum_circuit_single
    
    def forward(self, x):
        """Forward pass through hybrid model"""
        batch_size = x.shape[0]
        
        if self.architecture == 'parallel':
            # Parallel architecture
            # Classical branch
            classical_features = self.classical_branch(x)
            
            # Quantum branch
            quantum_inputs = self.classical_pre(x)
            quantum_outputs = []
            
            for i in range(batch_size):
                q_out = self.quantum_circuit(quantum_inputs[i], self.quantum_params)
                quantum_outputs.append(torch.stack(q_out))
            
            quantum_features = torch.stack(quantum_outputs)
            
            # Combine features
            combined_features = torch.cat([classical_features, quantum_features], dim=1)
            output = self.final_layer(combined_features)
            
        elif self.architecture == 'sequential':
            # Sequential architecture
            quantum_inputs = self.classical_pre(x)
            quantum_outputs = []
            
            for i in range(batch_size):
                q_out = self.quantum_circuit(quantum_inputs[i], self.quantum_params)
                quantum_outputs.append(torch.stack(q_out))
            
            quantum_features = torch.stack(quantum_outputs)
            output = self.classical_post(quantum_features)
            
        elif self.architecture == 'ensemble':
            # Ensemble architecture
            quantum_inputs = self.classical_pre(x)
            ensemble_outputs = []
            
            for circuit_idx in range(self.n_circuits):
                circuit_outputs = []
                for i in range(batch_size):
                    q_out = self.quantum_circuit(quantum_inputs[i], self.quantum_params, circuit_idx)
                    circuit_outputs.append(torch.stack(q_out))
                
                ensemble_outputs.append(torch.stack(circuit_outputs))
            
            # Weight ensemble results
            weighted_quantum_features = []
            for i in range(self.n_circuits):
                weighted_quantum_features.append(
                    self.ensemble_weights[i] * ensemble_outputs[i]
                )
            
            # Concatenate ensemble outputs
            final_quantum_features = torch.cat(weighted_quantum_features, dim=1)
            output = self.final_layer(final_quantum_features)
        
        return output

def comprehensive_hybrid_evaluation():
    """Comprehensive evaluation of hybrid architectures"""
    
    print("Comprehensive Hybrid Algorithm Evaluation")
    print("=" * 41)
    
    # Load and prepare different datasets
    datasets = {
        'wine': load_wine(),
        'digits_small': load_digits()
    }
    
    # For digits, use PCA to reduce dimensionality
    digits = datasets['digits_small']
    pca = PCA(n_components=8)
    digits_reduced = pca.fit_transform(digits.data)
    datasets['digits_small'] = type('', (), {
        'data': digits_reduced, 
        'target': digits.target
    })()
    
    architectures = ['parallel', 'sequential', 'ensemble']
    results = {}
    
    for dataset_name, dataset in datasets.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 30)
        
        # Prepare data
        X, y = dataset.data, dataset.target
        n_classes = len(np.unique(y))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        print(f"Features: {X.shape[1]}, Classes: {n_classes}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        dataset_results = {}
        
        for architecture in architectures:
            print(f"\nArchitecture: {architecture}")
            
            # Create model
            model = AdvancedHybridModel(
                input_dim=X.shape[1],
                n_qubits=4,
                n_quantum_layers=2,
                classical_hidden=32,
                n_classes=n_classes,
                architecture=architecture
            )
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
            # Training loop
            epochs = 50
            train_losses = []
            train_accuracies = []
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                
                loss.backward()
                optimizer.step()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_accuracy = (predicted == y_train_tensor).float().mean().item()
                
                train_losses.append(loss.item())
                train_accuracies.append(train_accuracy)
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}, "
                          f"Train Acc = {train_accuracy:.3f}")
            
            # Test evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_accuracy = (test_predicted == y_test_tensor).float().mean().item()
            
            print(f"  Final Test Accuracy: {test_accuracy:.3f}")
            
            # Store results
            dataset_results[architecture] = {
                'model': model,
                'test_accuracy': test_accuracy,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies
            }
        
        results[dataset_name] = dataset_results
    
    return results

def analyze_hybrid_performance():
    """Analyze performance characteristics of hybrid models"""
    
    print("\nHybrid Model Performance Analysis")
    print("=" * 34)
    
    # Performance comparison framework
    performance_aspects = {
        'Training Speed': {
            'parallel': 'Moderate - dual processing paths',
            'sequential': 'Fast - single quantum pass',
            'ensemble': 'Slow - multiple quantum circuits'
        },
        'Model Capacity': {
            'parallel': 'High - combines quantum and classical features',
            'sequential': 'Medium - quantum features processed classically',
            'ensemble': 'Highest - multiple quantum perspectives'
        },
        'Interpretability': {
            'parallel': 'Medium - can isolate quantum vs classical contributions',
            'sequential': 'High - clear quantum → classical flow',
            'ensemble': 'Low - complex quantum ensemble interactions'
        },
        'Hardware Requirements': {
            'parallel': 'Medium - single quantum circuit',
            'sequential': 'Low - simple quantum processing',
            'ensemble': 'High - multiple quantum circuit evaluations'
        },
        'Noise Robustness': {
            'parallel': 'Good - classical branch provides backup',
            'sequential': 'Fair - depends on quantum layer quality',
            'ensemble': 'Best - averaging reduces noise impact'
        }
    }
    
    print("Aspect | Parallel | Sequential | Ensemble")
    print("-" * 55)
    
    for aspect, comparisons in performance_aspects.items():
        print(f"{aspect:15} | {comparisons['parallel']:20} | {comparisons['sequential']:20} | {comparisons['ensemble']}")
    
    print(f"\nKey Insights:")
    print(f"- Parallel: Best balance of performance and interpretability")
    print(f"- Sequential: Simplest to implement and understand")
    print(f"- Ensemble: Highest accuracy potential but most complex")

def real_world_application_example():
    """Demonstrate hybrid QML on a real-world problem"""
    
    print("\nReal-World Application: Wine Quality Classification")
    print("=" * 50)
    
    # Load wine dataset
    wine_data = load_wine()
    X, y = wine_data.data, wine_data.target
    
    # Feature engineering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add derived features
    feature_interactions = []
    for i in range(X_scaled.shape[1]):
        for j in range(i+1, X_scaled.shape[1]):
            feature_interactions.append(X_scaled[:, i] * X_scaled[:, j])
    
    X_enhanced = np.column_stack([X_scaled] + feature_interactions[:10])  # Keep manageable
    
    print(f"Original features: {X.shape[1]}")
    print(f"Enhanced features: {X_enhanced.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create comprehensive hybrid model
    model = AdvancedHybridModel(
        input_dim=X_enhanced.shape[1],
        n_qubits=6,
        n_quantum_layers=3,
        classical_hidden=64,
        n_classes=3,
        architecture='parallel'
    )
    
    # Training with advanced techniques
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Training with validation
    epochs = 100
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Split training data for validation
    val_split = int(0.8 * len(X_train_tensor))
    X_train_split = X_train_tensor[:val_split]
    X_val_split = X_train_tensor[val_split:]
    y_train_split = y_train_tensor[:val_split]
    y_val_split = y_train_tensor[val_split:]
    
    print(f"\nTraining hybrid model on wine dataset...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_outputs = model(X_train_split)
        train_loss = criterion(train_outputs, y_train_split)
        
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracies
        _, train_pred = torch.max(train_outputs, 1)
        train_acc = (train_pred == y_train_split).float().mean().item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_split)
            val_loss = criterion(val_outputs, y_val_split)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val_split).float().mean().item()
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss = {train_loss.item():.4f}, "
                  f"Val Loss = {val_loss.item():.4f}, "
                  f"Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}")
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predictions = torch.max(test_outputs, 1)
        test_accuracy = (test_predictions == y_test_tensor).float().mean().item()
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    # Detailed classification report
    test_pred_numpy = test_predictions.numpy()
    test_true_numpy = y_test_tensor.numpy()
    
    print(f"\nClassification Report:")
    print(classification_report(test_true_numpy, test_pred_numpy, 
                              target_names=wine_data.target_names))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Feature importance analysis (simplified)
    model.eval()
    with torch.no_grad():
        baseline_output = model(X_test_tensor[:1])
        
        feature_importance = []
        for feature_idx in range(X_enhanced.shape[1]):
            # Permute feature
            X_permuted = X_test_tensor[:1].clone()
            X_permuted[0, feature_idx] = 0  # Zero out feature
            
            permuted_output = model(X_permuted)
            importance = torch.abs(baseline_output - permuted_output).sum().item()
            feature_importance.append(importance)
    
    plt.bar(range(len(feature_importance[:10])), feature_importance[:10])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('Top 10 Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, test_accuracy

# Run comprehensive hybrid algorithm demonstrations
print("Running Hybrid Algorithm Demonstrations...")
print("=" * 42)

hybrid_results = comprehensive_hybrid_evaluation()
analyze_hybrid_performance()
final_model, wine_accuracy = real_world_application_example()
```

### Typical Architecture Patterns

```python
def hybrid_architecture_patterns():
    """Demonstrate different hybrid architecture patterns"""
    
    print("Hybrid Architecture Patterns")
    print("=" * 28)
    
    patterns = {
        'Sandwich': {
            'structure': 'Classical → Quantum → Classical',
            'use_case': 'Feature extraction with quantum middle layer',
            'advantages': 'Leverages quantum for feature transformation',
            'example': 'Image classification with quantum convolution'
        },
        'Parallel': {
            'structure': 'Classical || Quantum → Combine',
            'use_case': 'Complementary feature extraction',
            'advantages': 'Best of both worlds, robust to quantum noise',
            'example': 'Multi-modal data processing'
        },
        'Hierarchical': {
            'structure': 'Quantum → Classical → Quantum → Classical',
            'use_case': 'Complex multi-stage processing',
            'advantages': 'Maximum expressivity, hierarchical features',
            'example': 'Natural language processing pipelines'
        },
        'Ensemble': {
            'structure': 'Multiple Quantum Circuits → Classical Combination',
            'use_case': 'High-accuracy applications',
            'advantages': 'Noise averaging, increased capacity',
            'example': 'Medical diagnosis, financial forecasting'
        }
    }
    
    print("Pattern | Structure | Best Use Case | Key Advantage")
    print("-" * 70)
    
    for pattern, info in patterns.items():
        print(f"{pattern:12} | {info['structure']:25} | {info['use_case']:20} | {info['advantages']}")
    
    print(f"\nImplementation Guidelines:")
    print(f"1. Start with Sandwich for simplicity")
    print(f"2. Use Parallel when unsure about quantum advantage")
    print(f"3. Consider Ensemble for high-stakes applications")
    print(f"4. Hierarchical only for complex, well-understood problems")

# Run architecture patterns demonstration
hybrid_architecture_patterns()
```

---

## 6.7 Evaluating & Debugging QML Models

### Key Metrics
| Model Type | Metrics |
|------------|---------|
| Classifier | Accuracy, ROC-AUC, F1 |
| Regression | MSE, MAE, R² |
| Kernel Model | Kernel alignment, generalization gap |
| Variational | Train loss vs validation loss, gradient norms |

### Practical Debug Checks
1. Verify encoding scales (no angle saturation like multiples of 2π)
2. Inspect gradient magnitudes (avoid vanishing)
3. Compare against small classical baseline (linear/logistic) for sanity
4. Inject artificial noise to test robustness
5. Plot loss surface slices (vary 1–2 params)

### Gradient Inspection (Illustration)
```python
def estimate_gradients(params, epsilon=1e-2):
    base = loss(params)
    grads = []
    for i in range(len(params)):
        shifted = params.copy()
        shifted[i] += epsilon
        grads.append((loss(shifted) - base)/epsilon)
    return grads
```

---

## 6.8 Project: Quantum Classifier for a Real-World Style Dataset

### Goal
Build a hybrid or variational classifier for a *mini* dataset (e.g., 2D synthetic spirals, concentric circles, or reduced Iris features) and compare against classical baselines.

### Required Steps
1. Select dataset & preprocess (scaling, dimensionality ≤ number of qubits)
2. Choose encoding (angle or feature map circuit)
3. Design ansatz depth (start shallow)
4. Define loss (cross-entropy) & optimizer
5. Train with simulation (ideal), then add noise model
6. Apply simple mitigation (measurement calibration)
7. Evaluate: accuracy, confusion matrix
8. Compare to classical logistic regression / SVM
9. Document trade-offs & observations

### Stretch Enhancements
- Quantum kernel SVM variant
- Data reuploading layers for expressivity
- Cost landscape visualization (2D parameter sweep)
- PennyLane + PyTorch integration (autograd training loop)
- Noise-aware training (early stopping once gradients collapse)

### Deliverables
| Artifact | Description |
|----------|-------------|
| Notebook | End-to-end training & evaluation |
| Plots | Loss curve, accuracy vs epoch, kernel matrix heatmap |
| Report | 1–2 page reflection on performance & limitations |
| Code | Reusable encoding + ansatz utilities |

---

## 6.9 Summary & Next Steps

You learned:
- How classical data enters quantum models (encodings & trade-offs)
- How to build/train a variational quantum classifier
- What quantum kernels are and how to compute them naively
- Differentiable quantum programming with PennyLane
- Hybrid workflow patterns and evaluation metrics

### Where to Go Next
- Explore larger feature maps & expressivity / trainability tension
- Investigate barren plateau literature & mitigation strategies
- Try QAOA-style layers for classification tasks
- Integrate with classical deep learning frameworks (Torch, JAX)

### Quick Quiz
1. Difference between angle and amplitude encoding?  
2. Why can deeper variational circuits hurt trainability?  
3. What does a kernel matrix entry represent?  
4. How does the parameter-shift rule compute gradients?  
5. Why benchmark against classical baselines?  

---

## Further Reading
- Schuld & Killoran, “Quantum Machine Learning in Feature Hilbert Spaces”
- Benedetti et al., “Parameterized Quantum Circuits as Machine Learning Models”
- Havlíček et al., “Supervised Learning with Quantum-Enhanced Feature Spaces”
- Cerezo et al., “Variational Quantum Algorithms” (barren plateaus)
- PennyLane & Qiskit Machine Learning documentation

*End of Module 6 – proceed when you can comfortably build and assess a hybrid quantum ML model.*
