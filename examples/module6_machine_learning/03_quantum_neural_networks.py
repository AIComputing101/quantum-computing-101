#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 6: Quantum Machine Learning
Example 3: Quantum Neural Networks

Implementation of quantum neural networks with gradient computation and hybrid architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.datasets import make_regression, make_classification, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class QuantumNeuralNetwork:
    def __init__(self, n_qubits, n_layers=2, task_type='regression', verbose=False):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.task_type = task_type
        self.verbose = verbose
        self.parameters = None
        self.training_history = []
        
    def create_encoding_layer(self, x):
        """Create data encoding layer."""
        circuit = QuantumCircuit(self.n_qubits, name='Encoding')
        
        # Angle encoding
        for i in range(min(len(x), self.n_qubits)):
            circuit.ry(x[i], i)
        
        return circuit
    
    def create_variational_layer(self, params, layer_idx):
        """Create a variational layer."""
        circuit = QuantumCircuit(self.n_qubits, name=f'Var_Layer_{layer_idx}')
        param_idx = layer_idx * 3 * self.n_qubits  # 3 parameters per qubit per layer
        
        # Single-qubit rotations
        for i in range(self.n_qubits):
            if param_idx < len(params):
                circuit.rx(params[param_idx], i)
                param_idx += 1
            if param_idx < len(params):
                circuit.ry(params[param_idx], i)
                param_idx += 1
            if param_idx < len(params):
                circuit.rz(params[param_idx], i)
                param_idx += 1
        
        # Entangling gates
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        
        return circuit
    
    def create_qnn_circuit(self, x, params):
        """Create complete QNN circuit."""
        # Start with encoding
        circuit = self.create_encoding_layer(x)
        
        # Add variational layers
        for layer in range(self.n_layers):
            var_layer = self.create_variational_layer(params, layer)
            circuit.compose(var_layer, inplace=True)
        
        return circuit
    
    def measure_expectation_values(self, circuit, observables=['Z0']):
        """Measure expectation values of observables."""
        expectations = []
        
        for obs in observables:
            if obs == 'Z0':
                # Measure Z expectation on first qubit
                meas_circuit = circuit.copy()
                meas_circuit.add_register(ClassicalRegister(1))
                meas_circuit.measure(0, 0)
                
                simulator = AerSimulator()
                job = simulator.run(meas_circuit, shots=1024)
                result = job.result()
                counts = result.get_counts()
                
                prob_0 = counts.get('0', 0) / 1024
                prob_1 = counts.get('1', 0) / 1024
                expectation = prob_0 - prob_1
                expectations.append(expectation)
                
            elif obs.startswith('Z'):
                # Measure Z on specified qubit
                qubit_idx = int(obs[1:])
                if qubit_idx < self.n_qubits:
                    meas_circuit = circuit.copy()
                    meas_circuit.add_register(ClassicalRegister(1))
                    meas_circuit.measure(qubit_idx, 0)
                    
                    simulator = AerSimulator()
                    job = simulator.run(meas_circuit, shots=1024)
                    result = job.result()
                    counts = result.get_counts()
                    
                    prob_0 = counts.get('0', 0) / 1024
                    prob_1 = counts.get('1', 0) / 1024
                    expectation = prob_0 - prob_1
                    expectations.append(expectation)
        
        return np.array(expectations)
    
    def forward_pass(self, X, params):
        """Forward pass through QNN."""
        outputs = []
        
        for x in X:
            circuit = self.create_qnn_circuit(x, params)
            
            if self.task_type == 'regression':
                # Single output for regression
                exp_vals = self.measure_expectation_values(circuit, ['Z0'])
                outputs.append(exp_vals[0])
            else:
                # Multiple outputs for classification
                observables = [f'Z{i}' for i in range(min(2, self.n_qubits))]
                exp_vals = self.measure_expectation_values(circuit, observables)
                outputs.append(exp_vals)
        
        return np.array(outputs)
    
    def compute_gradients(self, X, y, params, epsilon=0.01):
        """Compute gradients using parameter shift rule."""
        gradients = np.zeros_like(params)
        
        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += epsilon
            loss_plus = self.loss_function(X, y, params_plus)
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= epsilon
            loss_minus = self.loss_function(X, y, params_minus)
            
            # Central difference
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients
    
    def loss_function(self, X, y, params):
        """Compute loss function."""
        outputs = self.forward_pass(X, params)
        
        if self.task_type == 'regression':
            # Mean squared error
            loss = np.mean((outputs - y) ** 2)
        else:
            # Binary classification loss
            if outputs.ndim == 1:
                # Single output
                probs = (outputs + 1) / 2  # Convert from [-1,1] to [0,1]
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            else:
                # Multi-output classification (simplified)
                predictions = np.argmax(outputs, axis=1)
                loss = 1 - accuracy_score(y, predictions)
        
        return loss
    
    def fit(self, X, y, max_iter=100, learning_rate=0.01):
        """Train the QNN."""
        # Initialize parameters
        n_params = 3 * self.n_qubits * self.n_layers
        self.parameters = np.random.uniform(-np.pi, np.pi, n_params)
        
        self.training_history = []
        
        if self.verbose:
            print(f"Training QNN with {n_params} parameters...")
        
        for iteration in range(max_iter):
            # Compute loss
            loss = self.loss_function(X, y, self.parameters)
            
            # Compute gradients
            gradients = self.compute_gradients(X, y, self.parameters)
            
            # Update parameters
            self.parameters -= learning_rate * gradients
            
            # Store history
            outputs = self.forward_pass(X, y)
            if self.task_type == 'regression':
                metric = np.sqrt(mean_squared_error(y, outputs))
                metric_name = 'RMSE'
            else:
                if outputs.ndim == 1:
                    predictions = (outputs > 0).astype(int)
                else:
                    predictions = np.argmax(outputs, axis=1)
                metric = accuracy_score(y, predictions)
                metric_name = 'Accuracy'
            
            self.training_history.append({
                'iteration': iteration,
                'loss': loss,
                'metric': metric,
                'metric_name': metric_name
            })
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Loss = {loss:.4f}, {metric_name} = {metric:.4f}")
            
            # Early stopping
            if loss < 1e-6:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.parameters is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        outputs = self.forward_pass(X, self.parameters)
        
        if self.task_type == 'regression':
            return outputs
        else:
            if outputs.ndim == 1:
                return (outputs > 0).astype(int)
            else:
                return np.argmax(outputs, axis=1)

class HybridNeuralNetwork:
    """Hybrid quantum-classical neural network."""
    
    def __init__(self, n_qubits, classical_hidden_units=10, task_type='regression', verbose=False):
        self.n_qubits = n_qubits
        self.classical_hidden_units = classical_hidden_units
        self.task_type = task_type
        self.verbose = verbose
        
        # Quantum layer
        self.quantum_layer = QuantumNeuralNetwork(
            n_qubits=n_qubits, n_layers=1, task_type='regression', verbose=False
        )
        
        # Classical layer
        if task_type == 'regression':
            self.classical_layer = MLPRegressor(
                hidden_layer_sizes=(classical_hidden_units,),
                max_iter=1000,
                random_state=42
            )
        else:
            self.classical_layer = MLPClassifier(
                hidden_layer_sizes=(classical_hidden_units,),
                max_iter=1000,
                random_state=42
            )
    
    def fit(self, X, y):
        """Train hybrid network."""
        # Train quantum layer first
        if self.verbose:
            print("Training quantum layer...")
        
        # Use subset for quantum training (faster)
        n_quantum_samples = min(50, len(X))
        indices = np.random.choice(len(X), n_quantum_samples, replace=False)
        X_quantum = X[indices]
        y_quantum = y[indices]
        
        self.quantum_layer.fit(X_quantum, y_quantum, max_iter=50)
        
        # Get quantum features for all data
        if self.verbose:
            print("Computing quantum features...")
        
        quantum_features = self.quantum_layer.forward_pass(X, self.quantum_layer.parameters)
        
        # Ensure quantum features are 2D
        if quantum_features.ndim == 1:
            quantum_features = quantum_features.reshape(-1, 1)
        
        # Combine with classical features
        if quantum_features.shape[1] == 1:
            # Single quantum feature
            hybrid_features = np.column_stack([X, quantum_features])
        else:
            # Multiple quantum features
            hybrid_features = np.column_stack([X, quantum_features])
        
        # Train classical layer
        if self.verbose:
            print("Training classical layer...")
        
        self.classical_layer.fit(hybrid_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using hybrid network."""
        # Get quantum features
        quantum_features = self.quantum_layer.forward_pass(X, self.quantum_layer.parameters)
        
        if quantum_features.ndim == 1:
            quantum_features = quantum_features.reshape(-1, 1)
        
        # Combine features
        if quantum_features.shape[1] == 1:
            hybrid_features = np.column_stack([X, quantum_features])
        else:
            hybrid_features = np.column_stack([X, quantum_features])
        
        # Classical prediction
        return self.classical_layer.predict(hybrid_features)

class QNNAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def compare_architectures(self, X_train, y_train, X_test, y_test, task_type='regression'):
        """Compare different QNN architectures."""
        results = {}
        
        # Pure Quantum
        qnn = QuantumNeuralNetwork(
            n_qubits=2, n_layers=2, task_type=task_type, verbose=self.verbose
        )
        
        # Adjust features
        X_train_q = X_train[:, :2] if X_train.shape[1] > 2 else X_train
        X_test_q = X_test[:, :2] if X_test.shape[1] > 2 else X_test
        
        try:
            qnn.fit(X_train_q, y_train, max_iter=50)
            qnn_predictions = qnn.predict(X_test_q)
            
            if task_type == 'regression':
                qnn_score = np.sqrt(mean_squared_error(y_test, qnn_predictions))
                score_name = 'RMSE'
            else:
                qnn_score = accuracy_score(y_test, qnn_predictions)
                score_name = 'Accuracy'
            
            results['Pure Quantum'] = {
                'score': qnn_score,
                'score_name': score_name,
                'training_history': qnn.training_history
            }
        
        except Exception as e:
            results['Pure Quantum'] = {'error': str(e)}
        
        # Hybrid Network
        try:
            hybrid = HybridNeuralNetwork(
                n_qubits=2, task_type=task_type, verbose=self.verbose
            )
            hybrid.fit(X_train, y_train)
            hybrid_predictions = hybrid.predict(X_test)
            
            if task_type == 'regression':
                hybrid_score = np.sqrt(mean_squared_error(y_test, hybrid_predictions))
            else:
                hybrid_score = accuracy_score(y_test, hybrid_predictions)
            
            results['Hybrid'] = {
                'score': hybrid_score,
                'score_name': score_name
            }
        
        except Exception as e:
            results['Hybrid'] = {'error': str(e)}
        
        # Classical Neural Network
        if task_type == 'regression':
            classical = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42)
        else:
            classical = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42)
        
        classical.fit(X_train, y_train)
        classical_predictions = classical.predict(X_test)
        
        if task_type == 'regression':
            classical_score = np.sqrt(mean_squared_error(y_test, classical_predictions))
        else:
            classical_score = accuracy_score(y_test, classical_predictions)
        
        results['Classical'] = {
            'score': classical_score,
            'score_name': score_name
        }
        
        return results
    
    def analyze_gradient_flow(self, X, y, n_qubits=2, task_type='regression'):
        """Analyze gradient flow in QNN."""
        qnn = QuantumNeuralNetwork(n_qubits, n_layers=1, task_type=task_type, verbose=False)
        
        # Random parameters
        n_params = 3 * n_qubits * 1
        params = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Compute gradients
        gradients = qnn.compute_gradients(X, y, params)
        
        # Analyze gradient statistics
        return {
            'gradients': gradients,
            'gradient_norm': np.linalg.norm(gradients),
            'mean_gradient': np.mean(np.abs(gradients)),
            'max_gradient': np.max(np.abs(gradients)),
            'gradient_variance': np.var(gradients)
        }
    
    def visualize_results(self, architecture_results, gradient_analysis=None):
        """Visualize QNN analysis results."""
        fig = plt.figure(figsize=(15, 10))
        
        # Architecture comparison
        ax1 = plt.subplot(2, 3, 1)
        
        architectures = []
        scores = []
        score_name = None
        
        for arch, result in architecture_results.items():
            if 'error' not in result:
                architectures.append(arch)
                scores.append(result['score'])
                if score_name is None:
                    score_name = result['score_name']
        
        if architectures:
            colors = ['blue', 'green', 'red'][:len(architectures)]
            bars = ax1.bar(architectures, scores, alpha=0.7, color=colors)
            
            ax1.set_ylabel(score_name)
            ax1.set_title('Architecture Comparison')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # Training convergence (if available)
        ax2 = plt.subplot(2, 3, 2)
        
        if 'Pure Quantum' in architecture_results and 'training_history' in architecture_results['Pure Quantum']:
            history = architecture_results['Pure Quantum']['training_history']
            iterations = [h['iteration'] for h in history]
            losses = [h['loss'] for h in history]
            metrics = [h['metric'] for h in history]
            metric_name = history[0]['metric_name'] if history else 'Metric'
            
            ax2_twin = ax2.twinx()
            line1 = ax2.plot(iterations, losses, 'b-', label='Loss', linewidth=2)
            line2 = ax2_twin.plot(iterations, metrics, 'r-', label=metric_name, linewidth=2)
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss', color='blue')
            ax2_twin.set_ylabel(metric_name, color='red')
            ax2.set_title('QNN Training Convergence')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='best')
            
            ax2.grid(True, alpha=0.3)
        
        # Gradient analysis
        if gradient_analysis:
            ax3 = plt.subplot(2, 3, 3)
            gradients = gradient_analysis['gradients']
            
            ax3.hist(gradients, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(np.mean(gradients), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(gradients):.4f}')
            ax3.set_xlabel('Gradient Value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Gradient Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Gradient statistics subplot
            ax4 = plt.subplot(2, 3, 4)
            
            grad_stats = [
                gradient_analysis['gradient_norm'],
                gradient_analysis['mean_gradient'],
                gradient_analysis['max_gradient'],
                gradient_analysis['gradient_variance']
            ]
            
            stat_names = ['Norm', 'Mean |∇|', 'Max |∇|', 'Variance']
            
            ax4.bar(stat_names, grad_stats, alpha=0.7, color='orange')
            ax4.set_ylabel('Value')
            ax4.set_title('Gradient Statistics')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        # Performance comparison
        ax5 = plt.subplot(2, 3, 5)
        
        if len(architectures) >= 2:
            # Create radar chart for multi-metric comparison
            metrics = ['Performance', 'Complexity', 'Trainability']
            
            # Normalize scores for comparison
            if score_name == 'RMSE':
                # Lower is better for RMSE
                norm_scores = [1.0 / (1.0 + s) for s in scores]
            else:
                # Higher is better for accuracy
                norm_scores = scores
            
            # Simple performance comparison
            performance_data = {
                'Quantum': norm_scores[0] if 'Pure Quantum' in architectures else 0.5,
                'Hybrid': norm_scores[1] if len(norm_scores) > 1 and 'Hybrid' in architectures else 0.7,
                'Classical': norm_scores[-1] if 'Classical' in architectures else 0.8
            }
            
            methods = list(performance_data.keys())
            values = list(performance_data.values())
            
            ax5.bar(methods, values, alpha=0.7, color=['blue', 'green', 'red'])
            ax5.set_ylabel('Normalized Performance')
            ax5.set_title('Method Comparison')
            ax5.grid(True, alpha=0.3)
        
        # Summary text
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "QNN Analysis Summary:\n\n"
        
        if architectures:
            best_idx = np.argmax(scores) if score_name == 'Accuracy' else np.argmin(scores)
            best_arch = architectures[best_idx]
            best_score = scores[best_idx]
            
            summary_text += f"Best Method: {best_arch}\n"
            summary_text += f"{score_name}: {best_score:.3f}\n\n"
        
        if gradient_analysis:
            summary_text += f"Gradient Analysis:\n"
            summary_text += f"Mean |∇|: {gradient_analysis['mean_gradient']:.4f}\n"
            summary_text += f"Grad Norm: {gradient_analysis['gradient_norm']:.4f}\n\n"
        
        summary_text += "Key Insights:\n"
        summary_text += "• QNNs use quantum circuits\n"
        summary_text += "• Hybrid architectures combine\n  quantum and classical layers\n"
        summary_text += "• Gradient computation uses\n  parameter shift rule"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Quantum Neural Networks")
    parser.add_argument('--task', choices=['regression', 'classification'], default='regression')
    parser.add_argument('--dataset', choices=['synthetic', 'boston'], default='synthetic')
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--n-qubits', type=int, default=2)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--analyze-gradients', action='store_true')
    parser.add_argument('--show-visualization', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    print("Quantum Computing 101 - Module 6: Quantum Machine Learning")
    print("Example 3: Quantum Neural Networks")
    print("=" * 41)
    
    try:
        # Load dataset
        if args.task == 'regression':
            if args.dataset == 'boston':
                # Create synthetic regression data since Boston housing is deprecated
                X, y = make_regression(
                    n_samples=args.n_samples,
                    n_features=4,
                    noise=0.1,
                    random_state=42
                )
            else:
                X, y = make_regression(
                    n_samples=args.n_samples,
                    n_features=3,
                    noise=0.1,
                    random_state=42
                )
        else:
            X, y = make_classification(
                n_samples=args.n_samples,
                n_features=3,
                n_redundant=0,
                n_informative=3,
                n_clusters_per_class=1,
                random_state=42
            )
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"\n📊 Dataset: {args.task} task")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {X.shape[1]}")
        
        # Single QNN training
        print(f"\n🧠 Training QNN ({args.n_qubits} qubits, {args.n_layers} layers)...")
        
        qnn = QuantumNeuralNetwork(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            task_type=args.task,
            verbose=args.verbose
        )
        
        # Adjust features for quantum circuit
        X_train_q = X_train[:, :args.n_qubits] if X_train.shape[1] > args.n_qubits else X_train
        X_test_q = X_test[:, :args.n_qubits] if X_test.shape[1] > args.n_qubits else X_test
        
        qnn.fit(X_train_q, y_train, max_iter=args.max_iter)
        
        # Evaluate
        train_predictions = qnn.predict(X_train_q)
        test_predictions = qnn.predict(X_test_q)
        
        if args.task == 'regression':
            train_score = np.sqrt(mean_squared_error(y_train, train_predictions))
            test_score = np.sqrt(mean_squared_error(y_test, test_predictions))
            score_name = 'RMSE'
        else:
            train_score = accuracy_score(y_train, train_predictions)
            test_score = accuracy_score(y_test, test_predictions)
            score_name = 'Accuracy'
        
        print(f"\n📈 QNN Results:")
        print(f"   Training {score_name}: {train_score:.3f}")
        print(f"   Test {score_name}: {test_score:.3f}")
        print(f"   Parameters: {len(qnn.parameters)}")
        print(f"   Training iterations: {len(qnn.training_history)}")
        
        # Architecture comparison
        analyzer = QNNAnalyzer(verbose=args.verbose)
        
        print(f"\n🏗️  Comparing architectures...")
        architecture_results = analyzer.compare_architectures(
            X_train, y_train, X_test, y_test, args.task
        )
        
        print(f"\n📊 Architecture Comparison:")
        for arch, results in architecture_results.items():
            if 'error' not in results:
                print(f"   {arch:15s}: {results['score_name']} = {results['score']:.3f}")
            else:
                print(f"   {arch:15s}: Error - {results['error']}")
        
        # Gradient analysis
        gradient_analysis = None
        if args.analyze_gradients:
            print(f"\n🎯 Analyzing gradients...")
            gradient_analysis = analyzer.analyze_gradient_flow(
                X_train_q[:10], y_train[:10], args.n_qubits, args.task
            )
            
            print(f"   Gradient norm: {gradient_analysis['gradient_norm']:.4f}")
            print(f"   Mean |gradient|: {gradient_analysis['mean_gradient']:.4f}")
            print(f"   Max |gradient|: {gradient_analysis['max_gradient']:.4f}")
            print(f"   Gradient variance: {gradient_analysis['gradient_variance']:.4f}")
        
        # Find best method
        valid_results = {k: v for k, v in architecture_results.items() if 'error' not in v}
        if valid_results:
            if args.task == 'regression':
                best_method = min(valid_results.items(), key=lambda x: x[1]['score'])
            else:
                best_method = max(valid_results.items(), key=lambda x: x[1]['score'])
            
            print(f"\n🏆 Best Method: {best_method[0]}")
            print(f"   {score_name}: {best_method[1]['score']:.3f}")
        
        if args.show_visualization:
            analyzer.visualize_results(architecture_results, gradient_analysis)
        
        print(f"\n📚 Key Insights:")
        print(f"   • QNNs use parameterized quantum circuits for learning")
        print(f"   • Parameter shift rule enables gradient computation")
        print(f"   • Hybrid models combine quantum and classical processing")
        print(f"   • Circuit expressivity vs trainability trade-off exists")
        
        print(f"\n✅ Quantum neural network analysis completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
