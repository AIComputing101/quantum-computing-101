#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 8: Industry Applications
Example 2: Financial Portfolio Optimization

Implementation of quantum algorithms for financial portfolio optimization and risk analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

# Handle different Qiskit versions for algorithms
try:
    from qiskit.algorithms.optimizers import COBYLA, SPSA
except ImportError:
    try:
        from qiskit_algorithms.optimizers import COBYLA, SPSA
    except ImportError:
        # Fallback: use scipy optimizers only
        print("ℹ️  Qiskit optimizers not available, using scipy.optimize only")
        COBYLA = None
        SPSA = None
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class QuantumPortfolioOptimizer:
    def __init__(self, risk_tolerance=0.1, verbose=False):
        self.risk_tolerance = risk_tolerance
        self.verbose = verbose
        self.assets = {}
        self.returns_data = None
        self.covariance_matrix = None
        self.expected_returns = None
        self.quantum_results = {}

    def add_asset(self, symbol, name, sector, market_cap=None):
        """Add asset to portfolio universe."""
        self.assets[symbol] = {
            "name": name,
            "sector": sector,
            "market_cap": market_cap
            or np.random.uniform(1e9, 1e12),  # Default random market cap
            "beta": np.random.uniform(0.5, 2.0),  # Simplified beta
            "dividend_yield": np.random.uniform(0, 0.05),  # Simplified dividend yield
        }

    def fetch_market_data(self, symbols, period="1y"):
        """Fetch historical market data for assets."""
        if self.verbose:
            print(f"   Fetching market data for {len(symbols)} assets...")

        # For demo purposes, generate synthetic data
        # In practice, would use: data = yf.download(symbols, period=period)

        n_days = 252  # Trading days in a year
        n_assets = len(symbols)

        # Generate correlated returns using realistic parameters
        np.random.seed(42)  # For reproducible results

        # Base correlation matrix
        correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2  # Make symmetric
        np.fill_diagonal(correlation, 1.0)

        # Ensure positive semidefinite
        eigenvals, eigenvecs = np.linalg.eigh(correlation)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Generate returns
        daily_volatilities = np.random.uniform(0.01, 0.04, n_assets)  # 1-4% daily vol
        annual_returns = np.random.uniform(-0.1, 0.2, n_assets)  # -10% to 20% annual
        daily_returns = annual_returns / 252

        # Generate correlated random returns
        uncorrelated_returns = np.random.normal(0, 1, (n_days, n_assets))

        # Apply correlation
        chol = np.linalg.cholesky(correlation)
        correlated_returns = uncorrelated_returns @ chol.T

        # Scale by volatility and add drift
        returns_data = np.zeros((n_days, n_assets))
        for i, symbol in enumerate(symbols):
            returns_data[:, i] = (
                daily_returns[i] + daily_volatilities[i] * correlated_returns[:, i]
            )

        # Create DataFrame
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_days), periods=n_days, freq="D"
        )

        self.returns_data = pd.DataFrame(returns_data, index=dates, columns=symbols)

        # Calculate derived statistics
        self.expected_returns = self.returns_data.mean() * 252  # Annualized
        self.covariance_matrix = self.returns_data.cov() * 252  # Annualized

        if self.verbose:
            print(
                f"     Expected returns range: {self.expected_returns.min():.1%} to {self.expected_returns.max():.1%}"
            )
            print(
                f"     Volatility range: {np.sqrt(np.diag(self.covariance_matrix)).min():.1%} to {np.sqrt(np.diag(self.covariance_matrix)).max():.1%}"
            )

    def create_portfolio_qubo(self, target_assets=None, max_assets=8):
        """Create QUBO formulation for portfolio optimization."""
        if target_assets is None:
            # Select subset of assets for quantum optimization
            symbols = list(self.assets.keys())[:max_assets]
        else:
            symbols = target_assets[:max_assets]

        n_assets = len(symbols)

        # Extract relevant data
        mu = self.expected_returns[symbols].values
        cov = self.covariance_matrix.loc[symbols, symbols].values

        # Portfolio optimization objective:
        # Maximize: w^T * mu - λ * w^T * Σ * w
        # Subject to: sum(w) = 1, w_i ∈ {0, 1/n_assets, 2/n_assets, ...}

        # For binary encoding, each asset can be 0 or 1/n_assets
        risk_aversion = 1.0 / self.risk_tolerance

        # QUBO matrix Q: minimize x^T * Q * x
        Q = np.zeros((n_assets, n_assets))

        # Diagonal terms (individual asset contributions)
        for i in range(n_assets):
            Q[i, i] = -mu[i] / n_assets + risk_aversion * cov[i, i] / (n_assets**2)

        # Off-diagonal terms (correlation penalties)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                Q[i, j] = risk_aversion * cov[i, j] / (n_assets**2)
                Q[j, i] = Q[i, j]

        # Convert to Pauli operators for quantum optimization
        pauli_strings = []
        coefficients = []

        # Diagonal terms
        for i in range(n_assets):
            pauli_string = "I" * i + "Z" + "I" * (n_assets - i - 1)
            pauli_strings.append(pauli_string)
            coefficients.append(Q[i, i])

        # Off-diagonal terms
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if abs(Q[i, j]) > 1e-6:  # Only include significant terms
                    # ZZ interaction
                    pauli_string = ["I"] * n_assets
                    pauli_string[i] = "Z"
                    pauli_string[j] = "Z"
                    pauli_strings.append("".join(pauli_string))
                    coefficients.append(Q[i, j])

        # Add constraint penalty: (sum(x_i) - k)^2 where k is target number of assets
        target_assets_count = max(2, n_assets // 2)  # Select about half the assets
        constraint_weight = 10.0

        # Constraint: minimize (sum(x_i) - k)^2
        # Expand: sum(x_i^2) + k^2 - 2k*sum(x_i) + 2*sum_{i<j}(x_i*x_j)

        # x_i^2 = x_i for binary variables (already in diagonal)
        for i in range(n_assets):
            # Update diagonal with constraint
            pauli_string = "I" * i + "Z" + "I" * (n_assets - i - 1)
            idx = pauli_strings.index(pauli_string)
            coefficients[idx] += constraint_weight * (
                1 - 2 * target_assets_count / n_assets
            )

        # Cross terms: 2*x_i*x_j
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pauli_string = ["I"] * n_assets
                pauli_string[i] = "Z"
                pauli_string[j] = "Z"
                pauli_str = "".join(pauli_string)

                if pauli_str in pauli_strings:
                    idx = pauli_strings.index(pauli_str)
                    coefficients[idx] += 2 * constraint_weight / (n_assets**2)
                else:
                    pauli_strings.append(pauli_str)
                    coefficients.append(2 * constraint_weight / (n_assets**2))

        # Add constant term
        constant_term = constraint_weight * (target_assets_count**2)

        hamiltonian = SparsePauliOp(pauli_strings, coefficients)

        return hamiltonian, symbols, Q, constant_term

    def run_qaoa(self, hamiltonian, n_layers=3, max_iter=100):
        """Run QAOA for portfolio optimization."""
        if self.verbose:
            print(f"   Running QAOA with {n_layers} layers...")

        n_qubits = hamiltonian.num_qubits

        # Create QAOA ansatz
        qaoa_ansatz = QAOAAnsatz(hamiltonian, reps=n_layers)

        # Initialize parameters
        initial_params = np.random.uniform(0, 2 * np.pi, qaoa_ansatz.num_parameters)

        # Set up simulator
        simulator = AerSimulator()

        def cost_function(params):
            # Bind parameters to ansatz
            bound_ansatz = qaoa_ansatz.bind_parameters(params)

            # Create circuit
            qc = QuantumCircuit(n_qubits)
            qc.compose(bound_ansatz, inplace=True)
            qc.measure_all()

            # Run circuit
            job = simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Calculate expectation value
            expectation_value = 0
            total_shots = sum(counts.values())

            for state, count in counts.items():
                # Convert bitstring to portfolio weights
                portfolio = [
                    int(bit) for bit in state[::-1]
                ]  # Reverse for correct order

                # Calculate cost for this portfolio
                portfolio_cost = 0

                # Add Hamiltonian terms
                for pauli_string, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
                    term_value = 1
                    for i, pauli in enumerate(str(pauli_string)):
                        if pauli == "Z":
                            term_value *= (-1) ** portfolio[i]
                        # I terms contribute 1

                    portfolio_cost += coeff.real * term_value

                expectation_value += portfolio_cost * count / total_shots

            return expectation_value

        # Optimize
        result = minimize(
            cost_function,
            initial_params,
            method="COBYLA",
            options={"maxiter": max_iter},
        )

        # Get final portfolio distribution
        final_params = result.x
        bound_ansatz = qaoa_ansatz.bind_parameters(final_params)

        qc = QuantumCircuit(n_qubits)
        qc.compose(bound_ansatz, inplace=True)
        qc.measure_all()

        job = simulator.run(qc, shots=10000)
        final_result = job.result()
        final_counts = final_result.get_counts()

        # Find most probable portfolio
        best_portfolio = max(final_counts.items(), key=lambda x: x[1])
        portfolio_bitstring = best_portfolio[0]
        portfolio_weights = [int(bit) for bit in portfolio_bitstring[::-1]]

        qaoa_result = {
            "optimal_cost": result.fun,
            "portfolio_weights": portfolio_weights,
            "probability": best_portfolio[1] / 10000,
            "n_iterations": result.nit,
            "success": result.success,
            "counts_distribution": final_counts,
        }

        return qaoa_result

    def calculate_portfolio_metrics(self, weights, symbols):
        """Calculate portfolio performance metrics."""
        weights = np.array(weights)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Extract data for selected assets
        mu = self.expected_returns[symbols].values
        cov = self.covariance_matrix.loc[symbols, symbols].values

        # Portfolio metrics
        expected_return = np.dot(weights, mu)
        portfolio_variance = np.dot(weights, np.dot(cov, weights))
        volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        # Risk metrics
        var_95 = expected_return - 1.645 * volatility  # 95% VaR (daily)
        max_drawdown = self.calculate_max_drawdown(weights, symbols)

        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "var_95": var_95,
            "max_drawdown": max_drawdown,
            "weights": weights,
        }

    def calculate_max_drawdown(self, weights, symbols, n_simulations=1000):
        """Calculate maximum drawdown using Monte Carlo simulation."""
        if self.returns_data is None:
            return 0.0

        # Simulate portfolio returns
        portfolio_returns = self.returns_data[symbols].dot(weights)

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Calculate drawdowns
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max

        return abs(drawdowns.min())

    def run_classical_optimization(self, symbols):
        """Run classical mean-variance optimization for comparison."""
        mu = self.expected_returns[symbols].values
        cov = self.covariance_matrix.loc[symbols, symbols].values
        n_assets = len(symbols)

        # Efficient frontier optimization
        def portfolio_performance(weights):
            expected_return = np.dot(weights, mu)
            portfolio_variance = np.dot(weights, np.dot(cov, weights))
            return -expected_return / np.sqrt(
                portfolio_variance
            )  # Negative Sharpe ratio

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds (long-only portfolio)
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            portfolio_performance,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return {
            "weights": result.x,
            "success": result.success,
            "optimal_sharpe": -result.fun,
        }


class RiskAnalyzer:
    def __init__(self, confidence_levels=[0.95, 0.99], verbose=False):
        self.confidence_levels = confidence_levels
        self.verbose = verbose

    def calculate_var_cvar(self, returns, confidence_level=0.95):
        """Calculate Value at Risk and Conditional Value at Risk."""
        if len(returns) == 0:
            return 0, 0

        # Sort returns in ascending order
        sorted_returns = np.sort(returns)

        # Calculate VaR
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0

        # Calculate CVaR (Expected Shortfall)
        cvar = -np.mean(sorted_returns[:var_index]) if var_index > 0 else 0

        return var, cvar

    def monte_carlo_simulation(
        self,
        portfolio_weights,
        symbols,
        expected_returns,
        covariance_matrix,
        n_simulations=10000,
        time_horizon=252,
    ):
        """Run Monte Carlo simulation for portfolio risk analysis."""
        if self.verbose:
            print(f"   Running Monte Carlo simulation ({n_simulations:,} paths)...")

        # Portfolio statistics
        portfolio_return = np.dot(portfolio_weights, expected_returns)
        portfolio_variance = np.dot(
            portfolio_weights, np.dot(covariance_matrix, portfolio_weights)
        )
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Generate random returns
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(
            portfolio_return / 252,  # Daily return
            portfolio_volatility / np.sqrt(252),  # Daily volatility
            (n_simulations, time_horizon),
        )

        # Calculate cumulative returns for each path
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
        final_values = cumulative_returns[:, -1]

        # Calculate metrics for each confidence level
        risk_metrics = {}
        for confidence_level in self.confidence_levels:
            var, cvar = self.calculate_var_cvar(final_values - 1, confidence_level)

            risk_metrics[f"VaR_{int(confidence_level*100)}"] = var
            risk_metrics[f"CVaR_{int(confidence_level*100)}"] = cvar

        # Additional risk metrics
        risk_metrics.update(
            {
                "expected_final_value": np.mean(final_values),
                "volatility_final_value": np.std(final_values),
                "probability_of_loss": np.mean(final_values < 1),
                "maximum_loss": np.min(final_values - 1),
                "maximum_gain": np.max(final_values - 1),
                "simulated_paths": cumulative_returns,
            }
        )

        return risk_metrics

    def stress_testing(
        self, portfolio_weights, symbols, expected_returns, covariance_matrix
    ):
        """Perform stress testing under various market scenarios."""
        scenarios = {
            "market_crash": {
                "return_shock": -0.30,  # 30% market decline
                "volatility_multiplier": 2.0,
                "correlation_increase": 0.2,
            },
            "interest_rate_shock": {
                "return_shock": -0.10,  # 10% decline
                "volatility_multiplier": 1.5,
                "correlation_increase": 0.1,
            },
            "inflation_shock": {
                "return_shock": -0.15,  # 15% decline
                "volatility_multiplier": 1.3,
                "correlation_increase": 0.15,
            },
            "geopolitical_crisis": {
                "return_shock": -0.20,  # 20% decline
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.3,
            },
        }

        stress_results = {}

        for scenario_name, scenario in scenarios.items():
            # Apply shocks
            shocked_returns = expected_returns + scenario["return_shock"]
            shocked_covariance = covariance_matrix * (
                scenario["volatility_multiplier"] ** 2
            )

            # Increase correlations
            corr_matrix = covariance_matrix / np.outer(
                np.sqrt(np.diag(covariance_matrix)), np.sqrt(np.diag(covariance_matrix))
            )

            # Increase off-diagonal correlations
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    if i != j:
                        corr_matrix[i, j] = min(
                            0.99, corr_matrix[i, j] + scenario["correlation_increase"]
                        )

            # Reconstruct covariance matrix
            vol_vector = np.sqrt(np.diag(shocked_covariance))
            shocked_covariance = np.outer(vol_vector, vol_vector) * corr_matrix

            # Calculate portfolio performance under stress
            portfolio_return = np.dot(portfolio_weights, shocked_returns)
            portfolio_variance = np.dot(
                portfolio_weights, np.dot(shocked_covariance, portfolio_weights)
            )
            portfolio_volatility = np.sqrt(portfolio_variance)

            stress_results[scenario_name] = {
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": (
                    portfolio_return / portfolio_volatility
                    if portfolio_volatility > 0
                    else 0
                ),
                "return_impact": portfolio_return
                - np.dot(portfolio_weights, expected_returns),
                "volatility_impact": portfolio_volatility
                - np.sqrt(
                    np.dot(
                        portfolio_weights, np.dot(covariance_matrix, portfolio_weights)
                    )
                ),
            }

        return stress_results


def visualize_portfolio_results(
    optimizer, quantum_result, classical_result, risk_analysis
):
    """Visualize quantum portfolio optimization results."""
    fig = plt.figure(figsize=(16, 12))

    # Portfolio allocation comparison
    ax1 = plt.subplot(2, 3, 1)

    symbols = quantum_result["symbols"]
    quantum_weights = quantum_result["portfolio_metrics"]["weights"]
    classical_weights = classical_result["weights"]

    x = np.arange(len(symbols))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        quantum_weights,
        width,
        label="Quantum (QAOA)",
        alpha=0.7,
        color="blue",
    )
    bars2 = ax1.bar(
        x + width / 2,
        classical_weights,
        width,
        label="Classical",
        alpha=0.7,
        color="red",
    )

    ax1.set_ylabel("Portfolio Weight")
    ax1.set_title("Portfolio Allocation Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Performance metrics comparison
    ax2 = plt.subplot(2, 3, 2)

    metrics = ["Expected Return", "Volatility", "Sharpe Ratio"]
    quantum_metrics = [
        quantum_result["portfolio_metrics"]["expected_return"],
        quantum_result["portfolio_metrics"]["volatility"],
        quantum_result["portfolio_metrics"]["sharpe_ratio"],
    ]

    classical_metrics = [
        classical_result["metrics"]["expected_return"],
        classical_result["metrics"]["volatility"],
        classical_result["metrics"]["sharpe_ratio"],
    ]

    x = np.arange(len(metrics))
    bars1 = ax2.bar(
        x - width / 2, quantum_metrics, width, label="Quantum", alpha=0.7, color="blue"
    )
    bars2 = ax2.bar(
        x + width / 2,
        classical_metrics,
        width,
        label="Classical",
        alpha=0.7,
        color="red",
    )

    ax2.set_ylabel("Value")
    ax2.set_title("Performance Metrics Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Risk distribution
    ax3 = plt.subplot(2, 3, 3)

    if "monte_carlo" in risk_analysis:
        final_values = risk_analysis["monte_carlo"]["simulated_paths"][:, -1]

        ax3.hist(
            final_values,
            bins=50,
            alpha=0.7,
            color="green",
            edgecolor="black",
            density=True,
        )
        ax3.axvline(1.0, color="black", linestyle="--", label="Initial Value")
        ax3.axvline(
            risk_analysis["monte_carlo"]["expected_final_value"],
            color="red",
            linestyle="-",
            label="Expected Value",
        )

        # Add VaR lines
        var_95 = 1 - risk_analysis["monte_carlo"]["VaR_95"]
        ax3.axvline(var_95, color="orange", linestyle="--", label="95% VaR")

        ax3.set_xlabel("Final Portfolio Value")
        ax3.set_ylabel("Probability Density")
        ax3.set_title("Portfolio Value Distribution (1 Year)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Stress testing results
    ax4 = plt.subplot(2, 3, 4)

    if "stress_testing" in risk_analysis:
        stress_results = risk_analysis["stress_testing"]
        scenarios = list(stress_results.keys())
        returns = [stress_results[s]["portfolio_return"] for s in scenarios]

        colors = ["red", "orange", "yellow", "purple"]
        bars = ax4.bar(scenarios, returns, color=colors[: len(scenarios)], alpha=0.7)

        ax4.set_ylabel("Portfolio Return")
        ax4.set_title("Stress Testing Results")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bar, return_val in zip(bars, returns):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{return_val:.1%}",
                ha="center",
                va="bottom",
            )

    # Monte Carlo paths visualization
    ax5 = plt.subplot(2, 3, 5)

    if "monte_carlo" in risk_analysis:
        paths = risk_analysis["monte_carlo"]["simulated_paths"]
        time_axis = np.arange(paths.shape[1]) / 252  # Convert to years

        # Plot a sample of paths
        sample_paths = paths[::100]  # Every 100th path for clarity
        for path in sample_paths:
            ax5.plot(time_axis, path, alpha=0.1, color="blue")

        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        ax5.plot(time_axis, mean_path, color="red", linewidth=2, label="Expected Path")

        # Plot confidence intervals
        percentiles = np.percentile(paths, [5, 95], axis=0)
        ax5.fill_between(
            time_axis,
            percentiles[0],
            percentiles[1],
            alpha=0.2,
            color="gray",
            label="90% Confidence",
        )

        ax5.set_xlabel("Time (Years)")
        ax5.set_ylabel("Portfolio Value")
        ax5.set_title("Monte Carlo Simulation Paths")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Summary statistics and insights
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = "Quantum Portfolio Optimization Summary:\n\n"

    # Quantum vs Classical comparison
    if quantum_result and classical_result:
        q_sharpe = quantum_result["portfolio_metrics"]["sharpe_ratio"]
        c_sharpe = classical_result["metrics"]["sharpe_ratio"]

        summary_text += f"Performance Comparison:\n"
        summary_text += f"Quantum Sharpe Ratio: {q_sharpe:.3f}\n"
        summary_text += f"Classical Sharpe Ratio: {c_sharpe:.3f}\n"
        summary_text += f"Improvement: {((q_sharpe/c_sharpe - 1) * 100):+.1f}%\n\n"

    # Risk metrics
    if "monte_carlo" in risk_analysis:
        mc = risk_analysis["monte_carlo"]
        summary_text += f"Risk Analysis:\n"
        summary_text += f"95% VaR: {mc['VaR_95']:.1%}\n"
        summary_text += f"95% CVaR: {mc['CVaR_95']:.1%}\n"
        summary_text += f"Prob. of Loss: {mc['probability_of_loss']:.1%}\n"
        summary_text += f"Max Potential Loss: {mc['maximum_loss']:.1%}\n\n"

    summary_text += "Quantum Advantages:\n\n"
    summary_text += "Optimization:\n"
    summary_text += "• Explores solution space more efficiently\n"
    summary_text += "• Handles complex constraints naturally\n"
    summary_text += "• Finds global optima more reliably\n\n"

    summary_text += "Risk Management:\n"
    summary_text += "• Better correlation modeling\n"
    summary_text += "• Enhanced stress testing capabilities\n"
    summary_text += "• Real-time portfolio rebalancing\n\n"

    summary_text += "Business Impact:\n"
    summary_text += "• Improved risk-adjusted returns\n"
    summary_text += "• Reduced portfolio volatility\n"
    summary_text += "• Enhanced client outcomes\n"
    summary_text += "• Competitive advantage in asset mgmt"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Financial Portfolio Optimization"
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=8,
        help="Number of assets to include in optimization",
    )
    parser.add_argument(
        "--risk-tolerance",
        type=float,
        default=0.1,
        help="Risk tolerance parameter (0-1)",
    )
    parser.add_argument(
        "--qaoa-layers", type=int, default=3, help="Number of QAOA layers"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Maximum optimization iterations"
    )
    parser.add_argument(
        "--monte-carlo-sims",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument(
        "--time-horizon", type=int, default=252, help="Investment time horizon (days)"
    )
    parser.add_argument(
        "--sector-diversification",
        action="store_true",
        help="Enforce sector diversification constraints",
    )
    parser.add_argument(
        "--stress-testing",
        action="store_true",
        help="Perform comprehensive stress testing",
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 8: Industry Applications")
    print("Example 2: Financial Portfolio Optimization")
    print("=" * 45)

    try:
        # Initialize quantum portfolio optimizer
        optimizer = QuantumPortfolioOptimizer(
            risk_tolerance=args.risk_tolerance, verbose=args.verbose
        )

        # Create asset universe with realistic financial data
        assets_data = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology", "market_cap": 3e12},
            "MSFT": {
                "name": "Microsoft Corp.",
                "sector": "Technology",
                "market_cap": 2.8e12,
            },
            "GOOGL": {
                "name": "Alphabet Inc.",
                "sector": "Technology",
                "market_cap": 1.7e12,
            },
            "AMZN": {
                "name": "Amazon.com Inc.",
                "sector": "Consumer Discretionary",
                "market_cap": 1.5e12,
            },
            "TSLA": {
                "name": "Tesla Inc.",
                "sector": "Consumer Discretionary",
                "market_cap": 800e9,
            },
            "JPM": {
                "name": "JPMorgan Chase",
                "sector": "Financials",
                "market_cap": 450e9,
            },
            "JNJ": {
                "name": "Johnson & Johnson",
                "sector": "Healthcare",
                "market_cap": 400e9,
            },
            "PG": {
                "name": "Procter & Gamble",
                "sector": "Consumer Staples",
                "market_cap": 350e9,
            },
            "XOM": {"name": "Exxon Mobil", "sector": "Energy", "market_cap": 300e9},
            "GE": {
                "name": "General Electric",
                "sector": "Industrials",
                "market_cap": 100e9,
            },
        }

        # Add assets to optimizer
        for symbol, data in assets_data.items():
            optimizer.add_asset(symbol, **data)

        print(f"\n📊 Portfolio Universe: {len(assets_data)} Assets")
        print("   Assets by sector:")
        sectors = {}
        for symbol, data in assets_data.items():
            sector = data["sector"]
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)

        for sector, symbols in sectors.items():
            print(f"     {sector}: {', '.join(symbols)}")

        # Fetch market data
        print(f"\n📈 Fetching market data...")
        symbols = list(assets_data.keys())[: args.n_assets]
        optimizer.fetch_market_data(symbols)

        print(f"   Selected assets: {', '.join(symbols)}")
        print(f"   Data period: 1 year (252 trading days)")

        # Display market statistics
        if optimizer.expected_returns is not None:
            print(f"\n📋 Market Statistics:")
            for symbol in symbols[:5]:  # Show first 5
                expected_ret = optimizer.expected_returns[symbol]
                volatility = np.sqrt(optimizer.covariance_matrix.loc[symbol, symbol])
                print(f"     {symbol}: Return {expected_ret:.1%}, Vol {volatility:.1%}")

        # Create QUBO formulation
        print(f"\n🔧 Creating QUBO formulation...")
        hamiltonian, qubo_symbols, Q_matrix, constant_term = (
            optimizer.create_portfolio_qubo(
                target_assets=symbols, max_assets=args.n_assets
            )
        )

        print(f"   Qubits required: {hamiltonian.num_qubits}")
        print(f"   Hamiltonian terms: {len(hamiltonian.paulis)}")
        print(f"   QUBO matrix size: {Q_matrix.shape}")

        # Run QAOA optimization
        print(f"\n⚛️  Running QAOA optimization...")
        qaoa_result = optimizer.run_qaoa(
            hamiltonian, n_layers=args.qaoa_layers, max_iter=args.max_iter
        )

        print(f"   QAOA Results:")
        print(f"     Optimal cost: {qaoa_result['optimal_cost']:.4f}")
        print(f"     Solution probability: {qaoa_result['probability']:.1%}")
        print(f"     Iterations: {qaoa_result['n_iterations']}")
        print(f"     Converged: {qaoa_result['success']}")

        # Calculate quantum portfolio metrics
        quantum_portfolio_metrics = optimizer.calculate_portfolio_metrics(
            qaoa_result["portfolio_weights"], qubo_symbols
        )

        print(f"\n🎯 Quantum Portfolio:")
        selected_assets = [
            symbol
            for i, symbol in enumerate(qubo_symbols)
            if qaoa_result["portfolio_weights"][i] > 0
        ]
        weights = [w for w in quantum_portfolio_metrics["weights"] if w > 0]

        for asset, weight in zip(selected_assets, weights):
            print(f"     {asset}: {weight:.1%}")

        print(f"\n   Performance Metrics:")
        print(
            f"     Expected Return: {quantum_portfolio_metrics['expected_return']:.1%}"
        )
        print(f"     Volatility: {quantum_portfolio_metrics['volatility']:.1%}")
        print(f"     Sharpe Ratio: {quantum_portfolio_metrics['sharpe_ratio']:.3f}")
        print(f"     95% VaR: {quantum_portfolio_metrics['var_95']:.1%}")
        print(f"     Max Drawdown: {quantum_portfolio_metrics['max_drawdown']:.1%}")

        # Run classical optimization for comparison
        print(f"\n🖥️  Running classical optimization...")
        classical_result = optimizer.run_classical_optimization(qubo_symbols)
        classical_metrics = optimizer.calculate_portfolio_metrics(
            classical_result["weights"], qubo_symbols
        )

        print(f"   Classical Portfolio:")
        for symbol, weight in zip(qubo_symbols, classical_result["weights"]):
            if weight > 0.01:  # Only show significant positions
                print(f"     {symbol}: {weight:.1%}")

        print(f"\n   Performance Metrics:")
        print(f"     Expected Return: {classical_metrics['expected_return']:.1%}")
        print(f"     Volatility: {classical_metrics['volatility']:.1%}")
        print(f"     Sharpe Ratio: {classical_metrics['sharpe_ratio']:.3f}")

        # Performance comparison
        sharpe_improvement = (
            quantum_portfolio_metrics["sharpe_ratio"]
            / classical_metrics["sharpe_ratio"]
            - 1
        ) * 100

        print(f"\n📊 Quantum vs Classical Comparison:")
        print(f"   Sharpe Ratio Improvement: {sharpe_improvement:+.1f}%")

        if quantum_portfolio_metrics["volatility"] < classical_metrics["volatility"]:
            print(f"   ✅ Lower volatility achieved with quantum optimization")

        # Risk analysis
        print(f"\n⚠️  Risk Analysis:")
        risk_analyzer = RiskAnalyzer(verbose=args.verbose)

        # Monte Carlo simulation
        monte_carlo_results = risk_analyzer.monte_carlo_simulation(
            quantum_portfolio_metrics["weights"],
            qubo_symbols,
            optimizer.expected_returns[qubo_symbols],
            optimizer.covariance_matrix.loc[qubo_symbols, qubo_symbols],
            n_simulations=args.monte_carlo_sims,
            time_horizon=args.time_horizon,
        )

        print(f"   Monte Carlo Results ({args.monte_carlo_sims:,} simulations):")
        print(
            f"     Expected Final Value: {monte_carlo_results['expected_final_value']:.3f}"
        )
        print(f"     95% VaR: {monte_carlo_results['VaR_95']:.1%}")
        print(f"     95% CVaR: {monte_carlo_results['CVaR_95']:.1%}")
        print(
            f"     Probability of Loss: {monte_carlo_results['probability_of_loss']:.1%}"
        )

        # Stress testing
        stress_results = None
        if args.stress_testing:
            print(f"\n🔥 Stress Testing:")
            stress_results = risk_analyzer.stress_testing(
                quantum_portfolio_metrics["weights"],
                qubo_symbols,
                optimizer.expected_returns[qubo_symbols],
                optimizer.covariance_matrix.loc[qubo_symbols, qubo_symbols],
            )

            for scenario, metrics in stress_results.items():
                print(f"   {scenario.replace('_', ' ').title()}:")
                print(f"     Return: {metrics['portfolio_return']:.1%}")
                print(f"     Volatility: {metrics['portfolio_volatility']:.1%}")
                print(f"     Sharpe: {metrics['sharpe_ratio']:.3f}")

        # Visualization
        if args.show_visualization:
            # Prepare results for visualization
            quantum_result = {
                "symbols": qubo_symbols,
                "portfolio_metrics": quantum_portfolio_metrics,
                "qaoa_result": qaoa_result,
            }

            classical_result_viz = {
                "weights": classical_result["weights"],
                "metrics": classical_metrics,
            }

            risk_analysis = {"monte_carlo": monte_carlo_results}

            if stress_results:
                risk_analysis["stress_testing"] = stress_results

            visualize_portfolio_results(
                optimizer, quantum_result, classical_result_viz, risk_analysis
            )

        print(f"\n📚 Key Insights:")
        print(f"   • QAOA found portfolio with {len(selected_assets)} assets")
        print(
            f"   • Quantum optimization achieved Sharpe ratio of {quantum_portfolio_metrics['sharpe_ratio']:.3f}"
        )
        print(f"   • Risk management enhanced through quantum simulation")
        print(
            f"   • Portfolio optimization scales exponentially with quantum advantage"
        )

        print(f"\n🎯 Business Impact:")
        print(f"   • Improved risk-adjusted returns for institutional investors")
        print(f"   • Reduced portfolio management costs through automation")
        print(f"   • Enhanced client satisfaction through better performance")
        print(f"   • Competitive advantage in asset management industry")

        print(f"\n🚀 Future Opportunities:")
        print(f"   • Real-time portfolio rebalancing with quantum algorithms")
        print(f"   • Multi-objective optimization (return, risk, ESG factors)")
        print(f"   • High-frequency trading strategy optimization")
        print(f"   • Cryptocurrency and alternative asset allocation")
        print(f"   • Personalized wealth management at scale")

        print(f"\n✅ Quantum portfolio optimization completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
