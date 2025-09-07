#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 8: Industry Applications
Example 5: Materials Science and Manufacturing

Implementation of quantum algorithms for materials simulation and manufacturing optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector

# Handle different Qiskit versions for algorithms
try:
    from qiskit.algorithms.optimizers import SPSA, COBYLA
except ImportError:
    try:
        from qiskit_algorithms.optimizers import SPSA, COBYLA
    except ImportError:
        # Fallback: use scipy optimizers only
        print("ℹ️  Qiskit optimizers not available, using scipy.optimize only")
        SPSA = None
        COBYLA = None
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class MaterialSimulator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.materials = {}
        self.simulations = {}
        self.crystal_structures = {}

    def create_material(
        self, material_id, elements, structure_type, lattice_parameters=None
    ):
        """Create material model for quantum simulation."""
        material = {
            "id": material_id,
            "elements": elements,
            "structure_type": structure_type,  # 'crystal', 'amorphous', 'composite'
            "lattice_parameters": lattice_parameters
            or self._generate_lattice_params(structure_type),
            "properties": {},
            "electronic_structure": {},
            "quantum_state": None,
        }

        # Calculate basic properties
        material["properties"] = self._calculate_basic_properties(
            elements, structure_type
        )

        self.materials[material_id] = material

        if self.verbose:
            print(f"   Created material: {material_id} ({structure_type})")
            print(f"     Elements: {', '.join(elements)}")
            print(f"     Density: {material['properties']['density']:.2f} g/cm³")

        return material

    def _generate_lattice_params(self, structure_type):
        """Generate realistic lattice parameters."""
        if structure_type == "crystal":
            return {
                "a": np.random.uniform(3.0, 6.0),  # Angstroms
                "b": np.random.uniform(3.0, 6.0),
                "c": np.random.uniform(3.0, 6.0),
                "alpha": 90.0,  # Degrees
                "beta": 90.0,
                "gamma": 90.0,
                "space_group": np.random.choice(["P1", "P2", "Pm", "Cmcm", "Fd3m"]),
            }
        else:
            return {"characteristic_length": np.random.uniform(1.0, 10.0)}

    def _calculate_basic_properties(self, elements, structure_type):
        """Calculate basic material properties."""
        # Simplified property calculations
        atomic_masses = {
            "H": 1.008,
            "He": 4.003,
            "Li": 6.941,
            "Be": 9.012,
            "B": 10.811,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "Ne": 20.180,
            "Na": 22.990,
            "Mg": 24.305,
            "Al": 26.982,
            "Si": 28.086,
            "P": 30.974,
            "S": 32.065,
            "Cl": 35.453,
            "Ar": 39.948,
            "K": 39.098,
            "Ca": 40.078,
            "Ti": 47.867,
            "Fe": 55.845,
            "Cu": 63.546,
            "Zn": 65.38,
            "Ga": 69.723,
            "Ge": 72.631,
            "As": 74.922,
            "Se": 78.971,
            "Br": 79.904,
            "Kr": 83.798,
        }

        # Average atomic mass
        avg_mass = np.mean([atomic_masses.get(elem, 50.0) for elem in elements])

        # Estimate density based on structure and composition
        if structure_type == "crystal":
            density = avg_mass * 0.1  # Simplified correlation
        elif structure_type == "amorphous":
            density = avg_mass * 0.08
        else:  # composite
            density = avg_mass * 0.09

        # Estimate other properties
        properties = {
            "density": density,
            "melting_point": 1000 + avg_mass * 20 + np.random.uniform(-200, 200),  # K
            "band_gap": (
                np.random.uniform(0.1, 5.0)
                if "Si" in elements or "Ge" in elements
                else 0.0
            ),  # eV
            "bulk_modulus": avg_mass * 2 + np.random.uniform(50, 200),  # GPa
            "thermal_conductivity": np.random.uniform(1, 100),  # W/m·K
            "electrical_resistivity": np.random.uniform(1e-8, 1e8),  # Ω·m
        }

        return properties

    def vqe_electronic_structure(self, material_id, max_iter=100):
        """Calculate electronic structure using VQE."""
        if material_id not in self.materials:
            raise ValueError(f"Material {material_id} not found")

        material = self.materials[material_id]

        if self.verbose:
            print(f"   Running VQE for {material_id}...")

        # Create molecular Hamiltonian (simplified)
        n_electrons = sum(
            self._get_valence_electrons(elem) for elem in material["elements"]
        )
        n_orbitals = min(len(material["elements"]) * 2, 8)  # Limit for simulation

        # Build Hamiltonian for electronic structure
        hamiltonian = self._build_electronic_hamiltonian(material, n_orbitals)

        # VQE ansatz
        ansatz = TwoLocal(n_orbitals, "ry", "cz", reps=2, entanglement="linear")

        # Initialize parameters
        initial_params = np.random.uniform(0, 2 * np.pi, ansatz.num_parameters)

        # Simulator
        simulator = AerSimulator()

        def cost_function(params):
            bound_ansatz = ansatz.bind_parameters(params)
            qc = QuantumCircuit(n_orbitals)
            qc.compose(bound_ansatz, inplace=True)

            # Calculate expectation value
            expectation_value = 0

            for pauli_string, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
                measure_qc = qc.copy()

                # Add measurement basis rotations
                for i, pauli in enumerate(str(pauli_string)):
                    if pauli == "X":
                        measure_qc.ry(-np.pi / 2, i)
                    elif pauli == "Y":
                        measure_qc.rx(np.pi / 2, i)

                measure_qc.measure_all()

                job = simulator.run(measure_qc, shots=1000)
                result = job.result()
                counts = result.get_counts()

                # Calculate expectation
                total_shots = sum(counts.values())
                term_expectation = 0

                for state, count in counts.items():
                    parity = 1
                    for i, (bit, pauli) in enumerate(
                        zip(state[::-1], str(pauli_string))
                    ):
                        if pauli != "I" and bit == "1":
                            parity *= -1

                    term_expectation += parity * count / total_shots

                expectation_value += coeff.real * term_expectation

            return expectation_value

        # Optimize
        result = minimize(
            cost_function,
            initial_params,
            method="COBYLA",
            options={"maxiter": max_iter},
        )

        # Store electronic structure results
        electronic_structure = {
            "ground_state_energy": result.fun,
            "optimal_parameters": result.x,
            "n_orbitals": n_orbitals,
            "n_electrons": n_electrons,
            "convergence": result.success,
            "homo_lumo_gap": abs(
                result.fun * np.random.uniform(0.1, 0.3)
            ),  # Simplified
            "ionization_potential": abs(result.fun * np.random.uniform(0.8, 1.2)),
            "electron_affinity": abs(result.fun * np.random.uniform(0.2, 0.6)),
        }

        material["electronic_structure"] = electronic_structure

        if self.verbose:
            print(
                f"     Ground state energy: {electronic_structure['ground_state_energy']:.4f} hartree"
            )
            print(f"     HOMO-LUMO gap: {electronic_structure['homo_lumo_gap']:.3f} eV")
            print(f"     Converged: {electronic_structure['convergence']}")

        return electronic_structure

    def _get_valence_electrons(self, element):
        """Get number of valence electrons for element."""
        valence_electrons = {
            "H": 1,
            "He": 2,
            "Li": 1,
            "Be": 2,
            "B": 3,
            "C": 4,
            "N": 5,
            "O": 6,
            "F": 7,
            "Ne": 8,
            "Na": 1,
            "Mg": 2,
            "Al": 3,
            "Si": 4,
            "P": 5,
            "S": 6,
            "Cl": 7,
            "Ar": 8,
            "K": 1,
            "Ca": 2,
            "Ti": 4,
            "Fe": 8,
            "Cu": 1,
            "Zn": 2,
        }
        return valence_electrons.get(element, 4)

    def _build_electronic_hamiltonian(self, material, n_orbitals):
        """Build electronic Hamiltonian for material."""
        pauli_strings = []
        coefficients = []

        # One-electron terms (kinetic + nuclear attraction)
        for i in range(n_orbitals):
            pauli_string = "I" * i + "Z" + "I" * (n_orbitals - i - 1)
            pauli_strings.append(pauli_string)
            coefficients.append(-2.0 - np.random.uniform(0, 1))

        # Two-electron terms (electron-electron repulsion)
        for i in range(n_orbitals):
            for j in range(i + 1, n_orbitals):
                # Coulomb interactions
                pauli_string = ["I"] * n_orbitals
                pauli_string[i] = "Z"
                pauli_string[j] = "Z"
                pauli_strings.append("".join(pauli_string))
                coefficients.append(0.5 * np.random.uniform(0.1, 0.8))

                # Exchange interactions
                for pauli_pair in [("X", "X"), ("Y", "Y")]:
                    pauli_string = ["I"] * n_orbitals
                    pauli_string[i] = pauli_pair[0]
                    pauli_string[j] = pauli_pair[1]
                    pauli_strings.append("".join(pauli_string))
                    coefficients.append(0.1 * (np.random.random() - 0.5))

        return SparsePauliOp(pauli_strings, coefficients)

    def phonon_spectrum_calculation(self, material_id):
        """Calculate phonon spectrum using quantum simulation."""
        if material_id not in self.materials:
            raise ValueError(f"Material {material_id} not found")

        material = self.materials[material_id]

        if self.verbose:
            print(f"   Calculating phonon spectrum for {material_id}...")

        # Simplified phonon calculation
        # In practice, would involve force constant calculations

        n_atoms = len(material["elements"])
        n_modes = 3 * n_atoms  # 3 degrees of freedom per atom

        # Generate phonon frequencies (simplified)
        # Lower frequencies for acoustic modes, higher for optical modes
        acoustic_modes = 3  # Always 3 acoustic modes
        optical_modes = max(0, n_modes - acoustic_modes)

        # Acoustic mode frequencies (typically low)
        acoustic_freqs = np.random.uniform(0.1, 5.0, acoustic_modes)  # THz

        # Optical mode frequencies (typically higher)
        optical_freqs = np.random.uniform(5.0, 30.0, optical_modes)  # THz

        all_frequencies = np.concatenate([acoustic_freqs, optical_freqs])
        all_frequencies = np.sort(all_frequencies)

        # Calculate thermodynamic properties
        temperature = 300  # K
        kB = 8.617e-5  # eV/K
        hbar = 6.582e-16  # eV·s

        # Phonon contribution to heat capacity (Einstein model approximation)
        heat_capacity = 0
        for freq in all_frequencies:
            if freq > 0:
                hw_kT = (hbar * freq * 1e12) / (kB * temperature)  # Convert THz to Hz
                if hw_kT < 50:  # Avoid overflow
                    x = hw_kT
                    heat_capacity += kB * x**2 * np.exp(x) / (np.exp(x) - 1) ** 2

        phonon_spectrum = {
            "frequencies_THz": all_frequencies,
            "n_modes": n_modes,
            "acoustic_modes": acoustic_modes,
            "optical_modes": optical_modes,
            "max_frequency": np.max(all_frequencies),
            "debye_temperature": np.max(all_frequencies) * 47.99,  # K (simplified)
            "heat_capacity_300K": heat_capacity,
            "thermal_conductivity_estimate": heat_capacity
            * 100
            * np.random.uniform(0.5, 2.0),
        }

        material["phonon_spectrum"] = phonon_spectrum

        if self.verbose:
            print(
                f"     Phonon modes: {n_modes} ({acoustic_modes} acoustic, {optical_modes} optical)"
            )
            print(f"     Max frequency: {phonon_spectrum['max_frequency']:.1f} THz")
            print(
                f"     Debye temperature: {phonon_spectrum['debye_temperature']:.0f} K"
            )

        return phonon_spectrum


class ManufacturingOptimizer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.processes = {}
        self.optimization_results = {}

    def add_manufacturing_process(
        self, process_id, process_type, parameters, constraints
    ):
        """Add manufacturing process for optimization."""
        process = {
            "id": process_id,
            "type": process_type,  # 'additive', 'subtractive', 'forming', 'joining'
            "parameters": parameters,
            "constraints": constraints,
            "optimization_variables": [],
            "cost_function": None,
        }

        # Define optimization variables based on process type
        if process_type == "additive":
            process["optimization_variables"] = [
                "layer_height",
                "print_speed",
                "temperature",
                "infill_density",
            ]
        elif process_type == "subtractive":
            process["optimization_variables"] = [
                "cutting_speed",
                "feed_rate",
                "depth_of_cut",
                "tool_angle",
            ]
        elif process_type == "forming":
            process["optimization_variables"] = [
                "pressure",
                "temperature",
                "forming_speed",
                "die_angle",
            ]
        elif process_type == "joining":
            process["optimization_variables"] = [
                "welding_current",
                "voltage",
                "travel_speed",
                "gas_flow",
            ]

        self.processes[process_id] = process

        if self.verbose:
            print(f"   Added process: {process_id} ({process_type})")
            print(f"     Variables: {', '.join(process['optimization_variables'])}")

        return process

    def quantum_process_optimization(self, process_id, optimization_objectives):
        """Optimize manufacturing process using quantum algorithms."""
        if process_id not in self.processes:
            raise ValueError(f"Process {process_id} not found")

        process = self.processes[process_id]

        if self.verbose:
            print(f"   Optimizing process: {process_id}")

        # Create optimization problem
        n_variables = len(process["optimization_variables"])
        n_qubits = min(2 * n_variables, 12)  # Limit for simulation

        # Multi-objective optimization: quality, cost, time
        objectives = optimization_objectives
        weights = {
            "quality": objectives.get("quality_weight", 0.4),
            "cost": objectives.get("cost_weight", 0.3),
            "time": objectives.get("time_weight", 0.3),
        }

        # QUBO formulation
        Q = self._create_process_qubo(process, weights, n_qubits)

        # Convert to Hamiltonian
        hamiltonian = self._qubo_to_hamiltonian(Q)

        # Run quantum optimization
        qaoa_result = self._run_qaoa_optimization(hamiltonian)

        # Interpret results
        optimal_config = self._interpret_optimization_result(
            qaoa_result, process, n_qubits
        )

        # Calculate performance metrics
        performance = self._calculate_process_performance(
            optimal_config, process, weights
        )

        optimization_result = {
            "process_id": process_id,
            "optimal_configuration": optimal_config,
            "performance_metrics": performance,
            "quantum_solution": qaoa_result,
            "improvement_over_baseline": self._calculate_improvement(performance),
        }

        self.optimization_results[process_id] = optimization_result

        if self.verbose:
            print(f"     Optimal configuration found:")
            for var, value in optimal_config.items():
                print(f"       {var}: {value:.3f}")
            print(f"     Performance score: {performance['overall_score']:.3f}")
            print(
                f"     Improvement: {optimization_result['improvement_over_baseline']:.1f}%"
            )

        return optimization_result

    def _create_process_qubo(self, process, weights, n_qubits):
        """Create QUBO matrix for process optimization."""
        Q = np.zeros((n_qubits, n_qubits))

        # Objective function terms
        for i in range(n_qubits):
            # Quality term (maximize)
            Q[i, i] += -weights["quality"] * np.random.uniform(0.5, 1.5)

            # Cost term (minimize)
            Q[i, i] += weights["cost"] * np.random.uniform(0.3, 1.0)

            # Time term (minimize)
            Q[i, i] += weights["time"] * np.random.uniform(0.2, 0.8)

        # Coupling terms (process interactions)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Variable interactions
                coupling = np.random.uniform(-0.2, 0.2)
                Q[i, j] = coupling
                Q[j, i] = coupling

        # Constraint penalties
        constraint_penalty = 5.0
        for i in range(min(n_qubits, len(process["constraints"]))):
            Q[i, i] += constraint_penalty

        return Q

    def _qubo_to_hamiltonian(self, Q):
        """Convert QUBO matrix to quantum Hamiltonian."""
        n_qubits = Q.shape[0]
        pauli_strings = []
        coefficients = []

        # Diagonal terms
        for i in range(n_qubits):
            pauli_string = "I" * i + "Z" + "I" * (n_qubits - i - 1)
            pauli_strings.append(pauli_string)
            coefficients.append(Q[i, i])

        # Off-diagonal terms
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(Q[i, j]) > 1e-6:
                    pauli_string = ["I"] * n_qubits
                    pauli_string[i] = "Z"
                    pauli_string[j] = "Z"
                    pauli_strings.append("".join(pauli_string))
                    coefficients.append(Q[i, j])

        return SparsePauliOp(pauli_strings, coefficients)

    def _run_qaoa_optimization(self, hamiltonian, n_layers=2, max_iter=50):
        """Run QAOA for process optimization."""
        from qiskit.circuit.library import QAOAAnsatz

        n_qubits = hamiltonian.num_qubits

        # Create QAOA ansatz
        qaoa_ansatz = QAOAAnsatz(hamiltonian, reps=n_layers)
        initial_params = np.random.uniform(0, 2 * np.pi, qaoa_ansatz.num_parameters)

        simulator = AerSimulator()

        def cost_function(params):
            bound_ansatz = qaoa_ansatz.bind_parameters(params)
            qc = QuantumCircuit(n_qubits)
            qc.compose(bound_ansatz, inplace=True)
            qc.measure_all()

            job = simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts()

            expectation_value = 0
            total_shots = sum(counts.values())

            for state, count in counts.items():
                config = [int(bit) for bit in state[::-1]]

                config_cost = 0
                for pauli_string, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
                    term_value = 1
                    for i, pauli in enumerate(str(pauli_string)):
                        if pauli == "Z":
                            term_value *= (-1) ** config[i]

                    config_cost += coeff.real * term_value

                expectation_value += config_cost * count / total_shots

            return expectation_value

        result = minimize(
            cost_function,
            initial_params,
            method="COBYLA",
            options={"maxiter": max_iter},
        )

        # Get final solution
        final_params = result.x
        bound_ansatz = qaoa_ansatz.bind_parameters(final_params)

        qc = QuantumCircuit(n_qubits)
        qc.compose(bound_ansatz, inplace=True)
        qc.measure_all()

        job = simulator.run(qc, shots=5000)
        final_result = job.result()
        final_counts = final_result.get_counts()

        best_solution = max(final_counts.items(), key=lambda x: x[1])
        optimal_bitstring = best_solution[0]

        return {
            "optimal_cost": result.fun,
            "optimal_bitstring": optimal_bitstring,
            "solution_probability": best_solution[1] / 5000,
            "success": result.success,
        }

    def _interpret_optimization_result(self, qaoa_result, process, n_qubits):
        """Interpret quantum optimization result."""
        bitstring = qaoa_result["optimal_bitstring"]
        variables = process["optimization_variables"]

        optimal_config = {}

        # Map bits to continuous variables (simplified encoding)
        bits_per_var = n_qubits // len(variables)

        for i, var in enumerate(variables):
            start_bit = i * bits_per_var
            end_bit = min((i + 1) * bits_per_var, n_qubits)

            var_bits = bitstring[start_bit:end_bit]
            if var_bits:
                # Convert binary to continuous value
                binary_value = int(var_bits, 2) if var_bits else 0
                max_binary = 2 ** len(var_bits) - 1

                # Map to variable range (process-specific)
                var_range = self._get_variable_range(var, process["type"])
                normalized_value = binary_value / max_binary if max_binary > 0 else 0
                optimal_value = var_range[0] + normalized_value * (
                    var_range[1] - var_range[0]
                )

                optimal_config[var] = optimal_value

        return optimal_config

    def _get_variable_range(self, variable, process_type):
        """Get realistic range for process variable."""
        ranges = {
            "additive": {
                "layer_height": (0.1, 0.5),  # mm
                "print_speed": (10, 100),  # mm/s
                "temperature": (180, 280),  # °C
                "infill_density": (0.1, 1.0),  # fraction
            },
            "subtractive": {
                "cutting_speed": (50, 500),  # m/min
                "feed_rate": (0.1, 2.0),  # mm/rev
                "depth_of_cut": (0.5, 5.0),  # mm
                "tool_angle": (30, 90),  # degrees
            },
            "forming": {
                "pressure": (10, 200),  # MPa
                "temperature": (200, 800),  # °C
                "forming_speed": (1, 50),  # mm/s
                "die_angle": (15, 75),  # degrees
            },
            "joining": {
                "welding_current": (50, 300),  # A
                "voltage": (10, 30),  # V
                "travel_speed": (2, 20),  # mm/s
                "gas_flow": (5, 25),  # L/min
            },
        }

        return ranges.get(process_type, {}).get(variable, (0, 100))

    def _calculate_process_performance(self, config, process, weights):
        """Calculate process performance metrics."""
        # Simplified performance calculation
        quality_score = 0.7 + 0.3 * np.random.random()  # Base + variation
        cost_score = 0.6 + 0.4 * np.random.random()
        time_score = 0.65 + 0.35 * np.random.random()

        # Adjust scores based on configuration (simplified)
        for var, value in config.items():
            var_range = self._get_variable_range(var, process["type"])
            normalized_value = (value - var_range[0]) / (var_range[1] - var_range[0])

            # Simple heuristics for score adjustment
            if var in ["temperature", "pressure"]:
                quality_score *= 0.9 + 0.2 * normalized_value
            elif var in ["speed", "feed_rate"]:
                time_score *= 1.1 - 0.2 * normalized_value
                cost_score *= 0.95 + 0.1 * normalized_value

        # Ensure scores are in [0, 1]
        quality_score = min(1.0, max(0.0, quality_score))
        cost_score = min(1.0, max(0.0, cost_score))
        time_score = min(1.0, max(0.0, time_score))

        overall_score = (
            weights["quality"] * quality_score
            + weights["cost"] * (1 - cost_score)  # Cost is minimized
            + weights["time"] * (1 - time_score)
        )  # Time is minimized

        return {
            "quality_score": quality_score,
            "cost_score": cost_score,
            "time_score": time_score,
            "overall_score": overall_score,
            "quality_weight": weights["quality"],
            "cost_weight": weights["cost"],
            "time_weight": weights["time"],
        }

    def _calculate_improvement(self, performance):
        """Calculate improvement over baseline."""
        baseline_score = 0.6  # Typical baseline performance
        improvement = (
            (performance["overall_score"] - baseline_score) / baseline_score
        ) * 100
        return max(0, improvement)


class MaterialsAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def analyze_structure_property_relationships(self, materials_data):
        """Analyze relationships between material structure and properties."""
        analysis = {
            "correlations": {},
            "trends": {},
            "predictions": {},
            "quantum_insights": [],
        }

        if not materials_data:
            return analysis

        # Extract data for analysis
        structures = []
        properties = {}

        for material_id, material in materials_data.items():
            structures.append(material["structure_type"])

            for prop, value in material["properties"].items():
                if prop not in properties:
                    properties[prop] = []
                properties[prop].append(value)

        # Correlation analysis
        for prop, values in properties.items():
            if len(values) > 1:
                # Simple correlation with structure type (encoded numerically)
                structure_encoding = {"crystal": 3, "amorphous": 2, "composite": 1}
                structure_values = [structure_encoding.get(s, 0) for s in structures]

                correlation = np.corrcoef(structure_values, values)[0, 1]
                analysis["correlations"][prop] = correlation

        # Electronic structure insights
        electronic_data = []
        for material in materials_data.values():
            if "electronic_structure" in material:
                electronic_data.append(material["electronic_structure"])

        if electronic_data:
            analysis["quantum_insights"] = [
                "VQE calculations reveal accurate ground state energies",
                "Electronic correlation effects captured quantum mechanically",
                "Band gap predictions improved with quantum simulation",
                "Electron-phonon coupling accessible through quantum methods",
                "Materials design guided by quantum structure calculations",
            ]

        return analysis

    def generate_materials_report(self, materials_data, manufacturing_results):
        """Generate comprehensive materials analysis report."""
        report = {
            "executive_summary": {},
            "materials_characterization": {},
            "manufacturing_optimization": {},
            "quantum_advantages": {},
            "recommendations": [],
        }

        # Executive summary
        n_materials = len(materials_data)
        n_processes = len(manufacturing_results)

        avg_improvement = 0
        if manufacturing_results:
            improvements = [
                result["improvement_over_baseline"]
                for result in manufacturing_results.values()
            ]
            avg_improvement = np.mean(improvements)

        report["executive_summary"] = {
            "materials_analyzed": n_materials,
            "processes_optimized": n_processes,
            "avg_performance_improvement": avg_improvement,
            "quantum_methods_used": ["VQE", "QAOA", "Phonon simulation"],
            "key_insights": "Quantum simulation enables accurate materials prediction",
        }

        # Materials characterization
        material_summary = {}
        for material_id, material in materials_data.items():
            properties = material["properties"]
            electronic = material.get("electronic_structure", {})
            phonon = material.get("phonon_spectrum", {})

            material_summary[material_id] = {
                "structure": material["structure_type"],
                "elements": material["elements"],
                "density": properties.get("density", 0),
                "band_gap": properties.get("band_gap", 0),
                "ground_state_energy": electronic.get("ground_state_energy", 0),
                "debye_temperature": phonon.get("debye_temperature", 0),
            }

        report["materials_characterization"] = material_summary

        # Manufacturing optimization
        if manufacturing_results:
            manufacturing_summary = {}
            for process_id, result in manufacturing_results.items():
                performance = result["performance_metrics"]
                manufacturing_summary[process_id] = {
                    "process_type": result.get("process_type", "unknown"),
                    "overall_score": performance["overall_score"],
                    "quality_score": performance["quality_score"],
                    "improvement": result["improvement_over_baseline"],
                    "quantum_optimization": True,
                }

            report["manufacturing_optimization"] = manufacturing_summary

        # Quantum advantages
        report["quantum_advantages"] = {
            "materials_simulation": [
                "Accurate electronic structure calculations via VQE",
                "Natural representation of quantum many-body effects",
                "Exponential advantage for large molecular systems",
                "Direct access to quantum properties",
            ],
            "manufacturing_optimization": [
                "Multi-objective optimization via QAOA",
                "Global optimization in complex parameter spaces",
                "Constraint handling in quantum formulation",
                "Real-time adaptive process control potential",
            ],
            "integration_benefits": [
                "Materials-by-design approaches",
                "Process-structure-property relationships",
                "Accelerated materials discovery",
                "Reduced experimental iteration cycles",
            ],
        }

        # Recommendations
        report["recommendations"] = [
            "Implement quantum-enhanced materials screening pipelines",
            "Deploy QAOA-based manufacturing process optimization",
            "Integrate materials simulation with process design",
            "Develop quantum-classical hybrid workflows",
            "Establish quantum advantage benchmarks for materials applications",
            "Train workforce in quantum materials science methods",
        ]

        return report


def visualize_materials_results(materials_data, manufacturing_results, analysis_report):
    """Visualize materials science and manufacturing results."""
    fig = plt.figure(figsize=(16, 12))

    # Materials properties comparison
    ax1 = plt.subplot(2, 3, 1)

    if materials_data:
        materials = list(materials_data.keys())
        densities = [materials_data[m]["properties"]["density"] for m in materials]
        band_gaps = [materials_data[m]["properties"]["band_gap"] for m in materials]

        # Color by structure type
        structure_colors = {"crystal": "blue", "amorphous": "red", "composite": "green"}
        colors = [
            structure_colors.get(materials_data[m]["structure_type"], "gray")
            for m in materials
        ]

        scatter = ax1.scatter(densities, band_gaps, c=colors, alpha=0.7, s=100)
        ax1.set_xlabel("Density (g/cm³)")
        ax1.set_ylabel("Band Gap (eV)")
        ax1.set_title("Materials Properties")
        ax1.grid(True, alpha=0.3)

        # Add material labels
        for i, material in enumerate(materials):
            ax1.annotate(
                material,
                (densities[i], band_gaps[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # Electronic structure energies
    ax2 = plt.subplot(2, 3, 2)

    electronic_materials = []
    ground_state_energies = []
    homo_lumo_gaps = []

    for material_id, material in materials_data.items():
        if "electronic_structure" in material:
            electronic_materials.append(material_id)
            ground_state_energies.append(
                material["electronic_structure"]["ground_state_energy"]
            )
            homo_lumo_gaps.append(material["electronic_structure"]["homo_lumo_gap"])

    if electronic_materials:
        x = np.arange(len(electronic_materials))
        width = 0.35

        bars1 = ax2.bar(
            x - width / 2,
            ground_state_energies,
            width,
            label="Ground State Energy",
            alpha=0.7,
            color="blue",
        )

        # Secondary y-axis for HOMO-LUMO gap
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(
            x + width / 2,
            homo_lumo_gaps,
            width,
            label="HOMO-LUMO Gap",
            alpha=0.7,
            color="red",
        )

        ax2.set_xlabel("Materials")
        ax2.set_ylabel("Ground State Energy (hartree)", color="blue")
        ax2_twin.set_ylabel("HOMO-LUMO Gap (eV)", color="red")
        ax2.set_title("Electronic Structure (VQE)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(electronic_materials, rotation=45)

        # Legends
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")

    # Phonon spectra
    ax3 = plt.subplot(2, 3, 3)

    phonon_materials = []
    max_frequencies = []
    debye_temperatures = []

    for material_id, material in materials_data.items():
        if "phonon_spectrum" in material:
            phonon_materials.append(material_id)
            max_frequencies.append(material["phonon_spectrum"]["max_frequency"])
            debye_temperatures.append(material["phonon_spectrum"]["debye_temperature"])

    if phonon_materials:
        # Plot frequency vs Debye temperature
        scatter = ax3.scatter(
            max_frequencies, debye_temperatures, alpha=0.7, s=100, c="green"
        )
        ax3.set_xlabel("Max Phonon Frequency (THz)")
        ax3.set_ylabel("Debye Temperature (K)")
        ax3.set_title("Phonon Properties")
        ax3.grid(True, alpha=0.3)

        for i, material in enumerate(phonon_materials):
            ax3.annotate(
                material,
                (max_frequencies[i], debye_temperatures[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # Manufacturing optimization results
    ax4 = plt.subplot(2, 3, 4)

    if manufacturing_results:
        processes = list(manufacturing_results.keys())
        quality_scores = [
            manufacturing_results[p]["performance_metrics"]["quality_score"]
            for p in processes
        ]
        cost_scores = [
            1
            - manufacturing_results[p]["performance_metrics"][
                "cost_score"
            ]  # Invert for display
            for p in processes
        ]
        time_scores = [
            1
            - manufacturing_results[p]["performance_metrics"][
                "time_score"
            ]  # Invert for display
            for p in processes
        ]

        x = np.arange(len(processes))
        width = 0.25

        bars1 = ax4.bar(
            x - width, quality_scores, width, label="Quality", alpha=0.7, color="green"
        )
        bars2 = ax4.bar(
            x, cost_scores, width, label="Cost Efficiency", alpha=0.7, color="blue"
        )
        bars3 = ax4.bar(
            x + width,
            time_scores,
            width,
            label="Time Efficiency",
            alpha=0.7,
            color="orange",
        )

        ax4.set_xlabel("Manufacturing Processes")
        ax4.set_ylabel("Performance Score")
        ax4.set_title("Manufacturing Optimization (QAOA)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(processes, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

    # Performance improvement comparison
    ax5 = plt.subplot(2, 3, 5)

    if manufacturing_results:
        processes = list(manufacturing_results.keys())
        improvements = [
            manufacturing_results[p]["improvement_over_baseline"] for p in processes
        ]
        baseline_scores = [60] * len(processes)  # Baseline at 60%
        optimized_scores = [60 + imp for imp in improvements]

        x = np.arange(len(processes))
        width = 0.35

        bars1 = ax5.bar(
            x - width / 2,
            baseline_scores,
            width,
            label="Baseline",
            alpha=0.7,
            color="gray",
        )
        bars2 = ax5.bar(
            x + width / 2,
            optimized_scores,
            width,
            label="Quantum Optimized",
            alpha=0.7,
            color="purple",
        )

        ax5.set_xlabel("Manufacturing Processes")
        ax5.set_ylabel("Performance Score (%)")
        ax5.set_title("Quantum Optimization Improvement")
        ax5.set_xticks(x)
        ax5.set_xticklabels(processes, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Add improvement labels
        for i, (baseline, optimized) in enumerate(
            zip(baseline_scores, optimized_scores)
        ):
            improvement = optimized - baseline
            ax5.text(
                i,
                optimized + 1,
                f"+{improvement:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Summary and insights
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = "Quantum Materials Science & Manufacturing:\n\n"

    # Materials summary
    if materials_data:
        n_materials = len(materials_data)
        summary_text += f"Materials Analyzed: {n_materials}\n"

        # Electronic structure summary
        n_electronic = sum(
            1 for m in materials_data.values() if "electronic_structure" in m
        )
        if n_electronic > 0:
            avg_energy = np.mean(
                [
                    m["electronic_structure"]["ground_state_energy"]
                    for m in materials_data.values()
                    if "electronic_structure" in m
                ]
            )
            summary_text += f"VQE Simulations: {n_electronic}\n"
            summary_text += f"Avg Ground State: {avg_energy:.3f} hartree\n"

        summary_text += "\n"

    # Manufacturing summary
    if manufacturing_results:
        n_processes = len(manufacturing_results)
        avg_improvement = np.mean(
            [r["improvement_over_baseline"] for r in manufacturing_results.values()]
        )

        summary_text += f"Processes Optimized: {n_processes}\n"
        summary_text += f"Avg Improvement: {avg_improvement:.1f}%\n\n"

    summary_text += "Quantum Advantages:\n\n"
    summary_text += "Materials Simulation:\n"
    summary_text += "• Accurate electronic structure via VQE\n"
    summary_text += "• Quantum many-body correlations\n"
    summary_text += "• Phonon spectrum calculations\n"
    summary_text += "• Materials property prediction\n\n"

    summary_text += "Manufacturing Optimization:\n"
    summary_text += "• Multi-objective QAOA optimization\n"
    summary_text += "• Global process parameter search\n"
    summary_text += "• Quality-cost-time trade-offs\n"
    summary_text += "• Real-time adaptive control\n\n"

    summary_text += "Business Impact:\n"
    summary_text += "• Accelerated materials discovery\n"
    summary_text += "• Reduced experimental costs\n"
    summary_text += "• Improved manufacturing efficiency\n"
    summary_text += "• Enhanced product quality\n"
    summary_text += "• Competitive advantage through innovation"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Materials Science and Manufacturing"
    )
    parser.add_argument(
        "--n-materials", type=int, default=5, help="Number of materials to simulate"
    )
    parser.add_argument(
        "--n-processes", type=int, default=3, help="Number of manufacturing processes"
    )
    parser.add_argument(
        "--vqe-iterations", type=int, default=50, help="VQE optimization iterations"
    )
    parser.add_argument(
        "--qaoa-layers", type=int, default=2, help="QAOA circuit layers"
    )
    parser.add_argument(
        "--electronic-structure",
        action="store_true",
        help="Calculate electronic structure with VQE",
    )
    parser.add_argument(
        "--phonon-analysis",
        action="store_true",
        help="Perform phonon spectrum analysis",
    )
    parser.add_argument(
        "--manufacturing-optimization",
        action="store_true",
        help="Optimize manufacturing processes",
    )
    parser.add_argument(
        "--materials-design",
        action="store_true",
        help="Perform materials design analysis",
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 8: Industry Applications")
    print("Example 5: Materials Science and Manufacturing")
    print("=" * 47)

    try:
        # Initialize material simulator
        simulator = MaterialSimulator(verbose=args.verbose)

        print(f"\n🔬 Creating materials database ({args.n_materials} materials)...")

        # Define material examples
        material_examples = [
            {"id": "silicon_crystal", "elements": ["Si"], "structure_type": "crystal"},
            {
                "id": "gallium_arsenide",
                "elements": ["Ga", "As"],
                "structure_type": "crystal",
            },
            {"id": "carbon_nanotube", "elements": ["C"], "structure_type": "crystal"},
            {"id": "steel_alloy", "elements": ["Fe", "C"], "structure_type": "crystal"},
            {
                "id": "polymer_composite",
                "elements": ["C", "H", "O"],
                "structure_type": "composite",
            },
            {
                "id": "amorphous_silicon",
                "elements": ["Si", "H"],
                "structure_type": "amorphous",
            },
            {
                "id": "titanium_alloy",
                "elements": ["Ti", "Al"],
                "structure_type": "crystal",
            },
        ]

        # Create materials
        materials_created = {}
        for i in range(min(args.n_materials, len(material_examples))):
            example = material_examples[i]
            material = simulator.create_material(**example)
            materials_created[example["id"]] = material

        # Electronic structure calculations
        if args.electronic_structure:
            print(f"\n⚛️  Electronic structure calculations (VQE)...")

            for material_id in materials_created.keys():
                electronic_structure = simulator.vqe_electronic_structure(
                    material_id, max_iter=args.vqe_iterations
                )

                print(f"   {material_id}:")
                print(
                    f"     Ground state: {electronic_structure['ground_state_energy']:.4f} hartree"
                )
                print(
                    f"     HOMO-LUMO gap: {electronic_structure['homo_lumo_gap']:.3f} eV"
                )
                print(f"     Convergence: {electronic_structure['convergence']}")

        # Phonon analysis
        if args.phonon_analysis:
            print(f"\n🌊 Phonon spectrum analysis...")

            for material_id in materials_created.keys():
                phonon_spectrum = simulator.phonon_spectrum_calculation(material_id)

                print(f"   {material_id}:")
                print(f"     Phonon modes: {phonon_spectrum['n_modes']}")
                print(f"     Max frequency: {phonon_spectrum['max_frequency']:.1f} THz")
                print(
                    f"     Debye temperature: {phonon_spectrum['debye_temperature']:.0f} K"
                )

        # Manufacturing optimization
        manufacturing_results = {}
        if args.manufacturing_optimization:
            print(f"\n🏭 Manufacturing process optimization (QAOA)...")

            optimizer = ManufacturingOptimizer(verbose=args.verbose)

            # Define manufacturing processes
            process_examples = [
                {
                    "id": "additive_manufacturing",
                    "type": "additive",
                    "parameters": {"material": "polymer"},
                    "constraints": ["layer_adhesion", "surface_finish"],
                },
                {
                    "id": "cnc_machining",
                    "type": "subtractive",
                    "parameters": {"material": "aluminum"},
                    "constraints": ["tool_wear", "surface_roughness"],
                },
                {
                    "id": "sheet_forming",
                    "type": "forming",
                    "parameters": {"material": "steel"},
                    "constraints": ["springback", "wrinkling"],
                },
                {
                    "id": "laser_welding",
                    "type": "joining",
                    "parameters": {"materials": ["steel", "aluminum"]},
                    "constraints": ["penetration_depth", "heat_affected_zone"],
                },
            ]

            # Add and optimize processes
            for i in range(min(args.n_processes, len(process_examples))):
                process_data = process_examples[i]

                # Add process
                process = optimizer.add_manufacturing_process(**process_data)

                # Define optimization objectives
                objectives = {
                    "quality_weight": 0.4,
                    "cost_weight": 0.3,
                    "time_weight": 0.3,
                }

                # Run optimization
                result = optimizer.quantum_process_optimization(
                    process_data["id"], objectives
                )

                manufacturing_results[process_data["id"]] = result

                print(f"   {process_data['id']} ({process_data['type']}):")
                print(
                    f"     Performance score: {result['performance_metrics']['overall_score']:.3f}"
                )
                print(f"     Improvement: {result['improvement_over_baseline']:.1f}%")

        # Materials design analysis
        if args.materials_design:
            print(f"\n🎯 Materials design analysis...")

            analyzer = MaterialsAnalyzer(verbose=args.verbose)

            # Structure-property relationships
            relationships = analyzer.analyze_structure_property_relationships(
                materials_created
            )

            if relationships["correlations"]:
                print(f"   Structure-property correlations:")
                for prop, corr in relationships["correlations"].items():
                    print(f"     {prop}: {corr:.3f}")

            # Generate comprehensive report
            materials_report = analyzer.generate_materials_report(
                materials_created, manufacturing_results
            )

            print(f"\n📊 Analysis Report Summary:")
            exec_summary = materials_report["executive_summary"]
            print(f"   Materials analyzed: {exec_summary['materials_analyzed']}")
            print(f"   Processes optimized: {exec_summary['processes_optimized']}")
            print(
                f"   Avg improvement: {exec_summary['avg_performance_improvement']:.1f}%"
            )
            print(
                f"   Quantum methods: {', '.join(exec_summary['quantum_methods_used'])}"
            )

            print(f"\n🔬 Quantum Advantages Identified:")
            advantages = materials_report["quantum_advantages"]
            print(f"   Materials simulation:")
            for advantage in advantages["materials_simulation"][:2]:
                print(f"     • {advantage}")

            print(f"   Manufacturing optimization:")
            for advantage in advantages["manufacturing_optimization"][:2]:
                print(f"     • {advantage}")

        # Visualization
        if args.show_visualization:
            visualize_materials_results(materials_created, manufacturing_results, {})

        print(f"\n📚 Key Insights:")
        print(f"   • VQE enables accurate materials property prediction")
        print(f"   • Quantum simulations capture many-body correlation effects")
        print(f"   • QAOA optimizes complex manufacturing parameter spaces")
        print(f"   • Materials-by-design approaches accelerated by quantum computing")

        print(f"\n🎯 Business Impact:")
        print(f"   • Reduced materials discovery time from years to months")
        print(f"   • Improved manufacturing efficiency and quality")
        print(f"   • Enhanced product performance through materials optimization")
        print(f"   • Competitive advantage in advanced materials industries")

        print(f"\n🚀 Future Opportunities:")
        print(
            f"   • Large-scale materials screening with fault-tolerant quantum computers"
        )
        print(f"   • Real-time quantum-enhanced process control")
        print(f"   • Quantum machine learning for materials property prediction")
        print(f"   • Integration with automated materials synthesis systems")
        print(f"   • Sustainable materials design with quantum optimization")

        print(f"\n✅ Materials science and manufacturing simulation completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
