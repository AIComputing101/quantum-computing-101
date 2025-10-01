#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 8: Industry Applications
Example 4: Cryptography and Cybersecurity

Implementation of quantum cryptography protocols and post-quantum security analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_statevector, Statevector
from scipy.stats import entropy
import hashlib
import secrets
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import warnings

warnings.filterwarnings("ignore")


class QuantumKeyDistribution:
    def __init__(self, key_length=256, noise_level=0.05, verbose=False):
        self.key_length = key_length
        self.noise_level = noise_level
        self.verbose = verbose
        self.shared_key = None
        self.security_metrics = {}

    def bb84_protocol(self, eve_present=False, eve_intercept_rate=0.2):
        """Implement BB84 Quantum Key Distribution protocol."""
        if self.verbose:
            print(f"   Running BB84 protocol (key length: {self.key_length})...")

        # Alice's preparation
        alice_bits = [
            secrets.randbelow(2) for _ in range(self.key_length * 2)
        ]  # Double for sifting
        alice_bases = [
            secrets.randbelow(2) for _ in range(self.key_length * 2)
        ]  # 0: rectilinear, 1: diagonal

        # Quantum channel simulation
        transmitted_qubits = []

        for bit, basis in zip(alice_bits, alice_bases):
            qc = QuantumCircuit(1, 1)

            # Prepare qubit based on bit and basis
            if bit == 1:
                qc.x(0)  # |1⟩ state

            if basis == 1:  # Diagonal basis
                qc.h(0)  # Hadamard for diagonal basis

            transmitted_qubits.append(qc)

        # Eve's interception (if present)
        eve_measurements = []
        if eve_present:
            if self.verbose:
                print(f"     Eve intercepting {eve_intercept_rate:.1%} of qubits...")

            for i, qc in enumerate(transmitted_qubits):
                if secrets.randbelow(100) < eve_intercept_rate * 100:
                    # Eve measures with random basis
                    eve_basis = secrets.randbelow(2)

                    # Simulate Eve's measurement
                    measure_qc = qc.copy()
                    if eve_basis == 1:
                        measure_qc.h(0)  # Diagonal measurement
                    measure_qc.measure(0, 0)

                    # Run measurement
                    simulator = AerSimulator()
                    job = simulator.run(measure_qc, shots=1)
                    result = job.result()
                    counts = result.get_counts()
                    eve_bit = int(list(counts.keys())[0])

                    eve_measurements.append((i, eve_basis, eve_bit))

                    # Eve retransmits (introduces errors)
                    new_qc = QuantumCircuit(1, 1)
                    if eve_bit == 1:
                        new_qc.x(0)
                    if eve_basis == 1:
                        new_qc.h(0)

                    transmitted_qubits[i] = new_qc

        # Bob's measurement
        bob_bases = [secrets.randbelow(2) for _ in range(len(transmitted_qubits))]
        bob_measurements = []

        simulator = AerSimulator()

        for qc, bob_basis in zip(transmitted_qubits, bob_bases):
            measure_qc = qc.copy()

            if bob_basis == 1:  # Diagonal measurement
                measure_qc.h(0)

            measure_qc.measure(0, 0)

            # Add noise
            if secrets.randbelow(1000) < self.noise_level * 1000:
                # Flip measurement result due to noise
                measure_qc.x(0)

            job = simulator.run(measure_qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bob_bit = int(list(counts.keys())[0])

            bob_measurements.append(bob_bit)

        # Basis reconciliation (public channel)
        sifted_alice_bits = []
        sifted_bob_bits = []

        for i, (alice_basis, bob_basis) in enumerate(zip(alice_bases, bob_bases)):
            if alice_basis == bob_basis:  # Same basis used
                sifted_alice_bits.append(alice_bits[i])
                sifted_bob_bits.append(bob_measurements[i])

        # Error detection (sample subset)
        test_sample_size = min(len(sifted_alice_bits) // 4, 50)
        test_indices = secrets.SystemRandom().sample(
            range(len(sifted_alice_bits)), test_sample_size
        )

        error_count = 0
        for idx in sorted(test_indices, reverse=True):
            if sifted_alice_bits[idx] != sifted_bob_bits[idx]:
                error_count += 1
            # Remove test bits
            del sifted_alice_bits[idx]
            del sifted_bob_bits[idx]

        quantum_bit_error_rate = (
            error_count / test_sample_size if test_sample_size > 0 else 0
        )

        # Error correction (simplified)
        corrected_key = sifted_alice_bits[: self.key_length]

        # Privacy amplification (simplified hash-based)
        if len(corrected_key) >= self.key_length:
            key_string = "".join(map(str, corrected_key))
            hash_key = hashlib.sha256(key_string.encode()).digest()
            self.shared_key = hash_key
        else:
            self.shared_key = None

        # Security analysis
        self.security_metrics["bb84"] = {
            "initial_bits": len(alice_bits),
            "sifted_bits": len(sifted_alice_bits) + test_sample_size,
            "final_key_length": len(corrected_key),
            "qber": quantum_bit_error_rate,
            "sifting_efficiency": (len(sifted_alice_bits) + test_sample_size)
            / len(alice_bits),
            "eve_detected": quantum_bit_error_rate > 0.11,  # Threshold for detection
            "eve_present": eve_present,
            "eve_measurements": len(eve_measurements) if eve_present else 0,
            "security_parameter": max(0, 1 - 2 * quantum_bit_error_rate),
        }

        if self.verbose:
            print(
                f"     Sifted {len(sifted_alice_bits) + test_sample_size} bits from {len(alice_bits)}"
            )
            print(f"     QBER: {quantum_bit_error_rate:.3f}")
            print(f"     Eve detected: {self.security_metrics['bb84']['eve_detected']}")
            print(f"     Final key length: {len(corrected_key)} bits")

        return self.shared_key

    def e91_protocol(self, entanglement_fidelity=0.95):
        """Implement E91 (Ekert 91) entanglement-based QKD protocol."""
        if self.verbose:
            print(f"   Running E91 protocol...")

        # Generate entangled pairs
        n_pairs = self.key_length * 2  # Extra pairs for Bell test

        # Simulate Bell state measurements
        alice_measurements = []
        bob_measurements = []
        alice_angles = []  # Measurement angles
        bob_angles = []

        # Measurement angles for Bell inequality test
        angles = [0, np.pi / 4, np.pi / 2]  # 0°, 45°, 90°

        for _ in range(n_pairs):
            # Alice and Bob choose measurement angles
            alice_angle = secrets.choice(angles)
            bob_angle = secrets.choice(angles)

            alice_angles.append(alice_angle)
            bob_angles.append(bob_angle)

            # Simulate correlated measurements for Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
            # Perfect correlation when same angle, anti-correlation when different

            if entanglement_fidelity > secrets.random():
                # Perfect entanglement case
                if alice_angle == bob_angle:
                    # Perfect correlation
                    outcome = secrets.randbelow(2)
                    alice_bit = outcome
                    bob_bit = outcome
                else:
                    # Anti-correlation with quantum probability
                    angle_diff = abs(alice_angle - bob_angle)
                    correlation = np.cos(angle_diff) ** 2

                    alice_bit = secrets.randbelow(2)
                    if secrets.random() < correlation:
                        bob_bit = alice_bit
                    else:
                        bob_bit = 1 - alice_bit
            else:
                # Noise/imperfect entanglement
                alice_bit = secrets.randbelow(2)
                bob_bit = secrets.randbelow(2)

            alice_measurements.append(alice_bit)
            bob_measurements.append(bob_bit)

        # Bell inequality test (subset of measurements)
        bell_test_indices = [
            i for i in range(n_pairs) if alice_angles[i] != bob_angles[i]
        ][
            :100
        ]  # Sample for Bell test

        bell_violations = 0
        for idx in bell_test_indices:
            # Calculate CHSH value for sampled measurements
            if len(bell_test_indices) >= 4:
                # Simplified Bell test
                correlation = (
                    1 if alice_measurements[idx] == bob_measurements[idx] else -1
                )
                if abs(correlation) > 1 / np.sqrt(2):  # Quantum threshold
                    bell_violations += 1

        bell_violation_rate = (
            bell_violations / len(bell_test_indices) if bell_test_indices else 0
        )

        # Key extraction from correlated measurements
        key_indices = [i for i in range(n_pairs) if alice_angles[i] == bob_angles[i]][
            : self.key_length
        ]

        key_bits = [alice_measurements[i] for i in key_indices]

        if len(key_bits) >= self.key_length:
            key_string = "".join(map(str, key_bits[: self.key_length]))
            self.shared_key = hashlib.sha256(key_string.encode()).digest()
        else:
            self.shared_key = None

        self.security_metrics["e91"] = {
            "entangled_pairs": n_pairs,
            "bell_test_pairs": len(bell_test_indices),
            "bell_violations": bell_violations,
            "bell_violation_rate": bell_violation_rate,
            "entanglement_verified": bell_violation_rate > 0.7,  # Threshold
            "final_key_length": len(key_bits),
            "entanglement_fidelity": entanglement_fidelity,
        }

        if self.verbose:
            print(f"     Entangled pairs: {n_pairs}")
            print(f"     Bell violations: {bell_violations}/{len(bell_test_indices)}")
            print(
                f"     Entanglement verified: {self.security_metrics['e91']['entanglement_verified']}"
            )
            print(f"     Final key: {len(key_bits)} bits")

        return self.shared_key


class PostQuantumCryptography:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.algorithms = {}
        self.security_analysis = {}

    def lattice_based_encryption(self, message, n=512, q=8192):
        """Implement simplified lattice-based encryption (LWE-based)."""
        if self.verbose:
            print(f"   Lattice-based encryption (n={n}, q={q})...")

        # Generate LWE parameters
        # Secret key: s ∈ Z_q^n
        s = np.random.randint(0, q, n)

        # Public key: (A, b = As + e) where A ∈ Z_q^{m×n}, e is error
        m = n + 256  # Number of samples
        A = np.random.randint(0, q, (m, n))

        # Error distribution (discrete Gaussian approximation)
        error_std = np.sqrt(2)
        e = np.random.normal(0, error_std, m).astype(int) % q

        b = (A @ s + e) % q

        public_key = (A, b)
        private_key = s

        # Message encoding (simplified binary)
        if isinstance(message, str):
            message_bits = "".join(format(ord(c), "08b") for c in message)
        else:
            message_bits = str(message)

        # Pad to encryption block size
        block_size = 8
        while len(message_bits) % block_size != 0:
            message_bits += "0"

        ciphertext_blocks = []

        for i in range(0, len(message_bits), block_size):
            block = message_bits[i : i + block_size]
            m_bit = int(block, 2)

            # Encryption: sample random r, compute (u, v) = (Ar, br + m⌊q/2⌋)
            r = np.random.randint(0, 2, m)  # Binary random vector

            u = (A.T @ r) % q
            v = (b @ r + m_bit * (q // 2)) % q

            ciphertext_blocks.append((u, v))

        ciphertext = ciphertext_blocks

        # Security analysis
        lattice_dimension = n
        approximation_factor = np.sqrt(n)  # Simplified

        # Estimate security level (bits)
        security_level = min(128, lattice_dimension // 4)  # Conservative estimate

        self.security_analysis["lattice"] = {
            "algorithm": "LWE-based",
            "lattice_dimension": lattice_dimension,
            "modulus": q,
            "error_std": error_std,
            "approximation_factor": approximation_factor,
            "security_level_bits": security_level,
            "key_size_bits": n * np.log2(q),
            "ciphertext_expansion": len(ciphertext_blocks)
            * (len(u) + 1)
            * np.log2(q)
            / len(message_bits),
        }

        return {
            "ciphertext": ciphertext,
            "public_key": public_key,
            "private_key": private_key,
            "message_length": len(message_bits),
        }

    def hash_based_signatures(self, message, tree_height=10):
        """Implement simplified hash-based signature scheme (Merkle signatures)."""
        if self.verbose:
            print(f"   Hash-based signatures (tree height: {tree_height})...")

        # Generate one-time signature keys
        n_signatures = 2**tree_height

        # Lamport one-time signature key generation
        def generate_lamport_keypair():
            private_key = []
            public_key = []

            for _ in range(256):  # 256-bit hash
                # Two random values for each bit (0 and 1)
                sk_0 = secrets.token_bytes(32)
                sk_1 = secrets.token_bytes(32)

                pk_0 = hashlib.sha256(sk_0).digest()
                pk_1 = hashlib.sha256(sk_1).digest()

                private_key.append((sk_0, sk_1))
                public_key.append((pk_0, pk_1))

            return private_key, public_key

        # Generate key pairs for Merkle tree
        ots_private_keys = []
        ots_public_keys = []

        for _ in range(min(4, n_signatures)):  # Limit for demo
            sk, pk = generate_lamport_keypair()
            ots_private_keys.append(sk)
            ots_public_keys.append(pk)

        # Build Merkle tree
        def hash_combine(left, right):
            return hashlib.sha256(left + right).digest()

        # Leaf nodes (hash of public keys)
        leaves = []
        for pk in ots_public_keys:
            pk_serialized = b"".join(b"".join(pair) for pair in pk)
            leaf_hash = hashlib.sha256(pk_serialized).digest()
            leaves.append(leaf_hash)

        # Build tree bottom-up
        tree_levels = [leaves]
        current_level = leaves

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = hash_combine(current_level[i], current_level[i + 1])
                else:
                    combined = current_level[i]  # Odd number of nodes
                next_level.append(combined)
            tree_levels.append(next_level)
            current_level = next_level

        merkle_root = current_level[0]

        # Sign message with first one-time key
        message_hash = hashlib.sha256(
            message.encode() if isinstance(message, str) else message
        ).digest()
        message_bits = "".join(format(byte, "08b") for byte in message_hash)

        signature_parts = []
        ots_private_key = ots_private_keys[0]

        for i, bit in enumerate(message_bits):
            if bit == "0":
                signature_parts.append(ots_private_key[i][0])
            else:
                signature_parts.append(ots_private_key[i][1])

        # Authentication path for Merkle tree
        auth_path = []
        leaf_index = 0

        for level in range(len(tree_levels) - 1):
            level_nodes = tree_levels[level]
            if leaf_index % 2 == 0:
                # Left child, need right sibling
                if leaf_index + 1 < len(level_nodes):
                    auth_path.append(level_nodes[leaf_index + 1])
            else:
                # Right child, need left sibling
                auth_path.append(level_nodes[leaf_index - 1])

            leaf_index //= 2

        signature = {
            "ots_signature": signature_parts,
            "ots_public_key": ots_public_keys[0],
            "auth_path": auth_path,
            "leaf_index": 0,
        }

        # Security analysis
        self.security_analysis["hash_based"] = {
            "algorithm": "Merkle signatures",
            "tree_height": tree_height,
            "max_signatures": n_signatures,
            "hash_function": "SHA-256",
            "security_level_bits": 128,  # SHA-256 provides 128-bit security
            "signature_size_bytes": len(signature_parts) * 32 + len(auth_path) * 32,
            "public_key_size_bytes": 32,  # Just Merkle root
            "quantum_resistant": True,
        }

        return {
            "signature": signature,
            "merkle_root": merkle_root,
            "message_hash": message_hash,
        }

    def code_based_encryption(self, message, n=1024, k=512, t=50):
        """Implement simplified code-based encryption (McEliece-style)."""
        if self.verbose:
            print(f"   Code-based encryption (n={n}, k={k}, t={t})...")

        # Generate random linear code (simplified)
        # In practice, would use Goppa codes or similar

        # Generator matrix G (k × n)
        G = np.random.randint(0, 2, (k, n))

        # Ensure G is in systematic form [I_k | P]
        G[:, :k] = np.eye(k, dtype=int)

        # Public key is scrambled version of G
        # Scrambling matrix S (k × k)
        S = np.random.randint(0, 2, (k, k))
        while np.linalg.det(S) == 0:  # Ensure invertible
            S = np.random.randint(0, 2, (k, k))

        # Permutation matrix P (n × n) - represented as permutation
        perm = np.random.permutation(n)

        # Public key: G' = SGP
        G_pub = (S @ G) % 2
        G_pub = G_pub[:, perm]  # Apply permutation

        private_key = {
            "G": G,
            "S": S,
            "S_inv": np.linalg.inv(S).astype(int) % 2,
            "permutation": perm,
            "error_capacity": t,
        }

        public_key = G_pub

        # Message encoding
        if isinstance(message, str):
            message_bits = "".join(format(ord(c), "08b") for c in message)
        else:
            message_bits = str(message)

        # Pad message to k bits
        while len(message_bits) < k:
            message_bits += "0"

        message_vector = np.array([int(b) for b in message_bits[:k]])

        # Encoding: c = mG'
        codeword = (message_vector @ G_pub) % 2

        # Add random error
        error_vector = np.zeros(n, dtype=int)
        error_positions = np.random.choice(n, t, replace=False)
        error_vector[error_positions] = 1

        ciphertext = (codeword + error_vector) % 2

        # Security analysis
        code_dimension = k
        code_length = n
        min_distance = 2 * t + 1  # Designed minimum distance

        # Work factor for decoding (simplified)
        work_factor = min(2**k, 2 ** (n - k))  # Information set decoding
        security_bits = min(128, int(np.log2(work_factor)))

        self.security_analysis["code_based"] = {
            "algorithm": "McEliece-style",
            "code_length": code_length,
            "code_dimension": code_dimension,
            "error_capacity": t,
            "min_distance": min_distance,
            "security_level_bits": security_bits,
            "public_key_size_bits": k * n,
            "ciphertext_expansion": n / k,
            "quantum_resistant": True,
        }

        return {
            "ciphertext": ciphertext,
            "public_key": public_key,
            "private_key": private_key,
            "original_message_length": len(message_bits),
        }


class QuantumSecurityAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def analyze_classical_security(self, key_size, algorithm_type):
        """Analyze security of classical cryptographic algorithms."""
        # Classical security levels
        classical_security = {
            "AES-128": {
                "key_size": 128,
                "quantum_security": 64,
                "classical_security": 128,
            },
            "AES-256": {
                "key_size": 256,
                "quantum_security": 128,
                "classical_security": 256,
            },
            "RSA-2048": {
                "key_size": 2048,
                "quantum_security": 0,
                "classical_security": 112,
            },
            "RSA-3072": {
                "key_size": 3072,
                "quantum_security": 0,
                "classical_security": 128,
            },
            "ECC-256": {
                "key_size": 256,
                "quantum_security": 0,
                "classical_security": 128,
            },
            "ECC-384": {
                "key_size": 384,
                "quantum_security": 0,
                "classical_security": 192,
            },
        }

        security_info = classical_security.get(
            algorithm_type,
            {
                "key_size": key_size,
                "quantum_security": key_size // 2 if "AES" in algorithm_type else 0,
                "classical_security": key_size,
            },
        )

        return security_info

    def quantum_attack_analysis(self, algorithm_type, parameters):
        """Analyze quantum attack complexity."""
        attack_analysis = {}

        if algorithm_type == "RSA":
            # Shor's algorithm
            n = parameters.get("modulus_size", 2048)
            attack_analysis = {
                "attack_algorithm": "Shor's algorithm",
                "quantum_complexity": f"O((log n)³)",
                "qubits_required": n * 2,  # Rough estimate
                "gate_depth": n**3,
                "practical_threat": n <= 4096,  # Current/near-term quantum computers
                "time_to_break_classical": f"2^{n//2}",
                "time_to_break_quantum": f"O((log {n})³)",
            }

        elif algorithm_type == "ECC":
            # Shor's algorithm for elliptic curves
            field_size = parameters.get("field_size", 256)
            attack_analysis = {
                "attack_algorithm": "Shor's algorithm (EC variant)",
                "quantum_complexity": f"O((log p)³)",
                "qubits_required": field_size * 6,  # Rough estimate
                "gate_depth": field_size**3,
                "practical_threat": field_size <= 512,
                "time_to_break_classical": f"2^{field_size//2}",
                "time_to_break_quantum": f"O((log {field_size})³)",
            }

        elif algorithm_type == "AES":
            # Grover's algorithm
            key_size = parameters.get("key_size", 128)
            attack_analysis = {
                "attack_algorithm": "Grover's algorithm",
                "quantum_complexity": f"O(2^{key_size//2})",
                "qubits_required": key_size * 2,
                "gate_depth": 2 ** (key_size // 2),
                "practical_threat": key_size <= 128,
                "time_to_break_classical": f"2^{key_size}",
                "time_to_break_quantum": f"2^{key_size//2}",
            }

        elif algorithm_type == "Hash":
            # Grover's algorithm for preimage
            output_size = parameters.get("output_size", 256)
            attack_analysis = {
                "attack_algorithm": "Grover's algorithm",
                "quantum_complexity": f"O(2^{output_size//2})",
                "qubits_required": output_size,
                "gate_depth": 2 ** (output_size // 2),
                "practical_threat": output_size <= 256,
                "time_to_break_classical": f"2^{output_size}",
                "time_to_break_quantum": f"2^{output_size//2}",
            }

        return attack_analysis

    def post_quantum_security_assessment(self, pq_algorithms):
        """Assess security of post-quantum algorithms."""
        assessment = {}

        for alg_name, alg_data in pq_algorithms.items():
            if alg_name == "lattice":
                # Lattice-based security
                n = alg_data.get("lattice_dimension", 512)
                q = alg_data.get("modulus", 8192)

                # Best known attacks: BKZ, sieve algorithms
                bkz_complexity = 2 ** (0.292 * n)  # Simplified
                sieve_complexity = 2 ** (0.265 * n)  # Simplified

                assessment[alg_name] = {
                    "security_assumption": "Learning With Errors (LWE)",
                    "best_classical_attack": "BKZ reduction",
                    "best_quantum_attack": "Quantum sieving",
                    "classical_complexity": bkz_complexity,
                    "quantum_complexity": sieve_complexity ** (0.75),  # Quantum speedup
                    "security_level": min(
                        256, int(np.log2(sieve_complexity ** (0.75)))
                    ),
                    "standardized": "NIST PQC Round 3",
                    "quantum_resistant": True,
                }

            elif alg_name == "hash_based":
                # Hash-based security
                assessment[alg_name] = {
                    "security_assumption": "Hash function security",
                    "best_classical_attack": "Collision search",
                    "best_quantum_attack": "Grover's algorithm",
                    "classical_complexity": 2**128,  # SHA-256
                    "quantum_complexity": 2**64,  # Grover speedup
                    "security_level": 128,  # Post-quantum
                    "standardized": "NIST SP 800-208",
                    "quantum_resistant": True,
                }

            elif alg_name == "code_based":
                # Code-based security
                n = alg_data.get("code_length", 1024)
                k = alg_data.get("code_dimension", 512)
                t = alg_data.get("error_capacity", 50)

                # Information set decoding
                isd_complexity = min(2**k, 2 ** (n - k))

                assessment[alg_name] = {
                    "security_assumption": "Syndrome decoding problem",
                    "best_classical_attack": "Information set decoding",
                    "best_quantum_attack": "Quantum ISD variants",
                    "classical_complexity": isd_complexity,
                    "quantum_complexity": isd_complexity
                    ** (0.5),  # Square root speedup
                    "security_level": min(256, int(np.log2(isd_complexity ** (0.5)))),
                    "standardized": "Under evaluation",
                    "quantum_resistant": True,
                }

        return assessment

    def generate_security_report(self, qkd_results, pq_results, classical_comparison):
        """Generate comprehensive security report."""
        report = {
            "executive_summary": {},
            "qkd_analysis": {},
            "post_quantum_analysis": {},
            "migration_strategy": {},
            "recommendations": [],
        }

        # Executive summary
        report["executive_summary"] = {
            "quantum_threat_timeline": "2030-2040 (estimates)",
            "current_vulnerability": "RSA, ECC, DH vulnerable to quantum attacks",
            "qkd_feasibility": "Limited to point-to-point, short distances",
            "pqc_readiness": "NIST standards available, migration underway",
            "action_required": "Immediate migration planning recommended",
        }

        # QKD analysis
        if qkd_results:
            bb84_metrics = qkd_results.get("bb84", {})
            e91_metrics = qkd_results.get("e91", {})

            report["qkd_analysis"] = {
                "bb84_protocol": {
                    "security_level": "Information-theoretic",
                    "key_generation_rate": bb84_metrics.get("sifting_efficiency", 0),
                    "error_tolerance": bb84_metrics.get("qber", 0),
                    "eavesdropping_detection": bb84_metrics.get("eve_detected", False),
                    "practical_limitations": [
                        "Distance limited",
                        "Low key rates",
                        "Expensive infrastructure",
                    ],
                },
                "e91_protocol": {
                    "security_level": "Information-theoretic",
                    "entanglement_verification": e91_metrics.get(
                        "entanglement_verified", False
                    ),
                    "bell_violations": e91_metrics.get("bell_violation_rate", 0),
                    "practical_limitations": [
                        "Requires stable entanglement",
                        "Technical complexity",
                    ],
                },
                "overall_assessment": {
                    "security": "Highest possible (information-theoretic)",
                    "practicality": "Limited to specific use cases",
                    "cost": "Very high",
                    "scalability": "Poor",
                },
            }

        # Post-quantum analysis
        if pq_results:
            report["post_quantum_analysis"] = {
                "algorithm_comparison": pq_results,
                "deployment_readiness": {
                    "lattice_based": "Production ready (CRYSTALS-Kyber, Dilithium)",
                    "hash_based": "Production ready (XMSS, LMS)",
                    "code_based": "Under standardization",
                    "isogeny_based": "Broken (SIKE attack 2022)",
                },
                "performance_impact": {
                    "key_sizes": "2-10x larger than classical",
                    "computational_overhead": "1.5-5x slower",
                    "bandwidth_overhead": "2-50x larger signatures/ciphertexts",
                },
            }

        # Migration strategy
        report["migration_strategy"] = {
            "phase_1_immediate": [
                "Inventory current cryptographic assets",
                "Identify quantum-vulnerable systems",
                "Begin hybrid classical/post-quantum deployment",
            ],
            "phase_2_transition": [
                "Deploy NIST-standardized PQC algorithms",
                "Implement crypto-agility frameworks",
                "Establish quantum-safe communication channels",
            ],
            "phase_3_full_migration": [
                "Complete migration to post-quantum cryptography",
                "Decommission quantum-vulnerable systems",
                "Monitor quantum computing developments",
            ],
            "timeline": "2024-2030 (recommended)",
            "estimated_cost": "10-30% of IT security budget",
        }

        # Recommendations
        report["recommendations"] = [
            "Immediate: Begin post-quantum cryptography migration planning",
            "High priority: Deploy hybrid classical/PQC systems for critical assets",
            "Medium priority: Implement QKD for highest-security point-to-point links",
            "Ongoing: Monitor NIST PQC standardization and quantum computing progress",
            "Strategic: Invest in crypto-agile architectures for future adaptability",
            "Compliance: Align with emerging quantum-safe regulatory requirements",
        ]

        return report


def visualize_cryptography_results(qkd_metrics, pq_security, security_report):
    """Visualize quantum cryptography and security analysis results."""
    fig = plt.figure(figsize=(16, 12))

    # QKD Performance Metrics
    ax1 = plt.subplot(2, 3, 1)

    if "bb84" in qkd_metrics:
        bb84 = qkd_metrics["bb84"]
        metrics = ["Sifting\nEfficiency", "Security\nParameter", "Key Rate"]
        values = [
            bb84.get("sifting_efficiency", 0),
            bb84.get("security_parameter", 0),
            bb84.get("final_key_length", 0) / bb84.get("initial_bits", 1),
        ]

        bars = ax1.bar(metrics, values, color=["blue", "green", "orange"], alpha=0.7)
        ax1.set_ylabel("Performance")
        ax1.set_title("BB84 QKD Performance")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    # Post-Quantum Security Levels
    ax2 = plt.subplot(2, 3, 2)

    if pq_security:
        algorithms = list(pq_security.keys())
        security_levels = [pq_security[alg]["security_level"] for alg in algorithms]

        colors = ["skyblue", "lightgreen", "lightcoral"]
        bars = ax2.bar(
            algorithms, security_levels, color=colors[: len(algorithms)], alpha=0.7
        )

        ax2.set_ylabel("Security Level (bits)")
        ax2.set_title("Post-Quantum Security Levels")
        ax2.grid(True, alpha=0.3)

        for bar, level in zip(bars, security_levels):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{level}",
                ha="center",
                va="bottom",
            )

    # Classical vs Quantum Security Comparison
    ax3 = plt.subplot(2, 3, 3)

    classical_algs = ["RSA-2048", "ECC-256", "AES-128", "AES-256"]
    classical_security = [112, 128, 128, 256]
    quantum_security = [0, 0, 64, 128]  # After quantum attacks

    x = np.arange(len(classical_algs))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        classical_security,
        width,
        label="Classical Security",
        alpha=0.7,
        color="blue",
    )
    bars2 = ax3.bar(
        x + width / 2,
        quantum_security,
        width,
        label="Post-Quantum Security",
        alpha=0.7,
        color="red",
    )

    ax3.set_ylabel("Security Level (bits)")
    ax3.set_title("Classical vs Post-Quantum Security")
    ax3.set_xticks(x)
    ax3.set_xticklabels(classical_algs, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Quantum Attack Timeline
    ax4 = plt.subplot(2, 3, 4)

    years = [2025, 2030, 2035, 2040, 2045]
    rsa_vulnerability = [0.1, 0.3, 0.6, 0.8, 0.95]  # Probability
    ecc_vulnerability = [0.05, 0.25, 0.55, 0.75, 0.9]
    aes_vulnerability = [0.0, 0.05, 0.1, 0.2, 0.4]

    ax4.plot(years, rsa_vulnerability, "r-o", label="RSA", linewidth=2)
    ax4.plot(years, ecc_vulnerability, "b-s", label="ECC", linewidth=2)
    ax4.plot(years, aes_vulnerability, "g-^", label="AES", linewidth=2)

    ax4.set_xlabel("Year")
    ax4.set_ylabel("Vulnerability Probability")
    ax4.set_title("Quantum Attack Timeline")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    # Key Size Comparison
    ax5 = plt.subplot(2, 3, 5)

    algorithms = ["RSA", "ECC", "Lattice\n(Kyber)", "Hash\n(XMSS)", "Code\n(McEliece)"]
    key_sizes = [2048, 256, 1024, 32, 50000]  # Approximate sizes in bits

    # Normalize to logarithmic scale for visualization
    log_sizes = [np.log10(size) for size in key_sizes]
    colors = ["red", "red", "green", "green", "orange"]

    bars = ax5.bar(algorithms, log_sizes, color=colors, alpha=0.7)
    ax5.set_ylabel("log₁₀(Key Size in bits)")
    ax5.set_title("Cryptographic Key Sizes")
    ax5.grid(True, alpha=0.3)

    # Add actual values as labels
    for bar, size in zip(bars, key_sizes):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{size}b",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Security Analysis Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = "Quantum Cryptography & Security Summary:\n\n"

    # QKD Summary
    if "bb84" in qkd_metrics:
        bb84 = qkd_metrics["bb84"]
        summary_text += f"Quantum Key Distribution (BB84):\n"
        summary_text += f"• Final key: {bb84.get('final_key_length', 0)} bits\n"
        summary_text += f"• QBER: {bb84.get('qber', 0):.3f}\n"
        summary_text += f"• Eve detected: {bb84.get('eve_detected', False)}\n"
        summary_text += f"• Security: Information-theoretic\n\n"

    # Post-Quantum Summary
    if pq_security:
        summary_text += f"Post-Quantum Cryptography:\n"
        for alg, data in pq_security.items():
            summary_text += f"• {alg.title()}: {data['security_level']} bits\n"
        summary_text += f"• NIST standards available\n"
        summary_text += f"• Migration urgency: High\n\n"

    summary_text += "Quantum Threat Assessment:\n\n"
    summary_text += "Current Status:\n"
    summary_text += "• RSA/ECC: Vulnerable to quantum attacks\n"
    summary_text += "• AES: Reduced security (still usable)\n"
    summary_text += "• Hash functions: Reduced preimage security\n\n"

    summary_text += "Mitigation Strategies:\n"
    summary_text += "• Immediate: PQC algorithm deployment\n"
    summary_text += "• High-security: QKD for critical links\n"
    summary_text += "• Long-term: Crypto-agile architectures\n\n"

    summary_text += "Business Impact:\n"
    summary_text += "• Timeline: Migration needed by 2030\n"
    summary_text += "• Cost: 10-30% of security budget\n"
    summary_text += "• Risk: Data confidentiality threats\n"
    summary_text += "• Opportunity: Enhanced security posture"

    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightpink", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Cryptography and Cybersecurity"
    )
    parser.add_argument(
        "--qkd-protocol",
        choices=["bb84", "e91", "both"],
        default="bb84",
        help="QKD protocol to implement",
    )
    parser.add_argument(
        "--key-length", type=int, default=256, help="Quantum key length in bits"
    )
    parser.add_argument(
        "--noise-level", type=float, default=0.05, help="Quantum channel noise level"
    )
    parser.add_argument(
        "--eve-attack", action="store_true", help="Simulate eavesdropping attack"
    )
    parser.add_argument(
        "--post-quantum", action="store_true", help="Analyze post-quantum cryptography"
    )
    parser.add_argument(
        "--security-analysis",
        action="store_true",
        help="Perform comprehensive security analysis",
    )
    parser.add_argument(
        "--migration-planning", action="store_true", help="Generate migration strategy"
    )
    parser.add_argument("--show-visualization", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 8: Industry Applications")
    print("Example 4: Cryptography and Cybersecurity")
    print("=" * 43)

    try:
        qkd_results = {}
        pq_results = {}

        # Quantum Key Distribution
        if args.qkd_protocol in ["bb84", "both"]:
            print(f"\n🔐 Quantum Key Distribution - BB84 Protocol")

            qkd = QuantumKeyDistribution(
                key_length=args.key_length,
                noise_level=args.noise_level,
                verbose=args.verbose,
            )

            # Run BB84 without eavesdropper
            print(f"   Scenario 1: Secure channel")
            shared_key_secure = qkd.bb84_protocol(eve_present=False)

            if shared_key_secure:
                print(f"     ✅ Shared key established: {len(shared_key_secure)} bytes")
                print(f"     Key (first 16 bytes): {shared_key_secure[:16].hex()}")
            else:
                print(f"     ❌ Key establishment failed")

            qkd_results["bb84_secure"] = qkd.security_metrics["bb84"].copy()

            # Run BB84 with eavesdropper
            if args.eve_attack:
                print(f"\n   Scenario 2: Eavesdropping attack")
                qkd_eve = QuantumKeyDistribution(
                    key_length=args.key_length,
                    noise_level=args.noise_level,
                    verbose=args.verbose,
                )

                shared_key_eve = qkd_eve.bb84_protocol(
                    eve_present=True, eve_intercept_rate=0.3
                )

                if qkd_eve.security_metrics["bb84"]["eve_detected"]:
                    print(f"     ✅ Eavesdropping detected! Communication aborted.")
                else:
                    print(f"     ⚠️  Eavesdropping not detected (false negative)")

                qkd_results["bb84_attack"] = qkd_eve.security_metrics["bb84"].copy()

            qkd_results["bb84"] = qkd_results["bb84_secure"]

        if args.qkd_protocol in ["e91", "both"]:
            print(f"\n🔗 Quantum Key Distribution - E91 Protocol")

            qkd_e91 = QuantumKeyDistribution(
                key_length=args.key_length,
                noise_level=args.noise_level,
                verbose=args.verbose,
            )

            shared_key_e91 = qkd_e91.e91_protocol(entanglement_fidelity=0.95)

            if shared_key_e91:
                print(
                    f"     ✅ Entanglement-based key established: {len(shared_key_e91)} bytes"
                )
                print(f"     Key (first 16 bytes): {shared_key_e91[:16].hex()}")

            qkd_results["e91"] = qkd_e91.security_metrics["e91"]

        # Post-Quantum Cryptography
        if args.post_quantum:
            print(f"\n🛡️  Post-Quantum Cryptography Analysis")

            pqc = PostQuantumCryptography(verbose=args.verbose)

            # Test message
            test_message = "Quantum-safe communication test message"

            # Lattice-based encryption
            print(f"\n   Lattice-based Encryption (LWE):")
            lattice_result = pqc.lattice_based_encryption(test_message, n=512, q=8192)

            print(f"     Message length: {lattice_result['message_length']} bits")
            print(f"     Ciphertext blocks: {len(lattice_result['ciphertext'])}")
            print(
                f"     Public key size: ~{len(lattice_result['public_key'][0]) * len(lattice_result['public_key'][0][0]) * 13 // 8} bytes"
            )

            # Hash-based signatures
            print(f"\n   Hash-based Signatures (Merkle):")
            hash_result = pqc.hash_based_signatures(test_message, tree_height=8)

            print(
                f"     Signature components: {len(hash_result['signature']['ots_signature'])}"
            )
            print(
                f"     Authentication path: {len(hash_result['signature']['auth_path'])} nodes"
            )
            print(f"     Merkle root: {hash_result['merkle_root'][:8].hex()}...")

            # Code-based encryption
            print(f"\n   Code-based Encryption (McEliece-style):")
            code_result = pqc.code_based_encryption(test_message, n=1024, k=512, t=50)

            print(f"     Code parameters: [n={1024}, k={512}, t={50}]")
            print(f"     Ciphertext length: {len(code_result['ciphertext'])} bits")
            print(f"     Public key size: ~{512 * 1024 // 8} bytes")

            # Store results
            pq_results = {
                "lattice": pqc.security_analysis["lattice"],
                "hash_based": pqc.security_analysis["hash_based"],
                "code_based": pqc.security_analysis["code_based"],
            }

        # Security Analysis
        if args.security_analysis:
            print(f"\n🔍 Comprehensive Security Analysis")

            analyzer = QuantumSecurityAnalyzer(verbose=args.verbose)

            # Analyze classical algorithms
            classical_algorithms = {
                "RSA-2048": {"modulus_size": 2048},
                "ECC-256": {"field_size": 256},
                "AES-128": {"key_size": 128},
                "AES-256": {"key_size": 256},
            }

            print(f"\n   Classical Algorithm Vulnerability:")
            for alg, params in classical_algorithms.items():
                alg_type = alg.split("-")[0]
                attack_analysis = analyzer.quantum_attack_analysis(alg_type, params)

                print(f"     {alg}:")
                print(f"       Attack: {attack_analysis['attack_algorithm']}")
                print(
                    f"       Classical security: {attack_analysis['time_to_break_classical']}"
                )
                print(
                    f"       Quantum attack: {attack_analysis['time_to_break_quantum']}"
                )
                print(f"       Practical threat: {attack_analysis['practical_threat']}")

            # Post-quantum security assessment
            if pq_results:
                print(f"\n   Post-Quantum Algorithm Assessment:")
                pq_assessment = analyzer.post_quantum_security_assessment(pq_results)

                for alg, assessment in pq_assessment.items():
                    print(f"     {alg.title()}:")
                    print(
                        f"       Security assumption: {assessment['security_assumption']}"
                    )
                    print(
                        f"       Classical attack: {assessment['best_classical_attack']}"
                    )
                    print(f"       Quantum attack: {assessment['best_quantum_attack']}")
                    print(f"       Security level: {assessment['security_level']} bits")
                    print(
                        f"       Quantum resistant: {assessment['quantum_resistant']}"
                    )

            # Generate comprehensive report
            if args.migration_planning:
                print(f"\n📋 Migration Strategy Report")

                security_report = analyzer.generate_security_report(
                    qkd_results,
                    pq_assessment if pq_results else {},
                    classical_algorithms,
                )

                print(f"\n   Executive Summary:")
                exec_summary = security_report["executive_summary"]
                print(
                    f"     Quantum threat timeline: {exec_summary['quantum_threat_timeline']}"
                )
                print(
                    f"     Current vulnerability: {exec_summary['current_vulnerability']}"
                )
                print(f"     Action required: {exec_summary['action_required']}")

                print(f"\n   Migration Timeline:")
                migration = security_report["migration_strategy"]
                print(f"     Phase 1 (Immediate): {migration['phase_1_immediate'][0]}")
                print(
                    f"     Phase 2 (Transition): {migration['phase_2_transition'][0]}"
                )
                print(
                    f"     Phase 3 (Full migration): {migration['phase_3_full_migration'][0]}"
                )
                print(f"     Recommended timeline: {migration['timeline']}")
                print(f"     Estimated cost: {migration['estimated_cost']}")

                print(f"\n   Key Recommendations:")
                for i, rec in enumerate(security_report["recommendations"][:3], 1):
                    print(f"     {i}. {rec}")

        # Visualization
        if args.show_visualization:
            pq_security = {}
            if pq_results:
                analyzer = QuantumSecurityAnalyzer()
                pq_security = analyzer.post_quantum_security_assessment(pq_results)

            security_report = {}
            if args.security_analysis and args.migration_planning:
                security_report = analyzer.generate_security_report(
                    qkd_results, pq_security, {}
                )

            visualize_cryptography_results(qkd_results, pq_security, security_report)

        print(f"\n📚 Key Insights:")
        print(
            f"   • QKD provides information-theoretic security but limited practicality"
        )
        print(
            f"   • Post-quantum algorithms offer quantum resistance with computational assumptions"
        )
        print(f"   • Migration to quantum-safe cryptography is urgent and complex")
        print(
            f"   • Hybrid approaches balance security and performance during transition"
        )

        print(f"\n🎯 Business Impact:")
        print(f"   • Critical infrastructure protection from quantum threats")
        print(f"   • Regulatory compliance with emerging quantum-safe standards")
        print(
            f"   • Competitive advantage through early adoption of quantum-safe systems"
        )
        print(f"   • Long-term data protection and business continuity")

        print(f"\n🚀 Future Opportunities:")
        print(f"   • Quantum internet and distributed quantum computing")
        print(f"   • Quantum-enhanced blockchain and distributed ledgers")
        print(f"   • Post-quantum digital identity and authentication systems")
        print(f"   • Quantum-safe IoT and edge computing security")
        print(f"   • Advanced quantum random number generation")

        print(f"\n✅ Quantum cryptography and security analysis completed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
