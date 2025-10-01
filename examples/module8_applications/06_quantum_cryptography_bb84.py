#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 8, Example 6
Quantum Cryptography: BB84 Protocol

This example demonstrates the BB84 quantum key distribution protocol, one of
the first and most important quantum cryptography schemes. It shows how quantum
mechanics can be used to detect eavesdropping and establish secure keys.

Learning objectives:
- Understand the BB84 quantum key distribution protocol
- Learn how quantum mechanics enables secure communication
- See how eavesdropping can be detected using quantum properties
- Explore practical quantum cryptography applications

Based on concepts from "Quantum Computing in Action" Chapter 8

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_statevector
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from quantum_helpers import run_circuit_with_shots


def explain_classical_cryptography_problem():
    """Explain the key distribution problem in classical cryptography."""
    print("=== THE CLASSICAL CRYPTOGRAPHY CHALLENGE ===")
    print()

    print("🔐 THE KEY DISTRIBUTION PROBLEM:")
    print("Alice wants to send Bob a secret message securely.")
    print()

    print("Classical encryption (like AES) is very strong, BUT...")
    print("Alice and Bob need to share a secret key first!")
    print()

    print("📡 INSECURE CLASSICAL CHANNELS:")
    print("- Any classical communication can be intercepted")
    print("- Eve (eavesdropper) can copy classical bits perfectly")
    print("- No way to detect if someone is listening")
    print("- Once Eve has the key, all future messages are compromised")
    print()

    print("💡 QUANTUM SOLUTION IDEA:")
    print("What if we could use quantum mechanics to:")
    print("1. Detect if someone is eavesdropping")
    print("2. Generate truly random keys")
    print("3. Make copying impossible (no-cloning theorem)")
    print()


def demonstrate_one_time_pad():
    """Demonstrate the one-time pad encryption method."""
    print("=== ONE-TIME PAD ENCRYPTION ===")
    print()

    print("🔑 PERFECT SECRECY WITH ONE-TIME PAD:")
    print("If Alice and Bob share a truly random secret key...")
    print()

    # Example message
    message = "HELLO"
    print(f"Message: {message}")

    # Convert to binary
    message_bits = "".join(format(ord(c), "08b") for c in message)
    print(f"Message in binary: {message_bits}")

    # Generate random key (same length)
    np.random.seed(42)  # For reproducible demo
    key_bits = "".join(np.random.choice(["0", "1"]) for _ in range(len(message_bits)))
    print(f"Random key:       {key_bits}")

    # XOR encryption
    encrypted_bits = "".join(
        str(int(m) ^ int(k)) for m, k in zip(message_bits, key_bits)
    )
    print(f"Encrypted:        {encrypted_bits}")

    # Decryption (XOR again)
    decrypted_bits = "".join(
        str(int(e) ^ int(k)) for e, k in zip(encrypted_bits, key_bits)
    )
    print(f"Decrypted:        {decrypted_bits}")

    # Convert back to text
    decrypted_chars = []
    for i in range(0, len(decrypted_bits), 8):
        byte = decrypted_bits[i : i + 8]
        decrypted_chars.append(chr(int(byte, 2)))

    decrypted_message = "".join(decrypted_chars)
    print(f"Decrypted message: {decrypted_message}")
    print()

    print("✅ One-time pad properties:")
    print("- Mathematically unbreakable (if key is truly random)")
    print("- Key must be as long as the message")
    print("- Key must never be reused")
    print("- Problem: How to securely share the random key?")
    print()


def create_bb84_protocol():
    """Implement the complete BB84 quantum key distribution protocol."""
    print("=== BB84 QUANTUM KEY DISTRIBUTION PROTOCOL ===")
    print()

    print("👥 PARTICIPANTS:")
    print("- Alice: Wants to send a secret key to Bob")
    print("- Bob: Wants to receive the secret key from Alice")
    print("- Eve: Potential eavesdropper trying to intercept the key")
    print()

    # Protocol parameters
    n_bits = 20  # Number of qubits to send

    print(f"🎲 STEP 1: Alice generates {n_bits} random bits")
    np.random.seed(42)  # For reproducible demo
    alice_bits = np.random.randint(0, 2, n_bits)
    print(f"Alice's random bits: {alice_bits}")
    print()

    print("📐 STEP 2: Alice chooses random basis for each qubit")
    alice_bases = np.random.randint(0, 2, n_bits)  # 0=Z basis, 1=X basis
    print(f"Alice's bases (0=Z, 1=X): {alice_bases}")
    print()

    print("🔮 STEP 3: Alice prepares qubits in chosen states")
    print("Encoding rules:")
    print("- Z basis: |0⟩ for bit 0, |1⟩ for bit 1")
    print("- X basis: |+⟩ for bit 0, |-⟩ for bit 1")
    print()

    # Create quantum circuits for Alice's qubits
    alice_circuits = []
    for i in range(n_bits):
        qc = QuantumCircuit(1, 1)

        if alice_bases[i] == 0:  # Z basis
            if alice_bits[i] == 1:
                qc.x(0)  # |1⟩
            # |0⟩ is default, no gate needed
        else:  # X basis
            if alice_bits[i] == 0:
                qc.h(0)  # |+⟩ = H|0⟩
            else:
                qc.x(0)  # |-⟩ = HX|0⟩ = H|1⟩
                qc.h(0)

        alice_circuits.append(qc)

    print("📡 STEP 4: Alice sends qubits to Bob over quantum channel")
    print("(In reality, this uses photons sent through optical fibers)")
    print()

    print("🎯 STEP 5: Bob chooses random measurement bases")
    bob_bases = np.random.randint(0, 2, n_bits)
    print(f"Bob's bases (0=Z, 1=X): {bob_bases}")
    print()

    print("📏 STEP 6: Bob measures qubits in his chosen bases")

    # Bob's measurements
    bob_results = []
    simulator = AerSimulator()

    for i in range(n_bits):
        qc = alice_circuits[i].copy()

        # Bob's measurement basis
        if bob_bases[i] == 1:  # X basis measurement
            qc.h(0)  # Convert X basis to Z basis

        qc.measure(0, 0)

        # Run simulation
        transpiled = transpile(qc, simulator)
        job = simulator.run(transpiled, shots=1)
        result = job.result()
        counts = result.get_counts()

        # Get measurement result
        measured_bit = int(list(counts.keys())[0])
        bob_results.append(measured_bit)

    bob_results = np.array(bob_results)
    print(f"Bob's measurement results: {bob_results}")
    print()

    print("📞 STEP 7: Public discussion (classical channel)")
    print("Alice and Bob publicly compare their basis choices")
    print("They keep only bits where they used the same basis")
    print()

    # Find matching bases
    matching_bases = alice_bases == bob_bases
    shared_key_alice = alice_bits[matching_bases]
    shared_key_bob = bob_results[matching_bases]

    print(f"Matching basis positions: {np.where(matching_bases)[0]}")
    print(f"Alice's key bits: {shared_key_alice}")
    print(f"Bob's key bits:   {shared_key_bob}")
    print()

    # Check for errors (should be very low in ideal case)
    errors = np.sum(shared_key_alice != shared_key_bob)
    error_rate = errors / len(shared_key_alice) if len(shared_key_alice) > 0 else 0

    print(f"Number of errors: {errors}")
    print(f"Error rate: {error_rate:.2%}")
    print()

    print("🔍 STEP 8: Error checking and privacy amplification")
    if error_rate < 0.11:  # BB84 error threshold
        print("✅ Low error rate - channel appears secure")
        print("Alice and Bob can use this key for encryption")
    else:
        print("❌ High error rate - possible eavesdropping detected!")
        print("Alice and Bob should abort and try again")

    return shared_key_alice, shared_key_bob, error_rate


def demonstrate_eavesdropping_detection():
    """Show how BB84 can detect eavesdropping attempts."""
    print("=== EAVESDROPPING DETECTION ===")
    print()

    print("🕵️ Eve tries to intercept the quantum communication...")
    print()

    print("EVE'S STRATEGY:")
    print("1. Intercept each qubit Alice sends")
    print("2. Measure it to learn the bit value")
    print("3. Prepare a new qubit in the measured state")
    print("4. Send the new qubit to Bob")
    print()

    print("THE PROBLEM WITH EVE'S STRATEGY:")
    print("Eve doesn't know Alice's basis choice!")
    print("She must guess the measurement basis...")
    print()

    # Simulate eavesdropping
    n_bits = 100
    np.random.seed(123)  # Different seed for eavesdropping demo

    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)
    eve_bases = np.random.randint(0, 2, n_bits)  # Eve guesses randomly
    bob_bases = np.random.randint(0, 2, n_bits)

    print("🔬 QUANTUM MECHANICS TO THE RESCUE:")
    print("When Eve measures in wrong basis:")
    print("- Superposition collapses randomly")
    print("- She gets random result (50% chance wrong)")
    print("- She forwards wrong state to Bob")
    print("- Bob sees additional errors!")
    print()

    # Calculate expected error rate with eavesdropping
    print("ERROR RATE CALCULATION:")
    print("Without Eve: ~0% errors (perfect quantum channel)")
    print("With Eve intercepting everything:")

    # When Alice and Bob use same basis, but Eve used wrong basis
    # This happens with probability 1/4 and introduces 50% error rate
    expected_error_rate = 0.25 * 0.5  # 25% of qubits × 50% error rate

    print(f"Expected error rate: {expected_error_rate:.2%}")
    print("(Above the 11% security threshold!)")
    print()

    print("🚨 DETECTION MECHANISM:")
    print("1. Alice and Bob measure error rate in shared key")
    print("2. If error rate > 11%, eavesdropping likely detected")
    print("3. They abort the protocol and try again")
    print("4. Eve learns she was detected!")
    print()


def bb84_security_analysis():
    """Analyze the security properties of BB84."""
    print("=== BB84 SECURITY ANALYSIS ===")
    print()

    print("🔒 SECURITY FEATURES:")
    print()

    print("1. 🚫 NO-CLONING THEOREM:")
    print("   - Eve cannot perfectly copy unknown quantum states")
    print("   - Any copying attempt introduces errors")
    print("   - Fundamental quantum mechanical protection")
    print()

    print("2. 📏 MEASUREMENT DISTURBANCE:")
    print("   - Measuring a qubit changes its state")
    print("   - Eve's measurements disturb Alice's qubits")
    print("   - Disturbance shows up as errors in Bob's results")
    print()

    print("3. 🎲 BASIS RANDOMIZATION:")
    print("   - Alice and Bob choose random bases")
    print("   - Eve cannot predict the correct measurement basis")
    print("   - Wrong basis → random results → detectable errors")
    print()

    print("4. 🔍 ERROR THRESHOLD:")
    print("   - Error rate < 11%: Secure key can be extracted")
    print("   - Error rate > 11%: Eavesdropping likely detected")
    print("   - Mathematical security proof exists")
    print()

    print("⚠️  PRACTICAL LIMITATIONS:")
    print("- Requires perfect quantum channels (no loss)")
    print("- Distance limited by photon attenuation")
    print("- Detector efficiency and dark count issues")
    print("- Side-channel attacks on hardware")
    print("- Classical post-processing required")
    print()


def visualize_bb84_protocol():
    """Create visualizations showing the BB84 protocol."""
    print("Creating BB84 visualizations...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Qubit preparation and measurement
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)

    # Alice's preparation
    ax1.text(1, 7, "Alice's Qubit Preparation", fontsize=14, fontweight="bold")
    ax1.text(0.5, 6, "Bit 0, Z basis → |0⟩", fontsize=12)
    ax1.text(0.5, 5.5, "Bit 1, Z basis → |1⟩", fontsize=12)
    ax1.text(0.5, 5, "Bit 0, X basis → |+⟩", fontsize=12)
    ax1.text(0.5, 4.5, "Bit 1, X basis → |−⟩", fontsize=12)

    # Bob's measurement
    ax1.text(1, 3.5, "Bob's Measurement", fontsize=14, fontweight="bold")
    ax1.text(0.5, 2.5, "Z basis: measures in |0⟩, |1⟩", fontsize=12)
    ax1.text(0.5, 2, "X basis: measures in |+⟩, |−⟩", fontsize=12)

    ax1.set_title("BB84 Qubit States and Measurements")

    # 2. Error rate vs eavesdropping
    eavesdrop_levels = np.linspace(0, 1, 11)
    error_rates = eavesdrop_levels * 0.25  # Simplified model

    ax2.plot(
        eavesdrop_levels * 100, error_rates * 100, "b-o", linewidth=2, markersize=6
    )
    ax2.axhline(
        y=11, color="red", linestyle="--", linewidth=2, label="Security Threshold (11%)"
    )
    ax2.set_xlabel("Percentage of Qubits Eve Intercepts (%)")
    ax2.set_ylabel("Error Rate (%)")
    ax2.set_title("Error Rate vs Eavesdropping Level")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Key generation efficiency
    total_qubits = [50, 100, 200, 500, 1000]
    efficiency_no_eve = [
        q * 0.5 * 0.95 for q in total_qubits
    ]  # 50% basis match, 95% kept
    efficiency_with_eve = [
        q * 0.5 * 0.6 for q in total_qubits
    ]  # More discarded due to errors

    x = np.arange(len(total_qubits))
    width = 0.35

    ax3.bar(x - width / 2, efficiency_no_eve, width, label="Secure Channel", alpha=0.8)
    ax3.bar(x + width / 2, efficiency_with_eve, width, label="Under Attack", alpha=0.8)

    ax3.set_xlabel("Total Qubits Sent")
    ax3.set_ylabel("Final Key Bits")
    ax3.set_title("Key Generation Efficiency")
    ax3.set_xticks(x)
    ax3.set_xticklabels(total_qubits)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. BB84 protocol flow
    ax4.axis("off")
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    steps = [
        "1. Alice generates random bits",
        "2. Alice chooses random bases",
        "3. Alice prepares qubits",
        "4. Alice sends qubits to Bob",
        "5. Bob chooses random bases",
        "6. Bob measures qubits",
        "7. Public basis comparison",
        "8. Error rate checking",
        "9. Key extraction/abortion",
    ]

    for i, step in enumerate(steps):
        y_pos = 9 - i
        ax4.text(
            0.5,
            y_pos,
            step,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

    ax4.set_title("BB84 Protocol Steps", fontsize=14, fontweight="bold")

    plt.suptitle(
        "BB84 Quantum Key Distribution Protocol", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="BB84 Quantum Cryptography Protocol")
    parser.add_argument(
        "--key-length", type=int, default=20, help="Number of qubits to use in protocol"
    )
    parser.add_argument(
        "--skip-visualization", action="store_true", help="Skip the visualization plots"
    )
    parser.add_argument(
        "--show-eavesdropping",
        action="store_true",
        help="Demonstrate eavesdropping detection",
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 8: Applications")
    print("Example 6: Quantum Cryptography - BB84 Protocol")
    print("=" * 55)

    try:
        print("\n🔐 Welcome to Quantum Cryptography!")
        print("Learn how quantum mechanics enables perfectly secure communication.")
        print()

        # Classical cryptography challenge
        explain_classical_cryptography_problem()

        # One-time pad demonstration
        demonstrate_one_time_pad()

        # BB84 protocol implementation
        alice_key, bob_key, error_rate = create_bb84_protocol()

        # Eavesdropping detection
        if args.show_eavesdropping:
            demonstrate_eavesdropping_detection()

        # Security analysis
        bb84_security_analysis()

        # Summary
        print("🎓 KEY TAKEAWAYS:")
        print("=" * 40)
        print("1. 🔑 Quantum mechanics enables secure key distribution")
        print("2. 🚫 No-cloning theorem prevents perfect eavesdropping")
        print("3. 📏 Measurement disturbance reveals eavesdroppers")
        print("4. 🎲 Random basis choices provide security")
        print("5. 🔍 Error rates indicate channel security")
        print("6. 🌐 Quantum cryptography is commercially deployed")
        print()

        if not args.skip_visualization:
            visualize_bb84_protocol()

        print("✅ BB84 quantum cryptography demonstration completed!")
        print()
        print("🌟 REAL-WORLD STATUS:")
        print("- BB84 is commercially available today")
        print("- Companies: ID Quantique, MagiQ, Toshiba")
        print("- Distances: ~100-200 km over fiber")
        print("- Applications: Banking, government, critical infrastructure")
        print()
        print("💡 Next Steps:")
        print("- Explore other quantum protocols (E91, B92)")
        print("- Learn about post-quantum cryptography")
        print("- Study practical implementations and challenges")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
