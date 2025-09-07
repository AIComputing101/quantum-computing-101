#!/usr/bin/env python3
"""
Quantum Computing 101 - Module 1, Example 8
Hardware Reality Check

This example provides a realistic overview of current quantum computing hardware
capabilities, limitations, and timeline expectations. Essential for newcomers
to understand what quantum computers can and cannot do today.

Learning objectives:
- Understand current quantum hardware limitations
- Set realistic expectations about quantum computing
- Learn about different types of quantum computers
- Explore the timeline for practical quantum advantage

Based on concepts from "Quantum Computing in Action" Chapter 1

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_statevector
import sys
from pathlib import Path


def quantum_vs_classical_reality():
    """Compare current quantum vs classical computing capabilities."""
    print("=== QUANTUM vs CLASSICAL COMPUTING REALITY ===")
    print()

    print("📱 CLASSICAL COMPUTERS TODAY:")
    print("✅ Billions of operations per second")
    print("✅ Error rates: ~10^-17 (essentially perfect)")
    print("✅ Massive storage: terabytes to petabytes")
    print("✅ Global connectivity and networking")
    print("✅ 70+ years of optimization and engineering")
    print("✅ Cost: $500 - $50,000 for high-end systems")
    print()

    print("🌟 QUANTUM COMPUTERS TODAY:")
    print("⚠️  Thousands to millions of operations per second")
    print("❌ Error rates: ~0.1% - 1% (very noisy!)")
    print("❌ Storage: 50-1000 qubits (very limited)")
    print("❌ Isolated systems (no quantum internet yet)")
    print("⚠️  ~20 years of serious development")
    print("💰 Cost: $10 million - $100 million+")
    print()

    print("🎯 CURRENT QUANTUM ADVANTAGE:")
    print("✅ Specific research problems (quantum simulation)")
    print("✅ Proof-of-concept demonstrations")
    print("❌ NO practical advantage for everyday computing")
    print("❌ NO threat to current cryptography (yet)")
    print("❌ NO quantum internet or networking")
    print()


def current_quantum_hardware_overview():
    """Overview of current quantum hardware platforms."""
    print("=== CURRENT QUANTUM HARDWARE PLATFORMS ===")
    print()

    platforms = {
        "Superconducting (IBM, Google)": {
            "qubits": "50-1000+ qubits",
            "temp": "~15 millikelvin (-273.135°C)",
            "coherence": "~100 microseconds",
            "gates": "~10-100 nanoseconds",
            "pros": ["Fast gates", "Good connectivity", "Scalable"],
            "cons": ["Extremely cold", "Short coherence", "Complex control"],
        },
        "Trapped Ion (IonQ, Honeywell)": {
            "qubits": "10-100 qubits",
            "temp": "Room temperature lasers",
            "coherence": "~10 seconds",
            "gates": "~10-100 microseconds",
            "pros": ["Long coherence", "High fidelity", "Universal gates"],
            "cons": ["Slower gates", "Limited scaling", "Complex lasers"],
        },
        "Photonic (Xanadu, PsiQuantum)": {
            "qubits": "10-200+ modes",
            "temp": "Room temperature",
            "coherence": "Infinite (no decoherence)",
            "gates": "Speed of light",
            "pros": ["No decoherence", "Network compatible", "Fast"],
            "cons": ["Probabilistic gates", "Detection losses", "Limited operations"],
        },
        "Neutral Atom (QuEra, Pasqal)": {
            "qubits": "100-1000 atoms",
            "temp": "~microkelvin",
            "coherence": "~1 second",
            "gates": "~1 microsecond",
            "pros": ["Flexible connectivity", "Analog & digital", "Scalable"],
            "cons": ["Complex control", "Loading losses", "New technology"],
        },
    }

    for platform, specs in platforms.items():
        print(f"🔧 {platform}:")
        print(f"   Qubits: {specs['qubits']}")
        print(f"   Temperature: {specs['temp']}")
        print(f"   Coherence time: {specs['coherence']}")
        print(f"   Gate time: {specs['gates']}")
        print(f"   Pros: {', '.join(specs['pros'])}")
        print(f"   Cons: {', '.join(specs['cons'])}")
        print()


def demonstrate_current_limitations():
    """Show current quantum computing limitations with examples."""
    print("=== CURRENT LIMITATIONS DEMONSTRATION ===")
    print()

    print("🔥 CHALLENGE 1: QUANTUM DECOHERENCE")
    print("Quantum states are extremely fragile...")
    print()

    # Simulate decoherence
    qc = QuantumCircuit(5)
    qc.h(0)  # Create superposition
    for i in range(1, 5):
        qc.cx(0, i)  # Create entanglement

    print("Creating a 5-qubit entangled state...")
    print("In a perfect world: |00000⟩ + |11111⟩")
    print()

    print("But in reality:")
    print("- After ~100 microseconds: coherence starts to decay")
    print("- Random bit flips occur ~0.1% of the time per operation")
    print("- Phase errors accumulate continuously")
    print("- Final state: mostly noise!")
    print()

    print("🔥 CHALLENGE 2: LIMITED CONNECTIVITY")
    print("Not all qubits can interact directly...")
    print()

    # Show connectivity limitations
    qc_limited = QuantumCircuit(5)
    qc_limited.cx(0, 1)  # Adjacent qubits only
    qc_limited.cx(1, 2)
    qc_limited.cx(2, 3)
    qc_limited.cx(3, 4)

    print("Typical connectivity: linear chain or 2D grid")
    print("Want to entangle qubit 0 with qubit 4?")
    print("Need: 0→1→2→3→4 (many operations = more errors)")
    print()

    print("🔥 CHALLENGE 3: MEASUREMENT ERRORS")
    print("Even reading results is imperfect...")
    print()

    print("Readout fidelity: ~97-99%")
    print("Measuring |0⟩ might give |1⟩ ~1-3% of the time")
    print("For n qubits, error probability ≈ n × single_qubit_error")
    print("100 qubits → ~1-3% chance of wrong answer!")
    print()


def quantum_advantage_timeline():
    """Realistic timeline for quantum advantage in different domains."""
    print("=== QUANTUM ADVANTAGE TIMELINE ===")
    print()

    timeline = {
        "2019-2024 (NISQ Era)": {
            "achievements": [
                "✅ Quantum supremacy demonstrations",
                "✅ Small-scale quantum simulations",
                "✅ Proof-of-concept algorithms",
                "⚠️  Research and development focus",
            ],
            "limitations": [
                "❌ No practical applications yet",
                "❌ High error rates (0.1-1%)",
                "❌ Limited qubit counts (<1000)",
                "❌ No error correction",
            ],
        },
        "2025-2030 (Near-term)": {
            "achievements": [
                "🎯 Quantum advantage in optimization",
                "🎯 Drug discovery applications",
                "🎯 Materials science simulations",
                "🎯 Early error correction demos",
            ],
            "limitations": [
                "⚠️  Still mostly research",
                "⚠️  Specialized applications only",
                "⚠️  Expensive and complex",
                "⚠️  No quantum internet",
            ],
        },
        "2030-2040 (Medium-term)": {
            "achievements": [
                "🚀 Fault-tolerant quantum computers",
                "🚀 Quantum machine learning advantage",
                "🚀 Cryptographically relevant QCs",
                "🚀 Quantum networking protocols",
            ],
            "limitations": [
                "💰 Still very expensive",
                "🏭 Specialized facilities required",
                "📚 New algorithms needed",
                "👨‍💻 Specialized expertise required",
            ],
        },
        "2040+ (Long-term)": {
            "achievements": [
                "🌟 Practical quantum advantage",
                "🌟 Quantum internet",
                "🌟 Commercial applications",
                "🌟 Quantum-classical hybrid systems",
            ],
            "limitations": [
                "❓ Classical computers also improve",
                "❓ New physical challenges",
                "❓ Standardization needed",
                "❓ Education and workforce",
            ],
        },
    }

    for period, details in timeline.items():
        print(f"📅 {period}:")
        print("   Expected achievements:")
        for achievement in details["achievements"]:
            print(f"     {achievement}")
        print("   Remaining challenges:")
        for limitation in details["limitations"]:
            print(f"     {limitation}")
        print()


def what_quantum_computers_cannot_do():
    """Important reality check on quantum computing limitations."""
    print("=== WHAT QUANTUM COMPUTERS CANNOT DO ===")
    print()

    print("❌ QUANTUM COMPUTERS WILL NOT:")
    print()

    print("🖥️  REPLACE CLASSICAL COMPUTERS:")
    print("   - No advantage for word processing, web browsing, games")
    print("   - Classical computers are better for most everyday tasks")
    print("   - Quantum computers are specialized tools")
    print()

    print("⚡ MAKE ALL ALGORITHMS FASTER:")
    print("   - Quantum speedup only for specific problem types")
    print("   - Many problems have no known quantum advantage")
    print("   - Some problems are provably not faster on quantum computers")
    print()

    print("🔐 BREAK ALL ENCRYPTION IMMEDIATELY:")
    print("   - Current quantum computers are too small and noisy")
    print("   - Post-quantum cryptography is being developed")
    print("   - Timeline: 15-25 years for cryptographically relevant QCs")
    print()

    print("🧠 SOLVE ALL AI PROBLEMS:")
    print("   - Quantum machine learning is still experimental")
    print("   - Most ML algorithms don't have quantum advantage")
    print("   - Classical ML hardware (GPUs) continues improving rapidly")
    print()

    print("💊 IMMEDIATELY DISCOVER NEW DRUGS:")
    print("   - Quantum chemistry is promising but early stage")
    print("   - Current quantum computers too small for real molecules")
    print("   - Classical molecular simulation also improving")
    print()


def practical_advice_for_beginners():
    """Practical advice for those starting to learn quantum computing."""
    print("=== PRACTICAL ADVICE FOR BEGINNERS ===")
    print()

    print("🎓 SHOULD YOU LEARN QUANTUM COMPUTING?")
    print()

    print("✅ YES, IF YOU ARE:")
    print("   - Interested in cutting-edge technology")
    print("   - Working in research or academia")
    print("   - In cryptography, optimization, or simulation fields")
    print("   - Planning a 10-20 year career horizon")
    print("   - Curious about fundamental physics and mathematics")
    print()

    print("⚠️  THINK CAREFULLY IF YOU:")
    print("   - Expect immediate practical applications")
    print("   - Need quantum advantage for current projects")
    print("   - Have limited math/physics background")
    print("   - Are looking for quick career changes")
    print()

    print("📚 LEARNING PATH RECOMMENDATIONS:")
    print("1. 🎯 Start with simulators (like this course!)")
    print("2. 📐 Build strong linear algebra foundation")
    print("3. 🧮 Learn quantum algorithms (Grover, Shor, VQE)")
    print("4. 💻 Practice with real quantum hardware")
    print("5. 🔬 Specialize in application domain")
    print("6. 🤝 Join quantum computing community")
    print()

    print("💼 CAREER OPPORTUNITIES:")
    print("   - Quantum software engineer")
    print("   - Quantum algorithm researcher")
    print("   - Quantum hardware engineer")
    print("   - Quantum cryptography specialist")
    print("   - Quantum educator/consultant")
    print()


def visualize_quantum_hardware_trends():
    """Visualize quantum hardware progress and projections."""
    print("Creating hardware trends visualization...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Qubit count over time
    years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    qubits = np.array(
        [5, 20, 50, 65, 65, 127, 433, 1000, 1121]
    )  # Approximate IBM progress

    ax1.plot(years, qubits, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Qubits")
    ax1.set_title("Quantum Computer Qubit Count Progress")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Add projection
    future_years = np.array([2025, 2026, 2027, 2028, 2029, 2030])
    future_qubits = np.array([2000, 4000, 8000, 16000, 32000, 64000])  # Projected
    ax1.plot(
        future_years, future_qubits, "r--", linewidth=2, alpha=0.7, label="Projected"
    )
    ax1.legend()

    # 2. Error rates over time
    error_years = np.array([2016, 2018, 2020, 2022, 2024])
    error_rates = np.array([5, 1, 0.5, 0.1, 0.05])  # Percentage

    ax2.plot(error_years, error_rates, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Error Rate (%)")
    ax2.set_title("Quantum Gate Error Rate Improvement")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Add threshold line
    ax2.axhline(
        y=0.01,
        color="green",
        linestyle="--",
        label="Error Correction Threshold",
        linewidth=2,
    )
    ax2.legend()

    # 3. Quantum Volume comparison
    companies = ["IBM", "Google", "IonQ", "Honeywell\n(Quantinuum)", "Rigetti"]
    quantum_volumes = [512, 256, 4096, 65536, 128]  # Approximate 2024 values

    bars = ax3.bar(
        companies, quantum_volumes, color=["blue", "red", "green", "orange", "purple"]
    )
    ax3.set_ylabel("Quantum Volume")
    ax3.set_title("Quantum Volume by Company (2024)")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, quantum_volumes):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value}",
            ha="center",
            va="bottom",
        )

    # 4. Cost vs Performance
    performance = [100, 500, 1000, 2000, 5000]  # Relative performance
    cost_millions = [50, 30, 20, 15, 10]  # Cost in millions USD

    ax4.scatter(performance, cost_millions, s=200, alpha=0.7)
    ax4.set_xlabel("Relative Performance (Quantum Volume)")
    ax4.set_ylabel("Cost (Millions USD)")
    ax4.set_title("Cost vs Performance Trend")
    ax4.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(performance, cost_millions, 1)
    p = np.poly1d(z)
    ax4.plot(performance, p(performance), "r--", alpha=0.8, linewidth=2)

    plt.suptitle(
        "Quantum Computing Hardware Reality Check", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Quantum Hardware Reality Check")
    parser.add_argument(
        "--skip-visualization", action="store_true", help="Skip the visualization plots"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed hardware specifications"
    )

    args = parser.parse_args()

    print("Quantum Computing 101 - Module 1: Fundamental Concepts")
    print("Example 8: Hardware Reality Check")
    print("=" * 50)

    try:
        print("\n🚨 REALITY CHECK: Where Quantum Computing Stands Today")
        print("Setting realistic expectations for newcomers...")
        print()

        # Current reality comparison
        quantum_vs_classical_reality()

        # Hardware platforms overview
        if args.detailed:
            current_quantum_hardware_overview()

        # Current limitations
        demonstrate_current_limitations()

        # Timeline expectations
        quantum_advantage_timeline()

        # What QC cannot do
        what_quantum_computers_cannot_do()

        # Practical advice
        practical_advice_for_beginners()

        # Key summary
        print("🎯 BOTTOM LINE SUMMARY:")
        print("=" * 40)
        print("🔬 Quantum computing is REAL science with REAL potential")
        print("⏰ But practical applications are still 5-15 years away")
        print("🎓 Learning now positions you for the future")
        print("💡 Focus on fundamentals, not hype")
        print("🤝 Join the community, but manage expectations")
        print()

        if not args.skip_visualization:
            visualize_quantum_hardware_trends()

        print("✅ Hardware reality check completed!")
        print()
        print("💡 Next Steps:")
        print("- Explore quantum algorithms in Module 4")
        print("- Learn about real hardware in Module 7")
        print("- Check out quantum applications in Module 8")
        print("- But remember: we're still in the early research phase!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
