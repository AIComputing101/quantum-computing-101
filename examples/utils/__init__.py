"""Quantum Computing 101 - Shared Utilities Package

This package provides reusable helper modules used by example scripts:

Modules:
 - visualization: plotting helpers (Bloch spheres, histograms, comparisons)
 - quantum_helpers: small circuit/state preparation & analysis utilities
 - classical_helpers: classical benchmarking and analytics helpers
 - educational_tools: concept explanation, lightweight progress & quiz helpers

Import examples:
    from utils.visualization import plot_bloch_comparison
    from utils.quantum_helpers import create_bell_state
    from utils.educational_tools import explain_concept, create_quiz, grade_quiz

The package keeps dependencies minimal and aligned with the main examples.
"""

from .visualization import (
    plot_bloch_comparison,
    plot_measurement_comparison,
    plot_probability_evolution,
    plot_quantum_circuit_with_explanation,
    plot_correlation_matrix,
    plot_quantum_vs_classical,
    create_educational_animation_frames,
    plot_algorithm_complexity,
    save_figure_with_metadata,
    setup_plot_style,
)

from .quantum_helpers import (
    create_bell_state,
    prepare_plus_state,
    apply_random_single_qubit_rotation,
    measure_all,
    analyze_state,
)

from .classical_helpers import (
    time_function,
    basic_stats,
    hamming_distance,
    probability_distribution,
)

from .educational_tools import (
    explain_concept,
    format_equation,
    checkpoint,
    QuizQuestion,
    create_quiz,
    administer_quiz,
    grade_quiz,
)

__all__ = [
    # visualization
    "plot_bloch_comparison",
    "plot_measurement_comparison",
    "plot_probability_evolution",
    "plot_quantum_circuit_with_explanation",
    "plot_correlation_matrix",
    "plot_quantum_vs_classical",
    "create_educational_animation_frames",
    "plot_algorithm_complexity",
    "save_figure_with_metadata",
    "setup_plot_style",
    # quantum helpers
    "create_bell_state",
    "prepare_plus_state",
    "apply_random_single_qubit_rotation",
    "measure_all",
    "analyze_state",
    # classical helpers
    "time_function",
    "basic_stats",
    "hamming_distance",
    "probability_distribution",
    # educational tools
    "explain_concept",
    "format_equation",
    "checkpoint",
    "QuizQuestion",
    "create_quiz",
    "administer_quiz",
    "grade_quiz",
]
