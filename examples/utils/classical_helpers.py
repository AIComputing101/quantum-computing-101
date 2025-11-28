"""Classical Helpers
===================
Utility functions for classical baseline comparisons and simple analytics.

Functions:
- time_function(func, *args, repeat=3, **kwargs)
- basic_stats(data)
- hamming_distance(a: str, b: str)
- probability_distribution(counts_dict)

Lightweight and dependency-minimal (numpy optional if available).
"""

from __future__ import annotations
import time
from statistics import mean, pstdev
from typing import Callable, Any, Dict, List

try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None  # type: ignore


def time_function(func: Callable, *args, repeat: int = 3, **kwargs) -> Dict[str, Any]:
    """Time a function multiple times and return summary statistics."""
    durations: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args, **kwargs)
        durations.append(time.perf_counter() - start)
    return {
        "runs": repeat,
        "mean": mean(durations),
        "stdev": pstdev(durations) if repeat > 1 else 0.0,
        "min": min(durations),
        "max": max(durations),
        "raw": durations,
    }


def basic_stats(data: List[float]) -> Dict[str, float]:
    """Return simple statistics for numeric data."""
    if not data:
        raise ValueError("data must be non-empty")
    if np:
        arr = np.asarray(data)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "count": float(arr.size),
        }
    else:  # fallback
        return {
            "mean": mean(data),
            "std": pstdev(data) if len(data) > 1 else 0.0,
            "min": min(data),
            "max": max(data),
            "count": float(len(data)),
        }


def hamming_distance(a: str, b: str) -> int:
    """
    Compute Hamming distance between two equal-length strings.
    
    Hamming distance = number of positions where bits differ
    Example: "1010" vs "1110" → distance = 1
    
    Used in quantum error correction to measure error rates.
    """
    if len(a) != len(b):
        raise ValueError("Strings must be same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def probability_distribution(counts: Dict[str, int]) -> Dict[str, float]:
    """Normalize a counts dictionary to probabilities."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Total counts must be > 0")
    return {k: v / total for k, v in counts.items()}


if __name__ == "__main__":
    print("Timing demo:", time_function(sum, [1, 2, 3, 4, 5]))
    print("Stats demo:", basic_stats([1, 2, 3, 4, 5]))
    print("Hamming demo:", hamming_distance("1010", "1110"))
    print("Prob dist demo:", probability_distribution({"00": 5, "01": 5, "10": 10}))
