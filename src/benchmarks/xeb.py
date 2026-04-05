"""
benchmarks/xeb.py
-----------------
Linear Cross-Entropy Benchmarking (XEB) scorer and LCED computation.

XEB (Arute et al. 2019)
-----------------------
The linear XEB score measures how close an experimental (or simulated)
output distribution P_exp is to the ideal distribution P_ideal:

    XEB = 2^n · E_x~P_exp[ p_ideal(x) ] − 1

where n is the number of qubits and the expectation is taken over
bitstrings x sampled from the experimental device/simulator.

Interpretation:
    XEB = 1.0  → perfect agreement with ideal (noiseless simulation)
    XEB = 0.0  → outputs match uniform random (complete decoherence)
    XEB < 0    → worse than random (possible for highly structured noise)

LCED (Linear Cross-Entropy Difference)
---------------------------------------
    LCED = XEB_MBQC − XEB_circuit

Positive LCED: tetron MBQC preserves fidelity better than the circuit model.
Negative LCED: circuit model outperforms MBQC at the same noise level.

References
----------
    Arute et al. (2019), Nature 574, 505–510. Supplementary §IV.
"""

import numpy as np
from typing import Sequence


def xeb_score(
    sampled_bitstrings: Sequence[str],
    ideal_probs: dict[str, float],
) -> float:
    """
    Compute the linear XEB score.

    Parameters
    ----------
    sampled_bitstrings : sequence of bitstrings (e.g. ['0110', '1001', ...])
        Output bitstrings sampled from the device or MBQC simulator.
    ideal_probs : dict[str, float]
        Ideal output probability for each bitstring, p_ideal(x) = |⟨x|ψ⟩|².
        Computed from Qiskit Statevector on the noiseless circuit.

    Returns
    -------
    float — XEB score in [-1, 1]. Noiseless ideal = 1.0, random = 0.0.

    Notes
    -----
    n is inferred from the length of the first bitstring.
    Bitstrings not in ideal_probs are assigned probability 0 (shouldn't
    occur for complete ideal_probs dictionaries).
    """
    if not sampled_bitstrings:
        raise ValueError("sampled_bitstrings is empty")

    n = len(sampled_bitstrings[0])
    n_shots = len(sampled_bitstrings)

    p_values = np.array([ideal_probs.get(b, 0.0) for b in sampled_bitstrings])
    return float(2**n * np.mean(p_values) - 1)


def lced(xeb_mbqc: float, xeb_circuit: float) -> float:
    """
    Compute LCED = XEB_MBQC − XEB_circuit.

    Parameters
    ----------
    xeb_mbqc    : float — XEB score from tetron MBQC simulation
    xeb_circuit : float — XEB score from circuit model simulation

    Returns
    -------
    float — LCED value
    """
    return xeb_mbqc - xeb_circuit


def sample_bitstrings(
    probs: dict[str, float],
    n_shots: int,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """
    Sample bitstrings from a probability distribution.

    Parameters
    ----------
    probs   : dict[str, float] — probability distribution over bitstrings
    n_shots : int              — number of shots to sample
    rng     : np.random.Generator | None — for reproducibility

    Returns
    -------
    list[str] — sampled bitstrings
    """
    if rng is None:
        rng = np.random.default_rng()
    bitstrings = list(probs.keys())
    probabilities = np.array([probs[b] for b in bitstrings])
    probabilities /= probabilities.sum()  # normalise for floating point safety
    indices = rng.choice(len(bitstrings), size=n_shots, p=probabilities)
    return [bitstrings[i] for i in indices]


def xeb_vs_noise(
    noise_rates: Sequence[float],
    circuit,
    simulate_fn,
    ideal_probs: dict[str, float],
    n_shots: int = 10_000,
) -> dict[float, float]:
    """
    Sweep XEB score vs noise rate ε.

    Parameters
    ----------
    noise_rates  : sequence of float — noise rate values to sweep
    circuit      : the circuit to simulate (QuantumCircuit or equivalent)
    simulate_fn  : callable(circuit, noise_rate) -> list[str]
        Function that simulates the circuit at a given noise rate and
        returns a list of sampled bitstrings.
    ideal_probs  : dict[str, float] — ideal output probabilities
    n_shots      : int — shots per noise rate

    Returns
    -------
    dict mapping noise_rate -> XEB score
    """
    results = {}
    for eps in noise_rates:
        bitstrings = simulate_fn(circuit, eps)
        results[eps] = xeb_score(bitstrings, ideal_probs)
    return results
