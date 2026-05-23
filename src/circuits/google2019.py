"""
circuits/google2019.py
----------------------
Utilities for loading and working with the Google 2019 quantum supremacy
circuits (Arute et al., Nature 2019) for the MBQC circuit comparison project.

Google 2019 gate set
--------------------
    - SX   : single-qubit, √X = R_x(π/2)
    - SY   : single-qubit, √Y = R_y(π/2)
    - SW   : single-qubit, √W where W is a rotation around the (X+Y)/√2 axis
    - fSIM : two-qubit entangling gate parameterized by (θ, φ); the native
             Sycamore two-qubit gate, with per-gate calibrated angles in the
             supremacy dataset

Project scope
-------------
We target the 30 published 12-qubit circuits from the Google 2019 Dryad
dataset, running them on a simulated 8×3 tetron array that reproduces
Sycamore's rectangular-lattice nearest-neighbor connectivity. Each circuit
is translated gate-by-gate into MBQC parity-measurement equivalents, then
decomposed into Qiskit-compatible form for noiseless simulation and XEB /
LCED benchmarking against Google's published bitstrings.

This module contains two kinds of utilities:
    - Circuit loaders (.qasm via Qiskit, .json via Cirq)
    - Reference-output loaders for Google's published amplitudes files,
      which give the noiseless |amp|^2 = p_ideal distribution per circuit
      and can be used directly for XEB scoring without recomputing the
      statevector.

References
----------
    Arute et al. (2019), Nature 574, 505–510.
    Dataset: https://datadryad.org/dataset/doi:10.5061/dryad.k6t1rj8
    Cirq: Google's native framework with Sycamore gate definitions.
"""

from pathlib import Path
from typing import Optional

try:
    import cirq
    import cirq_google
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# ---------------------------------------------------------------------------
# Circuit loaders
# ---------------------------------------------------------------------------

def load_openqasm(path: str | Path) -> "QuantumCircuit":
    """
    Load a Google 2019 circuit from an OpenQASM 2.0 file.

    Parameters
    ----------
    path : str | Path — path to a .qasm file from the Dryad dataset

    Returns
    -------
    QuantumCircuit — Qiskit circuit object

    Raises
    ------
    ImportError  if Qiskit is not installed
    FileNotFoundError  if path does not exist
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required: pip install qiskit")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Circuit file not found: {path}")
    return QuantumCircuit.from_qasm_file(str(path))


def load_cirq_circuit(path: str | Path) -> "cirq.Circuit":
    """
    Load a Google 2019 circuit from a JSON file using Cirq.

    The Dryad dataset includes Cirq-native JSON format circuits.
    Cirq is used here only for circuit loading and ideal probability
    computation (XEB reference) — not for MBQC simulation.

    Returns
    -------
    cirq.Circuit
    """
    if not HAS_CIRQ:
        raise ImportError("Cirq is required: pip install cirq-core")
    import json
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return cirq.read_json(json_text=json.dumps(data))


# ---------------------------------------------------------------------------
# Ideal-output helpers
# ---------------------------------------------------------------------------

def ideal_probabilities(circuit: "QuantumCircuit") -> dict[str, float]:
    """
    Compute ideal output bitstring probabilities using Qiskit Statevector.

    Used as the reference distribution for XEB scoring.
    p_ideal(x) = |⟨x|ψ⟩|²  for each bitstring x.

    Parameters
    ----------
    circuit : QuantumCircuit — circuit WITHOUT measurement gates

    Returns
    -------
    dict mapping bitstring (e.g. '0110') → probability (float)
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required")
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(circuit)
    probs = sv.probabilities_dict()
    return probs


# ---------------------------------------------------------------------------
# Reference amplitudes (Google's published noiseless output)
# ---------------------------------------------------------------------------

def load_amplitudes(path: str | Path) -> dict[str, complex]:
    """
    Load a Google 2019 amplitudes file into a bitstring → complex map.

    File format
    -----------
    One bitstring per line, three whitespace-separated columns:
        <bitstring>   <real_part>   <imag_part>

    Example line:
        100001000001    0.0198028199    0.0106442748

    The leftmost character of the bitstring corresponds to whichever qubit
    Google's serialization put first. We do NOT convert to an integer
    index here to avoid silently introducing endianness bugs; if you need
    integer indices, do `int(bitstring, 2)` (or reverse first if your
    convention has qubit 0 on the right, as Qiskit does).

    Notes
    -----
    The file may contain all 2^N bitstrings or only a subset, depending on
    which variant was published for that circuit. The returned dict
    reflects whatever was in the file. Downstream code should not assume
    every bitstring is present.

    Parameters
    ----------
    path : str | Path — path to amplitudes_n{N}_m{M}_s{S}_e{E}_pEFGH.txt

    Returns
    -------
    dict[str, complex] — bitstring → amplitude
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Amplitudes file not found: {path}")
    amplitudes: dict[str, complex] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            bitstring = parts[0]
            real = float(parts[1])
            imag = float(parts[2])
            amplitudes[bitstring] = complex(real, imag)
    return amplitudes


def load_probabilities(path: str | Path) -> dict[str, float]:
    """
    Load a Google 2019 amplitudes file and return |amp|² probabilities.

    Convenience wrapper around `load_amplitudes` for the common case of
    XEB scoring, where the phase information isn't needed.

    Parameters
    ----------
    path : str | Path — path to amplitudes_n{N}_m{M}_s{S}_e{E}_pEFGH.txt

    Returns
    -------
    dict[str, float] — bitstring → probability
    """
    return {bs: abs(amp) ** 2 for bs, amp in load_amplitudes(path).items()}