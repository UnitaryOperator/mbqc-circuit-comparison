"""
circuits/google2019.py
----------------------
Utilities for loading and working with the Google 2019 quantum supremacy
circuits (Arute et al., Nature 2019).

Google 2019 gate set
--------------------
    - CZ  : native two-qubit entangling gate
    - √X  : single-qubit, = S·H·S  (Clifford)
    - √Y  : single-qubit, = H·√X·H (Clifford)
    - √W  : single-qubit, = (√X + √Y) / √2 class (Clifford)

All gates are Clifford — no T gates, no magic state injection required.
This means the full circuit is implementable using only parity measurements
on the tetron array.

Target circuits
---------------
The smallest circuits in the Google 2019 Dryad dataset are 12-qubit circuits.
For the 4×2 array (8 qubits) we either:
    (a) use synthesised 8-qubit RCS circuits with the same gate set
    (b) extract 8-qubit subgraphs from 12-qubit circuits
Approach TBD — confirm with Ed Chen / Adam Mills.

For the 12×2 array (12 data qubits) we can use the Dryad dataset directly.

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
    import numpy as np
    sv = Statevector.from_instruction(circuit)
    probs = sv.probabilities_dict()
    return probs


# TODO: add helper to extract n-qubit subgraph from a larger Sycamore circuit
# for the 4×2 array validation phase. Confirm approach with mentors first.
