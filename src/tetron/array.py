"""
tetron/array.py
---------------
Defines the physical 4×2 (and scalable n×2) tetron array layout.

Convention (Ed Chen, April 2026):
    4×2 means 4 rows, 2 columns.
    Left column  → data qubits    d1–d4  (qubit indices 0–3)
    Right column → ancilla qubits a1–a4  (qubit indices 4–7)

Scaling to 12×2:
    Same pattern: 12 data qubits (0–11) + 12 ancilla qubits (12–23).
    Targets the 12-qubit Google 2019 supremacy circuits.

Physical connectivity:
    A data qubit dᵢ can perform inter-tetron parity measurements with
    its adjacent ancilla aᵢ (same row). Cross-row coupling between
    dᵢ–aᵢ₊₁ or aᵢ–dᵢ₊₁ depends on the specific device routing;
    encode in CONNECTIVITY below once confirmed with mentors.

Qiskit usage:
    The array is simulated as a DensityMatrix over n_rows * 2 qubits.
    Parity measurements are applied as mid-circuit projectors.
    See tetron/parity.py for the measurement primitives.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TetronArray:
    """
    Represents an n_rows × 2 tetron array.

    Parameters
    ----------
    n_rows : int
        Number of tetron rows (4 for the initial array, 12 for the
        full supremacy-circuit array). Default 4.

    Attributes
    ----------
    n_qubits : int
        Total qubit count (n_rows * 2).
    data_qubits : list[int]
        Qubit indices for the data (left) column.
    ancilla_qubits : list[int]
        Qubit indices for the ancilla (right) column.
    connectivity : list[tuple[int, int]]
        (data_idx, ancilla_idx) pairs that can perform inter-tetron
        parity measurements based on physical adjacency.

    Example
    -------
    >>> arr = TetronArray(n_rows=4)
    >>> arr.data_qubits
    [0, 1, 2, 3]
    >>> arr.ancilla_qubits
    [4, 5, 6, 7]
    >>> arr.connectivity   # same-row pairs
    [(0, 4), (1, 5), (2, 6), (3, 7)]
    """

    n_rows: int = 4

    def __post_init__(self):
        self.n_qubits: int = self.n_rows * 2
        self.data_qubits: List[int] = list(range(self.n_rows))
        self.ancilla_qubits: List[int] = list(range(self.n_rows, self.n_rows * 2))

    @property
    def connectivity(self) -> List[Tuple[int, int]]:
        """
        Same-row (dᵢ, aᵢ) pairs available for inter-tetron ZZ/XX
        parity measurement.

        TODO: Extend with cross-row pairs once physical routing is
        confirmed with Ed Chen / Adam Mills.
        """
        return list(zip(self.data_qubits, self.ancilla_qubits))

    def ancilla_for(self, data_idx: int) -> int:
        """Return the ancilla qubit index paired with data_idx."""
        row = self.data_qubits.index(data_idx)
        return self.ancilla_qubits[row]
