"""
tetron/pauli_frame.py
---------------------
Classical Pauli frame tracker for the tetron MBQC simulation.

What is the Pauli frame?
------------------------
Every parity measurement in MBQC produces a random ±1 outcome that,
if uncorrected, leaves a Pauli byproduct on the logical state. Rather
than physically applying X/Z gates after each measurement (which would
require real-time classical control and add circuit depth), we track
a classical bit-pair (x_bit, z_bit) per qubit representing pending
corrections:

    x_bit = 1 → a pending X correction on this qubit
    z_bit = 1 → a pending Z correction on this qubit

These bits are propagated through the remaining circuit symbolically,
and folded into the final measurement outcomes at readout. No physical
gates are ever applied to the quantum state.

Update rules (Clifford propagation)
------------------------------------
When the frame (xᵢ, zᵢ) on qubit i propagates through a subsequent
parity measurement, it transforms under conjugation. Key rules:

    After ZZ(i,j) with outcome s:
        x-byproduct on i:  zᵢ ⊕= s         (Z correction from ZZ outcome)
        x-byproduct on j:  zⱼ ⊕= s

    After X measurement on ancilla aᵢ with outcome s₂:
        x-byproduct on data dᵢ: xᵢ ⊕= s₂   (X correction from X-basis meas)

    CNOT(ctrl=i, tgt=j):
        xᵢ unchanged,  zᵢ ⊕= zⱼ
        xⱼ ⊕= xᵢ,      zⱼ unchanged

Readout correction
------------------
At the end of the circuit, measurement outcome mᵢ ∈ {0,1} is corrected:
    corrected_mᵢ = mᵢ ⊕ zᵢ   (for Z-basis readout)
    corrected_mᵢ = mᵢ ⊕ xᵢ   (for X-basis readout)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PauliFrame:
    """
    Tracks accumulated Pauli byproducts (X and Z) per qubit.

    Parameters
    ----------
    n_qubits : int — total number of qubits in the simulation

    Attributes
    ----------
    x_frame : dict[int, int] — x_bit per qubit (0 or 1)
    z_frame : dict[int, int] — z_bit per qubit (0 or 1)

    Example
    -------
    >>> frame = PauliFrame(n_qubits=8)
    >>> frame.update_from_zz(data_qubit=0, ancilla_qubit=4, outcome=1)
    >>> frame.update_from_x_measurement(data_qubit=0, outcome=0)
    >>> frame.correct_z_readout(qubit=0, raw_outcome=1)
    1
    """

    n_qubits: int
    x_frame: Dict[int, int] = field(default_factory=dict)
    z_frame: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for q in range(self.n_qubits):
            self.x_frame[q] = 0
            self.z_frame[q] = 0

    def update_from_zz(self, data_qubit: int, ancilla_qubit: int, outcome: int) -> None:
        """
        Update frame after ZZ(data_qubit, ancilla_qubit) with given outcome.

        A ZZ measurement with outcome s introduces a Z byproduct on both
        qubits if s=1. Record this in the Z frame.
        """
        if outcome == 1:
            self.z_frame[data_qubit] ^= 1
            self.z_frame[ancilla_qubit] ^= 1

    def update_from_xx(self, data_qubit: int, ancilla_qubit: int, outcome: int) -> None:
        """
        Update frame after XX(data_qubit, ancilla_qubit) with given outcome.

        An XX measurement with outcome s introduces an X byproduct on both.
        """
        if outcome == 1:
            self.x_frame[data_qubit] ^= 1
            self.x_frame[ancilla_qubit] ^= 1

    def update_from_x_measurement(self, data_qubit: int, outcome: int) -> None:
        """
        Update frame after measuring ancilla qubit in X basis with given outcome.

        In gate teleportation step 3, measuring the ancilla in the X basis
        with outcome s₂ introduces an X byproduct on the data qubit.
        """
        if outcome == 1:
            self.x_frame[data_qubit] ^= 1

    def update_from_y_measurement(self, data_qubit: int, outcome: int) -> None:
        """
        Update frame after measuring ancilla qubit in Y basis with given outcome.

        A Y-basis measurement outcome introduces both X and Z byproducts
        on the data qubit (since Y = iXZ).
        """
        if outcome == 1:
            self.x_frame[data_qubit] ^= 1
            self.z_frame[data_qubit] ^= 1

    def correct_z_readout(self, qubit: int, raw_outcome: int) -> int:
        """
        Apply Z-frame correction to a Z-basis readout outcome.

        corrected = raw_outcome XOR z_bit
        """
        return raw_outcome ^ self.z_frame[qubit]

    def correct_x_readout(self, qubit: int, raw_outcome: int) -> int:
        """
        Apply X-frame correction to an X-basis readout outcome.

        corrected = raw_outcome XOR x_bit
        """
        return raw_outcome ^ self.x_frame[qubit]

    def get_frame(self, qubit: int) -> Tuple[int, int]:
        """Return (x_bit, z_bit) for qubit."""
        return self.x_frame[qubit], self.z_frame[qubit]

    def reset_qubit(self, qubit: int) -> None:
        """Reset frame for a qubit (e.g. after ancilla is discarded and re-initialised)."""
        self.x_frame[qubit] = 0
        self.z_frame[qubit] = 0

    def __repr__(self) -> str:
        pairs = [(q, self.x_frame[q], self.z_frame[q]) for q in range(self.n_qubits)]
        lines = [f"  q{q}: X={x} Z={z}" for q, x, z in pairs if x or z]
        body = "\n".join(lines) if lines else "  (all zero)"
        return f"PauliFrame(\n{body}\n)"
