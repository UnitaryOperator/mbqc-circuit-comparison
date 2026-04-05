"""
tetron/parity.py
----------------
Joint fermion parity measurement primitives for the tetron array.

Background
----------
In the tetron architecture, all operations are joint parity measurements
of MZM pairs. The two native inter-tetron operations are:

    ZZ(i, j):  measure iγ₂ᵢ γ₁ʲ  — produces ±1 outcome
    XX(i, j):  measure iγ₂ᵢ γ₃ʲ  — produces ±1 outcome

These are QND (quantum non-demolition): if the state is already a ZZ
eigenstate (e.g. a Bell state |Φ+⟩), the outcome is deterministic and
the state is unchanged (projection, not collapse).

In Qiskit DensityMatrix, a parity measurement on qubits (i, j) in the
ZZ basis is implemented as:

    1. Apply CNOT(i → ancilla scratch qubit) and CNOT(j → ancilla)
       to copy the ZZ parity into an ancilla bit, OR
    2. Directly project ρ → (I + (-1)^s ZᵢZⱼ) ρ (I + (-1)^s ZᵢZⱼ) / 2
       for outcome s ∈ {0, 1}.

This module uses approach 2: explicit density matrix projection.
The outcome s is sampled from Tr[(I + ZᵢZⱼ)/2 · ρ] = prob(s=0).

Pauli frame
-----------
Each measurement outcome s feeds into the PauliFrame (see pauli_frame.py).
No physical Pauli correction gates are ever applied.

Returns
-------
All measurement functions return (rho_post, outcome) where:
    rho_post : DensityMatrix — post-measurement state
    outcome  : int           — 0 (+1 eigenvalue) or 1 (−1 eigenvalue)
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Pauli, SparsePauliOp


def _project(rho: DensityMatrix, pauli_str: str, outcome: int) -> DensityMatrix:
    """
    Project density matrix ρ onto the ±1 eigenspace of a Pauli operator P.

    Projector: Π_s = (I + (-1)^s P) / 2   for s ∈ {0, 1}

    Parameters
    ----------
    rho        : DensityMatrix — current state
    pauli_str  : str          — Pauli string in Qiskit convention (rightmost = qubit 0)
    outcome    : int          — 0 for +1 eigenvalue, 1 for -1 eigenvalue

    Returns
    -------
    DensityMatrix — normalised post-projection state
    """
    n = rho.num_qubits
    P = SparsePauliOp(Pauli(pauli_str)).to_matrix()
    I = np.eye(2**n)
    sign = (-1) ** outcome
    proj = (I + sign * P) / 2
    rho_mat = np.array(rho.data)
    rho_post = proj @ rho_mat @ proj
    norm = np.real(np.trace(rho_post))
    if norm < 1e-12:
        raise ValueError(
            f"Projection onto outcome={outcome} has zero probability. "
            "Check that the outcome is consistent with the current state."
        )
    return DensityMatrix(rho_post / norm)


def _sample_outcome(rho: DensityMatrix, pauli_str: str) -> int:
    """
    Sample a measurement outcome (0 or 1) for Pauli operator P on ρ.

    P(outcome=0) = Tr[Π₀ ρ] = (1 + ⟨P⟩) / 2
    """
    P = SparsePauliOp(Pauli(pauli_str)).to_matrix()
    rho_mat = np.array(rho.data)
    expectation = np.real(np.trace(P @ rho_mat))
    prob_zero = (1 + expectation) / 2
    prob_zero = np.clip(prob_zero, 0, 1)
    return int(np.random.choice([0, 1], p=[prob_zero, 1 - prob_zero]))


def _pauli_string_zz(n_qubits: int, i: int, j: int) -> str:
    """
    Build Qiskit-convention Pauli string for ZᵢZⱼ on n_qubits.
    Qiskit convention: string index 0 = highest qubit index (right-to-left)
    """
    ops = ["I"] * n_qubits
    ops[n_qubits - 1 - i] = "Z"
    ops[n_qubits - 1 - j] = "Z"
    return "".join(ops)


def _pauli_string_xx(n_qubits: int, i: int, j: int) -> str:
    ops = ["I"] * n_qubits
    ops[n_qubits - 1 - i] = "X"
    ops[n_qubits - 1 - j] = "X"
    return "".join(ops)


def _pauli_string_yy(n_qubits: int, i: int, j: int) -> str:
    ops = ["I"] * n_qubits
    ops[n_qubits - 1 - i] = "Y"
    ops[n_qubits - 1 - j] = "Y"
    return "".join(ops)


def measure_zz(
    rho: DensityMatrix,
    i: int,
    j: int,
    forced_outcome: int | None = None,
) -> tuple[DensityMatrix, int]:
    """
    Perform a ZZ joint parity measurement on qubits i and j.

    This is the primary inter-tetron entangling operation. When applied
    to a ZZ eigenstate (e.g. a Bell state), the outcome is deterministic
    and the state is unchanged — the QND property.

    Parameters
    ----------
    rho            : DensityMatrix
    i, j           : int  — qubit indices (data and ancilla qubits)
    forced_outcome : int | None
        If provided (0 or 1), forces this outcome instead of sampling.
        Useful for deterministic tests and Pauli frame debugging.

    Returns
    -------
    (rho_post, outcome) : (DensityMatrix, int)
    """
    n = rho.num_qubits
    pauli_str = _pauli_string_zz(n, i, j)
    outcome = forced_outcome if forced_outcome is not None else _sample_outcome(rho, pauli_str)
    rho_post = _project(rho, pauli_str, outcome)
    return rho_post, outcome


def measure_xx(
    rho: DensityMatrix,
    i: int,
    j: int,
    forced_outcome: int | None = None,
) -> tuple[DensityMatrix, int]:
    """
    Perform an XX joint parity measurement on qubits i and j.

    Used for X-basis initialization of ancilla qubits (|+⟩ state) and
    for the XX entangling step in single-qubit gate teleportation.

    Returns
    -------
    (rho_post, outcome) : (DensityMatrix, int)
    """
    n = rho.num_qubits
    pauli_str = _pauli_string_xx(n, i, j)
    outcome = forced_outcome if forced_outcome is not None else _sample_outcome(rho, pauli_str)
    rho_post = _project(rho, pauli_str, outcome)
    return rho_post, outcome


def measure_yy(
    rho: DensityMatrix,
    i: int,
    j: int,
    forced_outcome: int | None = None,
) -> tuple[DensityMatrix, int]:
    """
    Perform a YY joint parity measurement on qubits i and j.

    Used for iSWAP-type two-qubit operations in the gate set.

    Returns
    -------
    (rho_post, outcome) : (DensityMatrix, int)
    """
    n = rho.num_qubits
    pauli_str = _pauli_string_yy(n, i, j)
    outcome = forced_outcome if forced_outcome is not None else _sample_outcome(rho, pauli_str)
    rho_post = _project(rho, pauli_str, outcome)
    return rho_post, outcome


def measure_z(
    rho: DensityMatrix,
    i: int,
    forced_outcome: int | None = None,
) -> tuple[DensityMatrix, int]:
    """
    Single-qubit Z measurement on qubit i.

    Used for:
    - Z-basis initialization of data qubits (|0⟩ state)
    - Z-basis readout at end of circuit
    - Intra-tetron parity check (iγ₁γ₂ on a single tetron)

    Returns
    -------
    (rho_post, outcome) : (DensityMatrix, int)
    """
    n = rho.num_qubits
    ops = ["I"] * n
    ops[n - 1 - i] = "Z"
    pauli_str = "".join(ops)
    outcome = forced_outcome if forced_outcome is not None else _sample_outcome(rho, pauli_str)
    rho_post = _project(rho, pauli_str, outcome)
    return rho_post, outcome


def measure_x(
    rho: DensityMatrix,
    i: int,
    forced_outcome: int | None = None,
) -> tuple[DensityMatrix, int]:
    """
    Single-qubit X measurement on qubit i.

    Used for X-basis initialization of a single ancilla qubit to |+⟩.

    Returns
    -------
    (rho_post, outcome) : (DensityMatrix, int)
    """
    n = rho.num_qubits
    ops = ["I"] * n
    ops[n - 1 - i] = "X"
    pauli_str = "".join(ops)
    outcome = forced_outcome if forced_outcome is not None else _sample_outcome(rho, pauli_str)
    rho_post = _project(rho, pauli_str, outcome)
    return rho_post, outcome
