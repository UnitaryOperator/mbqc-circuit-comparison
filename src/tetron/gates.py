"""
src/tetron/gates.py
--------------------
MBQC gate library for the tetron array simulator.

All functions extracted verbatim from Two_qubit_gate_supremacy.ipynb
(April 2026, authored by the team). This module makes those functions
importable by builder.py and tests without modifying the notebook.

Qubit convention (from notebook)
----------------------------------
Each logical qubit i occupies two Qiskit wire indices:
    data qubit  → 2*i
    ancilla     → 2*i+1

Example for a 4×2 array (N=4 logical qubits, 8 Qiskit qubits total):
    logical 0: data=0, ancilla=1
    logical 1: data=2, ancilla=3
    logical 2: data=4, ancilla=5
    logical 3: data=6, ancilla=7

Classical bit convention
--------------------------
All gate functions share a single ClassicalRegister of 8 bits.
Bits 0–4 are used by single-qubit Clifford sequences.
Bits 5–7 are reserved for add_CNOT and its callers.
The bits are reused across sequential gate applications because
each conditional is evaluated immediately after its measurement.

Y convention
-------------
Throughout this module:  Y = S X S†
This convention is set by add_H → add_YI and must be consistent
across all Pauli tracking operations.

References
-----------
- Two_qubit_gate_supremacy.ipynb (team notebook, April 2026)
- Karzig et al., PRB 95, 235305 (2017)  — tetron parity gate decomposition
- Arute et al., Nature 574, 505-510 (2019) — fSim gate definition
"""

import numpy as np
from qiskit.circuit.classical import expr


# ---------------------------------------------------------------------------
# Parity measurement primitives
# ---------------------------------------------------------------------------

def measure_ZZ(qc, q0, q1, cbit):
    """
    ZZ parity measurement on qubits q0, q1.
    Implemented via CNOT: CX(q0→q1), measure q1, CX(q0→q1).
    Result written to cbit: 0 = even parity (+1), 1 = odd parity (-1).
    """
    qc.cx(q0, q1)
    qc.measure(q1, cbit)
    qc.cx(q0, q1)


def measure_XI(qc, q0, q1, cbit):
    """
    X measurement on q0 (q1 unused).
    Rotates into X basis via H, measures, rotates back.
    """
    qc.h(q0)
    qc.measure(q0, cbit)
    qc.h(q0)


def measure_YI(qc, q0, q1, cbit):
    """
    Y measurement on q0 (q1 unused).
    Convention: Y = S X S†.
    Rotates into Y basis via S†, measures in X basis, rotates back.
    """
    qc.sdg(q0)
    measure_XI(qc, q0, q1, cbit)
    qc.s(q0)


def measure_ZX(qc, q0, q1, cbit):
    """
    ZX joint parity measurement: Z on q0, X on q1.
    Implemented by rotating q1 into Z basis (H), measuring ZZ, rotating back.
    """
    qc.h(q1)
    measure_ZZ(qc, q0, q1, cbit)
    qc.h(q1)


def measure_ZY(qc, q0, q1, cbit):
    """
    ZY joint parity measurement: Z on q0, Y on q1.
    Convention: Y = S X S† = S H Z H S†.
    """
    qc.sdg(q1)
    qc.h(q1)
    measure_ZZ(qc, q0, q1, cbit)
    qc.h(q1)
    qc.s(q1)


# ---------------------------------------------------------------------------
# Single-qubit Clifford building blocks
# ---------------------------------------------------------------------------

def add_H(qc, d, a, cbit):
    """
    Apply Hadamard gate to data qubit d using ancilla a.
    Measurement sequence: XI, ZY, YI, XI on (a, d).
    Feedforward: Y if parity(c0⊕c1⊕c2) = 0, then unconditional X.
    Uses cbit[0:4].
    """
    c = cbit
    measure_XI(qc, a, d, c[0])
    measure_ZY(qc, a, d, c[1])
    measure_YI(qc, a, d, c[2])
    measure_XI(qc, a, d, c[3])

    parity = expr.bit_xor(expr.bit_xor(c[0], c[1]), c[2])
    with qc.if_test(expr.logic_not(parity)):
        qc.y(d)
    qc.x(d)

    qc.reset(a)
    return qc


def add_S(qc, d, a, cbit):
    """
    Apply S gate (= √Z) to data qubit d using ancilla a.
    Measurement sequence: XI, ZZ, YI, XI on (a, d).
    Feedforward: Z if parity(c0⊕c1⊕c2) = 0.
    Uses cbit[0:4].
    """
    c = cbit
    measure_XI(qc, a, d, c[0])
    measure_ZZ(qc, a, d, c[1])
    measure_YI(qc, a, d, c[2])
    measure_XI(qc, a, d, c[3])

    parity = expr.bit_xor(expr.bit_xor(c[0], c[1]), c[2])
    with qc.if_test(expr.logic_not(parity)):
        qc.z(d)

    qc.reset(a)
    return qc


def add_SH(qc, d, a, cbit):
    """
    Apply SH gate to data qubit d using ancilla a.
    Measurement sequence: XI, ZZ, ZY, YI, XI on (a, d).
    Uses cbit[0:5].
    """
    c = cbit
    measure_XI(qc, a, d, c[0])
    measure_ZZ(qc, a, d, c[1])
    measure_ZY(qc, a, d, c[2])
    measure_YI(qc, a, d, c[3])
    measure_XI(qc, a, d, c[4])

    parity_023 = expr.bit_xor(expr.bit_xor(c[0], c[2]), c[3])
    with qc.if_test(parity_023):
        qc.y(d)

    parity_12 = expr.bit_xor(c[1], c[2])
    with qc.if_test(parity_12):
        qc.z(d)

    qc.reset(a)
    return qc


def add_HSH(qc, d, a, cbit):
    """
    Apply HSH gate to data qubit d using ancilla a.
    Implements √X = HSH.
    Measurement sequence: XI, ZZ, ZY, XI on (a, d).
    Uses cbit[0:4].
    """
    c = cbit
    measure_XI(qc, a, d, c[0])
    measure_ZZ(qc, a, d, c[1])
    measure_ZY(qc, a, d, c[2])
    measure_XI(qc, a, d, c[3])

    parity_03 = expr.bit_xor(c[0], c[3])
    with qc.if_test(parity_03):
        qc.y(d)

    parity_12 = expr.bit_xor(c[1], c[2])
    with qc.if_test(expr.logic_not(parity_12)):
        qc.x(d)

    qc.reset(a)
    return qc


def add_HS(qc, d, a, cbit):
    """
    Apply HS gate to data qubit d using ancilla a.
    Measurement sequence: XI, ZY, ZZ, YI, XI on (a, d).
    Uses cbit[0:5].
    """
    c = cbit
    measure_XI(qc, a, d, c[0])
    measure_ZY(qc, a, d, c[1])
    measure_ZZ(qc, a, d, c[2])
    measure_YI(qc, a, d, c[3])
    measure_XI(qc, a, d, c[4])

    parity_12 = expr.bit_xor(c[1], c[2])
    with qc.if_test(expr.logic_not(parity_12)):
        qc.x(d)

    parity_013 = expr.bit_xor(expr.bit_xor(c[0], c[1]), c[3])
    with qc.if_test(expr.logic_not(parity_013)):
        qc.z(d)

    qc.reset(a)
    return qc


# ---------------------------------------------------------------------------
# Single-qubit gates used in Google 2019 supremacy circuits
# ---------------------------------------------------------------------------

def add_sqrtX(qc, d, a, cbit):
    """
    Apply √X gate to data qubit d using ancilla a.
    Decomposition: √X = HSH (4 parity measurements).
    """
    return add_HSH(qc, d, a, cbit)


def add_sqrtY(qc, d, a, cbit):
    """
    Apply √Y gate to data qubit d using ancilla a.
    Decomposition: √Y = S · HS (7 parity measurements).
    """
    qc = add_S(qc, d, a, cbit)
    qc = add_HS(qc, d, a, cbit)
    return qc


def add_sqrtW(qc, d, a, cbit):
    """
    Apply √W gate to data qubit d using ancilla a.
    √W = T† · √X · T, where T = Rz(π/4).

    Note on T gates: in simulation T/T† are applied as physical Rz
    rotations on the data qubit. On actual tetron hardware these would
    require a separate non-Clifford implementation (magic state injection).
    For the noiseless Qiskit simulation they are treated as virtual
    frame operations with no physical cost.
    """
    qc.tdg(d)
    qc = add_sqrtX(qc, d, a, cbit)
    qc.t(d)
    return qc


# ---------------------------------------------------------------------------
# Two-qubit gate building blocks
# ---------------------------------------------------------------------------

def add_CNOT(qc, c, a, t, cbit):
    """
    Apply CNOT gate (control=c, target=t) via shared ancilla a.

    Ancilla routing convention:
        - ZX measurement: c → a  (vertical edge: control to ancilla)
        - ZX measurement: a → t  (horizontal edge: ancilla to target)
        - X measurement: a       (ancilla measured out in X basis)

    Uses cbit[5:8] to avoid collision with single-qubit gate bits [0:5].
    Feedforward: X on t if (P1 ⊕ M), Z on c if P2.
    """
    qc.reset(a)
    measure_ZX(qc, c, a, cbit[5])   # P1
    measure_ZX(qc, a, t, cbit[6])   # P2
    qc.h(a)
    qc.measure(a, cbit[7])           # M (X measurement on ancilla)
    qc.h(a)
    qc.reset(a)

    P1 = cbit[5]
    P2 = cbit[6]
    M  = cbit[7]

    with qc.if_test(expr.bit_xor(P1, M)):
        qc.x(t)
    with qc.if_test((P2, 1)):
        qc.z(c)

    return qc


def add_iSWAPdg(qc, c, a, t, theta, cbit):
    """
    Apply iSWAP†(θ) gate between data qubits c and t via ancilla a.

    iSWAP†(θ) = exp(+iθ(XX+YY)/2)

    Convention: uses the supremacy paper definition where iSWAP†
    corresponds to the (XX+YY)/2 generator with +θ, matching
    fSim(θ, φ) when combined with CPhase(φ).

    Note: qc.rz(theta, t) inside this function is non-Clifford for
    arbitrary θ. For θ=π/2 (our idealised case), Rz(π/2) = S which
    IS Clifford. In simulation both work identically.
    """
    qc = add_H(qc, c, a, cbit)
    qc = add_H(qc, t, a, cbit)
    qc = add_CNOT(qc, c, a, t, cbit)
    qc.rz(theta, t)
    qc = add_CNOT(qc, c, a, t, cbit)
    qc = add_H(qc, c, a, cbit)
    qc = add_H(qc, t, a, cbit)

    # S† = S³ (three S gates)
    for _ in range(3):
        qc = add_S(qc, c, a, cbit)
    for _ in range(3):
        qc = add_S(qc, t, a, cbit)

    qc = add_H(qc, c, a, cbit)
    qc = add_H(qc, t, a, cbit)
    qc = add_CNOT(qc, c, a, t, cbit)
    qc.rz(theta, t)
    qc = add_CNOT(qc, c, a, t, cbit)
    qc = add_SH(qc, c, a, cbit)
    qc = add_SH(qc, t, a, cbit)

    return qc


def add_CPhase(qc, c, a, t, phi, cbit):
    """
    Apply controlled-phase gate CPhase(φ) between data qubits c and t.
    Implemented as: Rz(-φ/2) on c, CNOT, Rz(φ/2) on t, CNOT, Rz(-φ/2) on t.

    Convention: matches the supremacy paper sign convention where
    fSim(θ, φ) has e^{-iφ} on the |11⟩ component.

    Note: for φ=π/6 (our idealised case), the Rz(±π/12) angles are
    non-Clifford. Valid for noiseless simulation; on hardware would
    require magic state injection.
    """
    qc.rz(-0.5 * phi, c)
    qc = add_CNOT(qc, c, a, t, cbit)
    qc.rz(0.5 * phi, t)
    qc = add_CNOT(qc, c, a, t, cbit)
    qc.rz(-0.5 * phi, t)
    return qc


def add_TwoQubitGate(qc, c, a, t, theta, phi, Z1, Z2, Z3, Z4, cbit):
    """
    Apply the full FSimGate(θ, φ) between data qubits c and t via ancilla a.

    FSimGate(θ, φ) = iSWAP†(θ) · CPhase(φ)

    Full signature matches the notebook exactly:
        c     : control data qubit (Qiskit wire index)
        a     : shared ancilla qubit (Qiskit wire index)
        t     : target data qubit (Qiskit wire index)
        theta : iSWAP† rotation angle (π/2 for idealised noiseless circuits)
        phi   : CPhase angle (π/6 for idealised noiseless circuits)
        Z1    : pre-gate Rz on c (set 0 for noiseless)
        Z2    : pre-gate Rz on t (set 0 for noiseless)
        Z3    : post-gate Rz on c (set 0 for noiseless)
        Z4    : post-gate Rz on t (set 0 for noiseless)
        cbit  : ClassicalRegister of size >= 8

    For noiseless idealised simulation: Z1=Z2=Z3=Z4=0, θ=π/2, φ=π/6.
    """
    qc.rz(Z1, c)
    qc.rz(Z2, t)
    qc = add_iSWAPdg(qc, c, a, t, theta, cbit)
    qc = add_CPhase(qc, c, a, t, phi, cbit)
    qc.rz(Z3, c)
    qc.rz(Z4, t)
    return qc


# ---------------------------------------------------------------------------
# Reference unitary (for verification)
# ---------------------------------------------------------------------------

def fSim(theta, phi):
    """
    Return the 4×4 FSimGate(θ, φ) unitary matrix as a numpy array.

    FSimGate(θ, φ) =
        [[1,          0,             0,        0      ],
         [0,    cos(θ),    -i·sin(θ),          0      ],
         [0,   -i·sin(θ),   cos(θ),            0      ],
         [0,          0,             0,   e^{-iφ}     ]]

    Used for direct circuit verification via Statevector comparison.
    """
    return np.array([
        [1, 0, 0, 0],
        [0,  np.cos(theta), -1j * np.sin(theta), 0],
        [0, -1j * np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, np.exp(-1j * phi)],
    ], dtype=complex)
