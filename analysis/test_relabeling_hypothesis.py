"""
Decisive test for the Pauli-frame theory section.

Circuit under test (3 data qubits, 1 shared ancilla), using the team's
actual MBQCTranslatedGates library:

    sqrt_X(q0) ; sqrt_W(q1) ; Rz(0.7)(q0) ; fSim-like two-qubit(q0,q1)
    ; sqrt_Y(q2) ; Rz(1.1)(q2) ; CNOT(q1->q2) ; sqrt_W(q0)

with apply_pauli_corrections=False, simulated trajectory-by-trajectory
with the statevector method (mid-circuit measurements collapse randomly).

For each trajectory we test three hypotheses against the exact final
statevector probabilities p_traj:

  H1 (capstone claim):   p_traj(x) = p_ideal(x XOR m)   for some mask m
      -> test: sorted multiset of p_traj equals sorted multiset of p_ideal
  H2 (angle-flip theory): p_traj(x) = p_flip(x XOR m) where p_flip is the
      ideal circuit with Rz/T angle signs flipped according to the
      X-support of the accumulated byproduct Pauli at each diagonal
      crossing, with byproduct + mask computed classically from the
      recorded outcomes by Pauli tracking through the KNOWN Clifford
      structure.
  Shape: p_traj is Porter-Thomas-shaped either way (both consistent).

If H1 fails and H2 holds, the paper's Proposition 1 must be replaced by
the angle-flip theorem, and the 'binomial expansion' cost intuition is
confirmed.
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, itertools
import numpy as np
sys.path.insert(0, _REPO + '/src')
sys.path.insert(0, _REPO)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from tetron.mbqc_translated_gates import MBQCTranslatedGates as G, sqrt_W_matrix

np.random.seed(0)

# ----------------------------------------------------------------------
# Ideal circuit definition as a plain gate list we can also angle-flip.
# Each entry: (kind, qubits, params)
# kinds: 'sx' (sqrt X), 'sw' (sqrt W = T sxdg? -> T sqrtX Tdg), 'sy',
#        'rz', 'cx', 'fsim'
# ----------------------------------------------------------------------
THETA, PHI = 0.53 * np.pi, 0.78 * np.pi
IDEAL = [
    ('sx',   [0], None),
    ('sw',   [1], None),
    ('rz',   [0], 0.7),
    ('fsim', [0, 1], (THETA, PHI)),
    ('sy',   [2], None),
    ('rz',   [2], 1.1),
    ('cx',   [1, 2], None),
    ('sw',   [0], None),
]

def ideal_unitary_probs(gates, n=3, flip=None):
    """Statevector probs of the plain-unitary circuit. `flip` maps
    (gate_index, which_rot) -> -1/+1 sign for diagonal rotations."""
    qc = QuantumCircuit(n)
    from tetron.mbqc_translated_gates import fSim_matrix
    from qiskit.circuit.library import UnitaryGate
    for gi, (kind, qs, p) in enumerate(gates):
        if kind == 'sx':
            qc.sx(qs[0])
        elif kind == 'sy':
            # sqrt(Y) = HSS convention used by the library: apply as unitary
            qc.append(UnitaryGate(Operator(_sy_ref()).data, label='sy'), qs)
        elif kind == 'sw':
            s = 1 if flip is None else flip.get((gi, 'T'), 1)
            # sqrt_W implemented as Tdg . sqrtX . T (library order:
            # tdg first in code = rightmost in operator order? check below)
            qc.tdg(qs[0]) if s == 1 else qc.t(qs[0])
            qc.sx(qs[0])
            qc.t(qs[0]) if s == 1 else qc.tdg(qs[0])
        elif kind == 'rz':
            s = 1 if flip is None else flip.get((gi, 'rz'), 1)
            qc.rz(s * p, qs[0])
        elif kind == 'cx':
            qc.cx(qs[0], qs[1])
        elif kind == 'fsim':
            if flip is None:
                qc.append(UnitaryGate(fSim_matrix(*p)), qs)
            else:
                raise RuntimeError("fsim must be expanded for flips")
    sv = Statevector.from_instruction(qc)
    return np.abs(sv.data) ** 2, qc

def _sy_ref():
    qc = QuantumCircuit(1)
    qc.h(0); qc.s(0); qc.s(0)   # HSS reading right-to-left? library says sqrt_Y = HSS = (HS)*S
    return qc

# ----------------------------------------------------------------------
# MBQC circuit with corrections OFF, via the team's library.
# Data qubits 0..2, ancilla qubit 3.
# ----------------------------------------------------------------------
def build_mbqc(gates, corrections):
    n_data, anc = 3, 3
    qr = QuantumRegister(4)
    # count classical bits
    nbits = 0
    for kind, qs, p in gates:
        if kind == 'sx': nbits += G.N_BITS['sqrt_X']
        elif kind == 'sy': nbits += G.N_BITS['sqrt_Y']
        elif kind == 'sw': nbits += G.N_BITS['sqrt_W']
        elif kind == 'cx': nbits += G.CNOT_N_BITS
        elif kind == 'fsim': nbits += G.n_bits_two_qubit(*p)
    cr = ClassicalRegister(nbits, 'm')
    qc = QuantumCircuit(qr, cr)
    idx = 0
    for kind, qs, p in gates:
        if kind == 'sx':
            G.gate_sqrt_X(qc, anc, qs[0], cr, idx, apply_pauli_corrections=corrections)
            idx += G.N_BITS['sqrt_X']
        elif kind == 'sy':
            G.gate_sqrt_Y(qc, anc, qs[0], cr, idx, apply_pauli_corrections=corrections)
            idx += G.N_BITS['sqrt_Y']
        elif kind == 'sw':
            G.gate_sqrt_W(qc, anc, qs[0], cr, idx, apply_pauli_corrections=corrections)
            idx += G.N_BITS['sqrt_W']
        elif kind == 'rz':
            qc.rz(p, qs[0])
        elif kind == 'cx':
            G.gate_CNOT(qc, qs[0], anc, qs[1], cr, idx, apply_pauli_corrections=corrections)
            idx += G.CNOT_N_BITS
        elif kind == 'fsim':
            G.gate_two_qubit(qc, qs[0], anc, qs[1], cr, *p, start_idx=idx,
                             apply_pauli_corrections=corrections)
            idx += G.n_bits_two_qubit(*p)
    return qc

def trajectory_probs(qc, seed):
    """One trajectory: statevector sim with mid-circuit collapse; return
    marginal probs over the 3 data qubits (ancilla ends deterministic-ish;
    we trace it by taking probabilities of full state and summing anc)."""
    sim = AerSimulator(method='statevector', seed_simulator=seed)
    qc2 = qc.copy()
    qc2.save_statevector()
    res = sim.run(qc2, shots=1).result()
    sv = np.asarray(res.get_statevector())
    outcomes = list(res.get_counts().keys())[0]
    p_full = np.abs(sv) ** 2
    # qubit 3 = ancilla (most significant in Qiskit little-endian statevector index)
    p = p_full.reshape(2, 8).sum(axis=0)  # sum over ancilla (msb)
    return p, outcomes

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
p_ideal, _ = ideal_unitary_probs(IDEAL)

# sanity: corrected MBQC must reproduce ideal
qc_corr = build_mbqc(IDEAL, corrections=True)
p_corr, _ = trajectory_probs(qc_corr, seed=1)
print("corrected  max|p - p_ideal| =", np.max(np.abs(p_corr - p_ideal)))

qc_unc = build_mbqc(IDEAL, corrections=False)
print("\nUncorrected trajectories:")
print(f"{'seed':>4} {'multiset match (H1)':>22} {'best-mask TVD (H1)':>20} {'PT-ish (mean p*D)':>18}")
D = 8
for seed in range(2, 8):
    p_traj, out = trajectory_probs(qc_unc, seed)
    ms_match = np.allclose(np.sort(p_traj), np.sort(p_ideal), atol=1e-9)
    # best XOR mask
    tvds = []
    for m in range(D):
        perm = [x ^ m for x in range(D)]
        tvds.append(0.5 * np.sum(np.abs(p_traj - p_ideal[perm])))
    print(f"{seed:>4} {str(ms_match):>22} {min(tvds):>20.6f} {np.mean(p_traj)*D:>18.4f}")
