"""
H2 test: every uncorrected trajectory equals p_eff(x XOR m), where the
effective circuit (Rz/T sign flips) and mask m are computed CLASSICALLY
from the recorded mid-circuit outcomes by Pauli-frame tracking.

Invariant maintained: |psi_actual> = F * E |0>, with F a Pauli (8x8 here)
and E the effective circuit built so far.
  MB Clifford block (logical U, byproduct B):  F <- B (U F U^дag), E <- U E
  Diagonal non-Clifford D (rz, t, tdg) on q:   E <- D' E where D' = D with
       flipped sign iff F anticommutes with Z_q; F unchanged
  Diagonal Clifford (S^k):                     F <- C F C^dag, E <- C E
Final: mask m from F|0> = phase * |m>; prediction p(x) = p_E(x XOR m).
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys
import numpy as np
sys.path.insert(0, _REPO + '/src')
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from tetron.mbqc_translated_gates import MBQCTranslatedGates as G

np.random.seed(0)
N = 3           # data qubits
ANC = 3         # ancilla index
D = 2**N

# ---------------- circuit under test (no sqrt_Y; fsim included) --------
THETA, PHI = 0.53*np.pi, 0.78*np.pi
IDEAL = [
    ('sx',   [0], None),
    ('sw',   [1], None),
    ('rz',   [0], 0.7),
    ('fsim', [0, 1], (THETA, PHI)),
    ('rz',   [2], 1.1),
    ('sx',   [2], None),
    ('cx',   [1, 2], None),
    ('sw',   [0], None),
]

# ---------------- expansion into primitive steps -----------------------
# step kinds: ('MB', logical_gate, qubits, nbits, byproduct_fn)
#             ('DIAG', 'rz'|'t'|'tdg', q, angle)
#             ('CLIFF_DIAG', k, q)          # S^k unitary
def hsh_byproduct(bits):            # bits = [c[i],..,c[i+3]]
    y   = bits[0] ^ bits[3]
    x12 = bits[1] ^ bits[2]
    return {'x': 1 ^ (y ^ x12), 'z': y}

def cnot_byproduct(bits):           # bits = [P1, P2, M]
    return {'xt': bits[0] ^ bits[2], 'zc': bits[1]}

def snap(angle):
    th = (angle + np.pi) % (2*np.pi) - np.pi
    k = round(th / (np.pi/2))
    return (k % 4) if abs(th - k*(np.pi/2)) < 1e-3 else None

def expand(gates):
    steps = []
    for kind, qs, p in gates:
        if kind == 'sx':
            steps.append(('MB_SX', qs[0], G.N_BITS['sqrt_X']))
        elif kind == 'sw':
            steps.append(('DIAG', 'tdg', qs[0], None))
            steps.append(('MB_SX', qs[0], G.N_BITS['sqrt_X']))
            steps.append(('DIAG', 't', qs[0], None))
        elif kind == 'rz':
            k = snap(p)
            if k is None: steps.append(('DIAG', 'rz', qs[0], p))
            else:         steps.append(('CLIFF_DIAG', k, qs[0], None))
        elif kind == 'cx':
            steps.append(('MB_CNOT', (qs[0], qs[1]), G.CNOT_N_BITS))
        elif kind == 'fsim':
            blocks, cx_dirs = G._zsx_blocks_and_cx(*p)
            c_, t_ = qs
            for layer in range(4):
                for q_local in (0, 1):
                    wire = c_ if q_local == 0 else t_
                    for name, angle in blocks[layer][q_local]:
                        if name == 'rz':
                            k = snap(angle)
                            if k is None: steps.append(('DIAG', 'rz', wire, angle))
                            else:         steps.append(('CLIFF_DIAG', k, wire, None))
                        elif name == 'sx':
                            steps.append(('MB_SX', wire, G.N_BITS['sqrt_X']))
                        elif name == 'x':
                            steps.append(('MB_SX', wire, G.N_BITS['sqrt_X']))
                            steps.append(('MB_SX', wire, G.N_BITS['sqrt_X']))
                if layer < 3:
                    ci, ti = cx_dirs[layer]
                    pair = (c_, t_) if ci == 0 else (t_, c_)
                    steps.append(('MB_CNOT', pair, G.CNOT_N_BITS))
    return steps

# ---------------- small operator helpers (n=3) --------------------------
def op1(mat, q):
    full = [np.eye(2)]*N; full[q] = mat
    out = full[N-1]
    for k in range(N-2, -1, -1): out = np.kron(out, full[k])
    return out
I2 = np.eye(2); Xm = np.array([[0,1],[1,0]],dtype=complex)
Zm = np.diag([1,-1]).astype(complex); Sm = np.diag([1,1j])
SXm = 0.5*np.array([[1+1j,1-1j],[1-1j,1+1j]])
Tm = np.diag([1, np.exp(1j*np.pi/4)])
def cxop(c, t):
    out = np.zeros((D, D), dtype=complex)
    for x in range(D):
        y = x ^ (1 << t) if (x >> c) & 1 else x
        out[y, x] = 1
    return out

# ---------------- build MBQC circuit + run one trajectory ---------------
def build_mbqc(gates):
    nbits = 0
    for kind, qs, p in gates:
        nbits += {'sx': G.N_BITS['sqrt_X'], 'sw': G.N_BITS['sqrt_W'],
                  'cx': G.CNOT_N_BITS}.get(kind, 0)
        if kind == 'fsim': nbits += G.n_bits_two_qubit(*p)
    qr, cr = QuantumRegister(N+1), ClassicalRegister(nbits, 'm')
    qc = QuantumCircuit(qr, cr); idx = 0
    for kind, qs, p in gates:
        if kind == 'sx':
            G.gate_sqrt_X(qc, ANC, qs[0], cr, idx, apply_pauli_corrections=False); idx += 4
        elif kind == 'sw':
            G.gate_sqrt_W(qc, ANC, qs[0], cr, idx, apply_pauli_corrections=False); idx += 4
        elif kind == 'rz':
            qc.rz(p, qs[0])
        elif kind == 'cx':
            G.gate_CNOT(qc, qs[0], ANC, qs[1], cr, idx, apply_pauli_corrections=False); idx += 3
        elif kind == 'fsim':
            G.gate_two_qubit(qc, qs[0], ANC, qs[1], cr, *p, start_idx=idx,
                             apply_pauli_corrections=False); idx += G.n_bits_two_qubit(*p)
    return qc, nbits

def run_traj(qc, seed):
    sim = AerSimulator(method='statevector', seed_simulator=seed)
    qc2 = qc.copy(); qc2.save_statevector()
    res = sim.run(qc2, shots=1).result()
    sv = np.asarray(res.get_statevector())
    key = list(res.get_counts().keys())[0].replace(' ', '')
    bits = [int(b) for b in key[::-1]]          # bits[i] = c[i]
    p_full = np.abs(sv)**2
    return p_full.reshape(2, D).sum(axis=0), bits

# ---------------- Pauli tracker -----------------------------------------
def predict(steps, bits):
    F = np.eye(D, dtype=complex)
    flips = []                                   # sign per DIAG step
    ptr = 0
    for st in steps:
        if st[0] == 'MB_SX':
            q, nb = st[1], st[2]
            b = bits[ptr:ptr+nb]; ptr += nb
            U = op1(SXm, q)
            F = U @ F @ U.conj().T
            bp = hsh_byproduct(b)
            B = (op1(Xm, q) if bp['x'] else np.eye(D)) @ (op1(Zm, q) if bp['z'] else np.eye(D))
            F = B @ F
        elif st[0] == 'MB_CNOT':
            (c, t), nb = st[1], st[2]
            b = bits[ptr:ptr+nb]; ptr += nb
            U = cxop(c, t)
            F = U @ F @ U.conj().T
            bp = cnot_byproduct(b)
            B = (op1(Xm, t) if bp['xt'] else np.eye(D)) @ (op1(Zm, c) if bp['zc'] else np.eye(D))
            F = B @ F
        elif st[0] == 'CLIFF_DIAG':
            k, q = st[1], st[2]
            U = op1(np.linalg.matrix_power(Sm, k), q)
            F = U @ F @ U.conj().T
        elif st[0] == 'DIAG':
            q = st[2]
            Zq = op1(Zm, q)
            anticommute = np.linalg.norm(F @ Zq - Zq @ F) > 1e-9
            flips.append(-1 if anticommute else +1)
    # mask from F|0>
    v = F @ np.eye(D)[:, 0]
    m = int(np.argmax(np.abs(v)))
    # effective circuit with flips
    qe = QuantumCircuit(N); fi = 0
    for st in steps:
        if st[0] == 'MB_SX': qe.sx(st[1])
        elif st[0] == 'MB_CNOT': qe.cx(*st[1])
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qe.s(st[2])
        elif st[0] == 'DIAG':
            s = flips[fi]; fi += 1
            kind, q, ang = st[1], st[2], st[3]
            if kind == 'rz': qe.rz(s*ang, q)
            elif kind == 't':   (qe.t(q) if s > 0 else qe.tdg(q))
            elif kind == 'tdg': (qe.tdg(q) if s > 0 else qe.t(q))
    p_eff = np.abs(Statevector.from_instruction(qe).data)**2
    perm = [x ^ m for x in range(D)]
    return p_eff[perm], m, flips

# ---------------- run ----------------------------------------------------
steps = expand(IDEAL)
n_diag = sum(1 for s in steps if s[0] == 'DIAG')
qc, nbits = build_mbqc(IDEAL)
print(f"primitive steps: {len(steps)}, non-Clifford diagonal sites: {n_diag}, classical bits: {nbits}")
print(f"{'seed':>4} {'max|p_traj - p_pred|':>22} {'mask':>5} {'#flips':>7}")
worst = 0
for seed in range(2, 12):
    p_traj, bits = run_traj(qc, seed)
    p_pred, m, flips = predict(steps, bits)
    err = np.max(np.abs(p_traj - p_pred))
    worst = max(worst, err)
    print(f"{seed:>4} {err:>22.3e} {m:>5} {sum(1 for f in flips if f<0):>7}/{len(flips)}")
print("\nWORST-CASE ERROR:", worst, "->", "H2 CONFIRMED" if worst < 1e-9 else "H2 FAILS")
