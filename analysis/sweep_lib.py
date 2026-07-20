"""
30-instance sweep + resource counting pass.

Per instance:
  F_pub : fidelity of corrected 12+1 MBQC data state vs Google's published
          amplitude file (rounded decimals) — 20 full instances only
  F_int : fidelity vs exact ideal statevector of the same circuit
  thm   : theorem check — one uncorrected trajectory vs tracker prediction
  counts: gate-class composition, measurements (1q/2q), classical bits,
          non-Clifford sites R, rank(A) of the record->flip map
Outputs sweep_results.csv and per-fSim bit statistics.
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))

import sys, os, glob, importlib.util, time, csv, functools
import numpy as np
sys.path.insert(0, _REPO + '/src/tetron')
import cirq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from mbqc_translated_gates import MBQCTranslatedGates as G

np.random.seed(0)
ND, ANC, D = 12, 12, 4096
REPO = _REPO

# pinned block logicals (verified previously by enumeration + corrected-run match)
SXm = 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]])
Sm  = np.diag([1, 1j]).astype(complex)
H2  = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
BLOCK_U = {'SX': SXm, 'S': Sm, 'HS': H2 @ Sm}
Xm = np.array([[0, 1], [1, 0]], complex); Zm = np.diag([1, -1]).astype(complex)

def pauli_bits(U, P):
    Q = U @ P @ U.conj().T
    for (x, z), M in {(0,0): np.eye(2), (1,0): Xm, (1,1): 1j*Xm@Zm, (0,1): Zm}.items():
        r = Q @ np.linalg.inv(M)
        if (abs(abs(r[0,0])-1) < 1e-8 and abs(r[0,1]) < 1e-8
                and abs(r[1,0]) < 1e-8 and abs(r[0,0]-r[1,1]) < 1e-8):
            return (x, z)
    raise RuntimeError('non-Pauli image')

CONJ = {k: (pauli_bits(U, Xm), pauli_bits(U, Zm)) for k, U in BLOCK_U.items()}
CONJ['S1'] = (pauli_bits(Sm, Xm), pauli_bits(Sm, Zm))

def snap(angle):
    th = (angle + np.pi) % (2*np.pi) - np.pi
    k = round(th/(np.pi/2))
    return (k % 4) if abs(th - k*(np.pi/2)) < 1e-3 else None

@functools.lru_cache(maxsize=None)
def fsim_blocks(theta_r, phi_r):
    return G._zsx_blocks_and_cx(theta_r, phi_r)

def is_sx(g): return isinstance(g, cirq.XPowGate) and np.isclose(g.exponent, 0.5)
def is_sy(g): return isinstance(g, cirq.YPowGate) and np.isclose(g.exponent, 0.5)
def is_sw(g): return (isinstance(g, cirq.PhasedXPowGate)
                      and np.isclose(g.phase_exponent, 0.25)
                      and np.isclose(g.exponent, 0.5))

def parse(circ_file):
    spec = importlib.util.spec_from_file_location('gc', circ_file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    QO, C = mod.QUBIT_ORDER, mod.CIRCUIT
    qi = {q: i for i, q in enumerate(QO)}
    steps, comp = [], dict(sx=0, sy=0, sw=0, rz_nc=0, rz_cl=0, fsim=0)
    fsim_bits_list = []
    for moment in C:
        for op in moment.operations:
            g = op.gate; qs = [qi[q] for q in op.qubits]
            if is_sx(g):
                steps.append(('MB', 'SX', qs[0], 4)); comp['sx'] += 1
            elif is_sy(g):
                steps.append(('MB', 'S', qs[0], 4))
                steps.append(('MB', 'HS', qs[0], 5)); comp['sy'] += 1
            elif is_sw(g):
                steps.append(('DIAG', 'tdg', qs[0], None))
                steps.append(('MB', 'SX', qs[0], 4))
                steps.append(('DIAG', 't', qs[0], None)); comp['sw'] += 1
            elif isinstance(g, cirq.ZPowGate):
                ang = float(np.pi*g.exponent); k = snap(ang)
                if k is None:
                    steps.append(('DIAG', 'rz', qs[0], ang)); comp['rz_nc'] += 1
                else:
                    steps.append(('CLIFF_DIAG', k, qs[0])); comp['rz_cl'] += 1
            elif isinstance(g, cirq.FSimGate):
                comp['fsim'] += 1
                th, ph = round(float(g.theta), 12), round(float(g.phi), 12)
                blocks, cx_dirs = fsim_blocks(th, ph)
                gbits = 0
                for layer in range(4):
                    for ql in (0, 1):
                        wire = qs[0] if ql == 0 else qs[1]
                        for nmn, ang in blocks[layer][ql]:
                            if nmn == 'rz':
                                k = snap(ang)
                                if k is None:
                                    steps.append(('DIAG', 'rz', wire, ang)); comp['rz_nc'] += 1
                                else:
                                    steps.append(('CLIFF_DIAG', k, wire)); comp['rz_cl'] += 1
                            elif nmn == 'sx':
                                steps.append(('MB', 'SX', wire, 4)); gbits += 4
                            elif nmn == 'x':
                                steps.append(('MB', 'SX', wire, 4))
                                steps.append(('MB', 'SX', wire, 4)); gbits += 8
                    if layer < 3:
                        ci, _ = cx_dirs[layer]
                        steps.append(('MB_CNOT', (qs[0], qs[1]) if ci == 0 else (qs[1], qs[0])))
                        gbits += 3
                fsim_bits_list.append(gbits)
            else:
                raise ValueError(f'unhandled: {g}')
    return QO, steps, comp, fsim_bits_list

def counts_of(steps):
    R = sum(1 for s in steps if s[0] == 'DIAG')
    bits = sum(s[3] for s in steps if s[0] == 'MB') + 3*sum(1 for s in steps if s[0] == 'MB_CNOT')
    m2 = 0; m1 = 0
    for s in steps:
        if s[0] == 'MB':
            m2 += {'SX': 2, 'S': 1, 'HS': 2}[s[1]]
            m1 += {'SX': 2, 'S': 3, 'HS': 3}[s[1]]
        elif s[0] == 'MB_CNOT':
            m2 += 2; m1 += 1
    return R, bits, m2, m1

def ideal_circuit(steps, flips=None):
    qc = QuantumCircuit(ND); fi = 0
    for st in steps:
        if st[0] == 'MB':
            qc.append(UnitaryGate(BLOCK_U[st[1]], label=st[1]), [st[2]])
        elif st[0] == 'MB_CNOT':
            qc.cx(*st[1])
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qc.s(st[2])
        else:
            sgn = 1 if flips is None else flips[fi]; fi += 1
            kind, q, ang = st[1], st[2], st[3]
            if kind == 'rz': qc.rz(sgn*ang, q)
            elif kind == 't':   (qc.t(q) if sgn > 0 else qc.tdg(q))
            else:               (qc.tdg(q) if sgn > 0 else qc.t(q))
    return qc

def mbqc_circuit(steps, nbits, corrections):
    qr, cr = QuantumRegister(ND+1), ClassicalRegister(nbits, 'm')
    qc = QuantumCircuit(qr, cr); idx = 0
    for st in steps:
        if st[0] == 'MB':
            fn = {'SX': G.gate_sqrt_X, 'S': G.gate_S, 'HS': G.gate_HS}[st[1]]
            fn(qc, ANC, st[2], cr, idx, apply_pauli_corrections=corrections)
            idx += st[3]
        elif st[0] == 'MB_CNOT':
            c, t = st[1]
            G.gate_CNOT(qc, c, ANC, t, cr, idx, apply_pauli_corrections=corrections)
            idx += 3
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qc.s(st[2])
        else:
            kind, q, ang = st[1], st[2], st[3]
            if kind == 'rz': qc.rz(ang, q)
            elif kind == 't': qc.t(q)
            else: qc.tdg(q)
    assert idx == nbits
    return qc

def run_mbqc(qc, seed, want_sv=False):
    sim = AerSimulator(method='statevector', seed_simulator=seed)
    qc2 = qc.copy(); qc2.save_statevector()
    res = sim.run(qc2, shots=1).result()
    sv = np.asarray(res.get_statevector())
    key = list(res.get_counts().keys())[0].replace(' ', '')
    bits = np.array([int(b) for b in key[::-1]], dtype=np.uint8)
    if want_sv:
        return sv, bits
    return sv.reshape(2, D, order='C')[0]*0, bits  # unused path

def subsys_fid(full_sv, target):
    psi = full_sv.reshape(2, D, order='C')     # [ancilla, data]
    amps = psi @ target.conj()
    return float(np.sum(np.abs(amps)**2))

def track(steps, bits):
    x = np.zeros(ND, np.uint8); z = np.zeros(ND, np.uint8)
    flips = []; ptr = 0
    for st in steps:
        if st[0] == 'MB':
            which, q, nb = st[1], st[2], st[3]
            b = bits[ptr:ptr+nb]; ptr += nb
            (xX, zX), (xZ, zZ) = CONJ[which]
            nx = (x[q] & xX) ^ (z[q] & xZ); nz = (x[q] & zX) ^ (z[q] & zZ)
            x[q], z[q] = nx, nz
            if which == 'SX':
                y = b[0]^b[3]; x12 = b[1]^b[2]; bx, bz = 1 ^ (y^x12), y
            elif which == 'S':
                bx, bz = 0, 1 ^ (b[0]^b[1]^b[2])
            else:
                bx, bz = 1 ^ (b[1]^b[2]), 1 ^ (b[0]^b[1]^b[3])
            x[q] ^= bx; z[q] ^= bz
        elif st[0] == 'MB_CNOT':
            c, t = st[1]; b = bits[ptr:ptr+3]; ptr += 3
            x[t] ^= x[c]; z[c] ^= z[t]
            x[t] ^= b[0]^b[2]; z[c] ^= b[1]
        elif st[0] == 'CLIFF_DIAG':
            k, q = st[1], st[2]
            for _ in range(k):
                (xX, zX), (xZ, zZ) = CONJ['S1']
                nx = (x[q] & xX) ^ (z[q] & xZ); nz = (x[q] & zX) ^ (z[q] & zZ)
                x[q], z[q] = nx, nz
        else:
            flips.append(-1 if x[st[2]] else 1)
    return flips, x.copy()

def f2_rank(M):
    M = (M.copy() % 2).astype(np.uint8); r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r

def rank_of_A(steps, nbits, R):
    f0, _ = track(steps, np.zeros(nbits, np.uint8))
    eps0 = np.array([f < 0 for f in f0], np.uint8)
    A = np.zeros((R, nbits), np.uint8)
    e = np.zeros(nbits, np.uint8)
    for j in range(nbits):
        e[j] = 1
        fj, _ = track(steps, e)
        A[:, j] = np.array([f < 0 for f in fj], np.uint8) ^ eps0
        e[j] = 0
    return f2_rank(A)

def load_pub_amps(path):
    tgt = np.zeros(D, complex)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            b, re_, im_ = line.split()
            tgt[int(b[::-1], 2)] = float(re_) + 1j*float(im_)
    n = np.linalg.norm(tgt)
    return tgt/n if n > 0 else tgt

# ----------------------------------------------------------------------
