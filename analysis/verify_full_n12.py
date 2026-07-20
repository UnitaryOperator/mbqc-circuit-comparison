"""
Full n=12, m=14 verification of the trajectory-decomposition theorem,
with sqrt_Y support, plus extraction of the F2-affine structure of the
outcome-record -> (sign pattern, mask) map.

Layout: 12 data qubits (index = position in Google's QUBIT_ORDER) + 1
shared ancilla (index 12). Gate dispatch mirrors the team's v2 notebook
(build_mbqc_tetron_circuit) exactly, with calibrated fSim angles and
Google's Rz kept as direct rotations.
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, os, importlib.util, time
import numpy as np
sys.path.insert(0, _REPO + '/src/tetron')
sys.path.insert(0, _REPO + '/src')
import cirq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from mbqc_translated_gates import MBQCTranslatedGates as G, fSim_matrix

np.random.seed(0)
ANC = 12
ND = 12

# ======================================================================
# 0. Pin the logical Cliffords of the gate_S and gate_HS blocks
#    (and re-confirm gate_HSH) by enumeration over the 24 1q Cliffords.
# ======================================================================
H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S2 = np.diag([1, 1j]).astype(complex)

def clifford_group():
    seen, out, frontier = {}, [], [np.eye(2, dtype=complex)]
    def key(U):
        # phase-invariant key
        idx = np.argmax(np.abs(U))
        V = U / (U.flat[idx] / abs(U.flat[idx]))
        return tuple(np.round(V, 6).flatten())
    while frontier:
        nxt = []
        for U in frontier:
            k = key(U)
            if k in seen: continue
            seen[k] = True; out.append(U)
            nxt += [H2 @ U, S2 @ U]
        frontier = nxt
    return out  # 24 elements

CLIFFORDS = clifford_group()

def run_block(block_fn, nbits, psi_in, seed):
    qr, cr = QuantumRegister(2), ClassicalRegister(nbits)
    qc = QuantumCircuit(qr, cr)
    qc.initialize(psi_in, [0])
    block_fn(qc, 1, 0, cr, 0, apply_pauli_corrections=True)
    qc.save_statevector()
    sv = np.asarray(AerSimulator(method='statevector', seed_simulator=seed)
                    .run(qc, shots=1).result().get_statevector())
    M = sv.reshape(2, 2, order='F')          # [data, ancilla]
    u_, s_, _ = np.linalg.svd(M)
    assert s_[1] < 1e-9, "data not pure after block"
    return u_[:, 0]

def pin_block(block_fn, nbits, name):
    inputs = [np.array([1, 0], complex),
              np.array([1, 1], complex)/np.sqrt(2),
              np.array([1, 1j], complex)/np.sqrt(2)]
    cands = list(range(24))
    for seed in range(1, 5):
        for psi in inputs:
            out = run_block(block_fn, nbits, psi, seed)
            cands = [i for i in cands
                     if abs(np.vdot(CLIFFORDS[i] @ psi, out)) > 1 - 1e-9]
    assert len(cands) == 1, f"{name}: {len(cands)} candidates remain"
    return CLIFFORDS[cands[0]]

U_S_BLOCK   = pin_block(G.gate_S,   4, 'S-block')
U_HS_BLOCK  = pin_block(G.gate_HS,  5, 'HS-block')
U_HSH_BLOCK = pin_block(G.gate_HSH, 4, 'HSH-block')

def name_of(U):
    named = {'I': np.eye(2), 'S': S2, 'Sdg': S2.conj().T, 'Z': S2@S2,
             'H': H2, 'HS': H2@S2, 'SH': S2@H2, 'SX': (np.eye(2)+ -1j*(np.eye(2)-np.array([[0,1],[1,0]])))/1}
    SX = 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]])
    named['SX'] = SX; named['SXdg'] = SX.conj().T
    named['ZH'] = S2@S2@H2; named['HZ'] = H2@S2@S2
    named['SdgH'] = S2.conj().T@H2; named['HSdg'] = H2@S2.conj().T
    for nm, V in named.items():
        ratio = None; ok = True
        for a in range(2):
            for b in range(2):
                if abs(V[a, b]) > 1e-8:
                    r = U[a, b]/V[a, b]
                    if ratio is None: ratio = r
                    elif abs(r - ratio) > 1e-6: ok = False
                elif abs(U[a, b]) > 1e-8: ok = False
        if ok and ratio is not None and abs(abs(ratio)-1) < 1e-6:
            return nm
    return 'other'

print(f"Pinned logicals: S-block={name_of(U_S_BLOCK)}, "
      f"HS-block={name_of(U_HS_BLOCK)}, HSH-block={name_of(U_HSH_BLOCK)}")

# ======================================================================
# 1. Parse the Google circuit into a primitive-step list
# ======================================================================
CIRC_FILE = _os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_m14_s0_e0_pEFGH.py'
spec = importlib.util.spec_from_file_location('gc', CIRC_FILE)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
QUBIT_ORDER, CIRCUIT = mod.QUBIT_ORDER, mod.CIRCUIT
QIDX = {q: i for i, q in enumerate(QUBIT_ORDER)}

def snap(angle):
    th = (angle + np.pi) % (2*np.pi) - np.pi
    k = round(th / (np.pi/2))
    return (k % 4) if abs(th - k*(np.pi/2)) < 1e-3 else None

def is_sx(g): return isinstance(g, cirq.XPowGate) and np.isclose(g.exponent, 0.5)
def is_sy(g): return isinstance(g, cirq.YPowGate) and np.isclose(g.exponent, 0.5)
def is_sw(g): return (isinstance(g, cirq.PhasedXPowGate)
                      and np.isclose(g.phase_exponent, 0.25)
                      and np.isclose(g.exponent, 0.5))

# step kinds:
#  ('MB', which, q, nbits)          which in {SX, S, HS}
#  ('MB_CNOT', (c,t))
#  ('DIAG', kind, q, angle)         kind in {rz, t, tdg}
#  ('CLIFF_DIAG', k, q)
def expand_fsim(theta, phi, d1, d2, steps):
    blocks, cx_dirs = G._zsx_blocks_and_cx(theta, phi)
    for layer in range(4):
        for q_local in (0, 1):
            wire = d1 if q_local == 0 else d2
            for nmn, ang in blocks[layer][q_local]:
                if nmn == 'rz':
                    k = snap(ang)
                    if k is None: steps.append(('DIAG', 'rz', wire, ang))
                    else:         steps.append(('CLIFF_DIAG', k, wire))
                elif nmn == 'sx':
                    steps.append(('MB', 'SX', wire, 4))
                elif nmn == 'x':
                    steps.append(('MB', 'SX', wire, 4))
                    steps.append(('MB', 'SX', wire, 4))
        if layer < 3:
            ci, _ = cx_dirs[layer]
            steps.append(('MB_CNOT', (d1, d2) if ci == 0 else (d2, d1)))

steps = []
for moment in CIRCUIT:
    for op in moment.operations:
        g = op.gate
        qs = [QIDX[q] for q in op.qubits]
        if is_sx(g):
            steps.append(('MB', 'SX', qs[0], 4))
        elif is_sy(g):
            steps.append(('MB', 'S',  qs[0], 4))
            steps.append(('MB', 'HS', qs[0], 5))
        elif is_sw(g):
            steps.append(('DIAG', 'tdg', qs[0], None))
            steps.append(('MB', 'SX', qs[0], 4))
            steps.append(('DIAG', 't', qs[0], None))
        elif isinstance(g, cirq.ZPowGate):
            ang = float(np.pi * g.exponent)
            k = snap(ang)
            if k is None: steps.append(('DIAG', 'rz', qs[0], ang))
            else:         steps.append(('CLIFF_DIAG', k, qs[0]))
        elif isinstance(g, cirq.FSimGate):
            expand_fsim(float(g.theta), float(g.phi), qs[0], qs[1], steps)
        else:
            raise ValueError(f'unhandled: {g}')

R = sum(1 for s in steps if s[0] == 'DIAG')
NBITS = sum(s[3] for s in steps if s[0] == 'MB') + 3*sum(1 for s in steps if s[0] == 'MB_CNOT')
n_mb = sum(1 for s in steps if s[0] in ('MB', 'MB_CNOT'))
print(f"steps={len(steps)}  MB blocks={n_mb}  classical bits={NBITS}  non-Clifford sites R={R}")

# ======================================================================
# 2. Builders: ideal unitary (12q) / MBQC 12+1 (corrected or not)
# ======================================================================
BLOCK_U = {'SX': 0.5*np.array([[1+1j,1-1j],[1-1j,1+1j]]),
           'S': U_S_BLOCK, 'HS': U_HS_BLOCK}
from qiskit.circuit.library import UnitaryGate

def ideal_circuit(flips=None):
    qc = QuantumCircuit(ND); fi = 0
    for st in steps:
        if st[0] == 'MB':
            qc.append(UnitaryGate(BLOCK_U[st[1]], label=st[1]), [st[2]])
        elif st[0] == 'MB_CNOT':
            qc.cx(*st[1])
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qc.s(st[2])
        elif st[0] == 'DIAG':
            sgn = 1 if flips is None else flips[fi]; fi += 1
            kind, q, ang = st[1], st[2], st[3]
            if kind == 'rz': qc.rz(sgn*ang, q)
            elif kind == 't':   (qc.t(q) if sgn > 0 else qc.tdg(q))
            else:               (qc.tdg(q) if sgn > 0 else qc.t(q))
    return qc

def mbqc_circuit(corrections):
    qr, cr = QuantumRegister(ND+1), ClassicalRegister(NBITS, 'm')
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
        elif st[0] == 'DIAG':
            kind, q, ang = st[1], st[2], st[3]
            if kind == 'rz': qc.rz(ang, q)
            elif kind == 't': qc.t(q)
            else: qc.tdg(q)
    assert idx == NBITS
    return qc

# ======================================================================
# 3. Symplectic tracker
# ======================================================================
def pauli_of(U, P):
    """Return (x,z) bits of the Pauli U P U^dag (single qubit), sign ignored."""
    Q = U @ P @ U.conj().T
    for (x, z), M in {(0,0): np.eye(2), (1,0): np.array([[0,1],[1,0]]),
                      (1,1): np.array([[0,-1j],[1j,0]]), (0,1): np.diag([1,-1])}.items():
        r = Q @ np.linalg.inv(M)
        if abs(abs(r[0,0])-1) < 1e-8 and abs(r[0,1]) < 1e-8 and abs(r[1,0]) < 1e-8 \
           and abs(r[0,0]-r[1,1]) < 1e-8:
            return (x, z)
    raise RuntimeError("not a Pauli image")

Xm = np.array([[0,1],[1,0]], complex); Zm = np.diag([1,-1]).astype(complex)
CONJ = {}   # which -> ((xX,zX),(xZ,zZ)): images of X and Z under the block
for which, U in BLOCK_U.items():
    CONJ[which] = (pauli_of(U, Xm), pauli_of(U, Zm))
CONJ['S1'] = (pauli_of(S2, Xm), pauli_of(S2, Zm))   # single S for CLIFF_DIAG

def track(bits):
    """bits[i] = c[i]. Returns (flips list over DIAG steps, mask over 12 qubits)."""
    x = np.zeros(ND, dtype=np.uint8); z = np.zeros(ND, dtype=np.uint8)
    flips = []; ptr = 0
    for st in steps:
        if st[0] == 'MB':
            which, q, nb = st[1], st[2], st[3]
            b = bits[ptr:ptr+nb]; ptr += nb
            (xX, zX), (xZ, zZ) = CONJ[which]
            nx = (x[q] & xX) ^ (z[q] & xZ)
            nz = (x[q] & zX) ^ (z[q] & zZ)
            x[q], z[q] = nx, nz
            if which == 'SX':
                y = b[0]^b[3]; x12 = b[1]^b[2]
                bx, bz = 1 ^ (y ^ x12), y
            elif which == 'S':
                bx, bz = 0, 1 ^ (b[0]^b[1]^b[2])
            elif which == 'HS':
                bx, bz = 1 ^ (b[1]^b[2]), 1 ^ (b[0]^b[1]^b[3])
            x[q] ^= bx; z[q] ^= bz
        elif st[0] == 'MB_CNOT':
            c, t = st[1]
            b = bits[ptr:ptr+3]; ptr += 3
            # CNOT conjugation: X_c -> X_c X_t ; Z_t -> Z_c Z_t
            x[t] ^= x[c]; z[c] ^= z[t]
            x[t] ^= b[0]^b[2]      # X on target
            z[c] ^= b[1]           # Z on control
        elif st[0] == 'CLIFF_DIAG':
            k, q = st[1], st[2]
            for _ in range(k):
                (xX, zX), (xZ, zZ) = CONJ['S1']
                nx = (x[q] & xX) ^ (z[q] & xZ)
                nz = (x[q] & zX) ^ (z[q] & zZ)
                x[q], z[q] = nx, nz
        elif st[0] == 'DIAG':
            flips.append(-1 if x[st[2]] else 1)
    return flips, x.copy()

# ======================================================================
# 4. Validation runs
# ======================================================================
t0 = time.time()
p_ideal = np.abs(Statevector.from_instruction(ideal_circuit()).data)**2
print(f"ideal statevector done ({time.time()-t0:.1f}s), D={len(p_ideal)}, "
      f"PT check mean(D*p)={np.mean(p_ideal)*len(p_ideal):.4f}")

def run_mbqc(qc, seed):
    sim = AerSimulator(method='statevector', seed_simulator=seed)
    qc2 = qc.copy(); qc2.save_statevector()
    res = sim.run(qc2, shots=1).result()
    sv = np.asarray(res.get_statevector())
    key = list(res.get_counts().keys())[0].replace(' ', '')
    bits = np.array([int(b) for b in key[::-1]], dtype=np.uint8)
    p_full = np.abs(sv)**2
    p = p_full.reshape(2, 2**ND, order='C').sum(axis=0)  # ancilla = msb
    return p, bits

t0 = time.time()
qc_corr = mbqc_circuit(True)
p_corr, _ = run_mbqc(qc_corr, seed=1)
print(f"corrected 12+1 vs ideal: max|dp|={np.max(np.abs(p_corr-p_ideal)):.2e} "
      f"({time.time()-t0:.1f}s)")

qc_unc = mbqc_circuit(False)
print("\nuncorrected trajectories vs tracker prediction:")
worst = 0.0
for seed in range(2, 8):
    t0 = time.time()
    p_traj, bits = run_mbqc(qc_unc, seed)
    flips, mask = track(bits)
    p_eff = np.abs(Statevector.from_instruction(ideal_circuit(flips)).data)**2
    m = int(sum(int(mask[i]) << i for i in range(ND)))
    perm = np.arange(2**ND) ^ m
    err = np.max(np.abs(p_traj - p_eff[perm]))
    worst = max(worst, err)
    ms_ideal = np.allclose(np.sort(p_traj), np.sort(p_ideal), atol=1e-12)
    print(f"  seed={seed}: max err={err:.2e}  #flips={sum(f<0 for f in flips)}/{R} "
        f" mask={m:4d}  multiset==ideal? {ms_ideal}  ({time.time()-t0:.1f}s)")
print(f"\nWORST-CASE ERROR: {worst:.3e} -> "
      f"{'THEOREM VERIFIED at n=12' if worst < 1e-9 else 'MISMATCH'}")

# ======================================================================
# 5. F2-affine structure of s -> (epsilon, m): extract A, c and rank(A)
# ======================================================================
print("\nExtracting F2-affine structure of the record -> flip-pattern map...")
t0 = time.time()
f0, m0 = track(np.zeros(NBITS, dtype=np.uint8))
eps0 = np.array([1 if f < 0 else 0 for f in f0], dtype=np.uint8)
mm0 = m0.astype(np.uint8)
A = np.zeros((R, NBITS), dtype=np.uint8)
B = np.zeros((ND, NBITS), dtype=np.uint8)
e = np.zeros(NBITS, dtype=np.uint8)
for j in range(NBITS):
    e[j] = 1
    fj, mj = track(e)
    A[:, j] = np.array([1 if f < 0 else 0 for f in fj], dtype=np.uint8) ^ eps0
    B[:, j] = mj.astype(np.uint8) ^ mm0
    e[j] = 0
print(f"  probing done ({time.time()-t0:.1f}s)")

def f2_rank(M):
    M = M.copy() % 2; r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]: piv = i; break
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
        if r == rows: break
    return r

rankA = f2_rank(A)
rankAB = f2_rank(np.vstack([A, B]))
# verify affinity on 5 random records
ok = True
for seed in range(2, 7):
    _, bits = run_mbqc(qc_unc, seed) if False else (None, np.random.randint(0, 2, NBITS).astype(np.uint8))
    fp, mp = track(bits)
    eps_p = np.array([1 if f < 0 else 0 for f in fp], dtype=np.uint8)
    pred = (A @ bits + eps0) % 2
    if not np.array_equal(pred, eps_p): ok = False
print(f"  affine map verified on random records: {ok}")
print(f"  R = {R}, rank(A) = {rankA}  -> flip-pattern ensemble size = 2^{rankA}")
print(f"  rank([A;B]) (flips+mask jointly) = {rankAB}")
