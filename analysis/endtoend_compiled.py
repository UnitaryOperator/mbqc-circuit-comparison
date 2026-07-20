"""
End-to-end verification on the Clifford+T-compiled circuit (eps=1e-2):
every non-Clifford is now a T gate. Pin gate_H's logical+byproduct, build
the compiled 12+1 MBQC circuit uncorrected, and check the trajectory
against the tracker prediction (R = total T-count). Also regenerate the
trade-off figure with |XEB deviation| handled correctly.
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, time, warnings, functools, csv
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, _REPO + '/src/tetron')
import mpmath
from multiprocessing import Pool
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from mbqc_translated_gates import MBQCTranslatedGates as G
from pygridsynth import gridsynth_gates
from tcount_tradeoff import parse, MAT1, BLOCK_U, snap  # reuse

ND, ANC, D = 12, 12, 4096
H2 = MAT1['H']; Sm = MAT1['S']; Tm = MAT1['T']; Xm = MAT1['X']
Zm = np.diag([1, -1]).astype(complex)

# ---- pin gate_H by enumeration over the 24 single-qubit Cliffords ----
def clifford_group():
    seen, out, frontier = {}, [], [np.eye(2, dtype=complex)]
    def key(U):
        idx = np.argmax(np.abs(U)); V = U/(U.flat[idx]/abs(U.flat[idx]))
        return tuple(np.round(V, 6).flatten())
    while frontier:
        nxt = []
        for U in frontier:
            k = key(U)
            if k in seen: continue
            seen[k] = True; out.append(U); nxt += [H2 @ U, Sm @ U]
        frontier = nxt
    return out

def run_block(psi_in, seed):
    qr, cr = QuantumRegister(2), ClassicalRegister(4)
    qc = QuantumCircuit(qr, cr)
    qc.initialize(psi_in, [0])
    G.gate_H(qc, 1, 0, cr, 0, apply_pauli_corrections=True)
    qc.save_statevector()
    sv = np.asarray(AerSimulator(method='statevector', seed_simulator=seed)
                    .run(qc, shots=1).result().get_statevector())
    M = sv.reshape(2, 2, order='F'); u_, s_, _ = np.linalg.svd(M)
    assert s_[1] < 1e-9
    return u_[:, 0]

CL = clifford_group()
cands = list(range(24))
for seed in range(1, 5):
    for psi in [np.array([1, 0], complex), np.array([1, 1], complex)/np.sqrt(2),
                np.array([1, 1j], complex)/np.sqrt(2)]:
        out = run_block(psi, seed)
        cands = [i for i in cands if abs(np.vdot(CL[i] @ psi, out)) > 1 - 1e-9]
assert len(cands) == 1
UH = CL[cands[0]]
isH = np.allclose(UH/(UH[0, 0]/abs(UH[0, 0])), H2) or np.allclose(UH @ H2.conj().T, np.eye(2)*(UH @ H2.conj().T)[0, 0])
print(f"gate_H logical pinned: {'H' if isH else 'NOT H?!'}")

# ---- compile at eps=1e-2 and build compiled step list ----
EPS = 1e-2
steps0 = parse(_os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_m14_s0_e0_pEFGH.py')
angles = sorted({round(st[3], 12) for st in steps0 if st[0] == 'DIAG' and st[1] == 'rz'})
def synth(a): return a, gridsynth_gates(theta=mpmath.mpf(a), epsilon=mpmath.mpf(EPS))
with Pool(8) as p: impl = dict(p.map(synth, angles))

csteps = []
for st in steps0:
    if st[0] == 'DIAG' and st[1] == 'rz':
        q = st[2]
        for ch in impl[round(st[3], 12)]:
            if ch == 'H': csteps.append(('MB', 'H', q, 4))
            elif ch == 'S': csteps.append(('MB', 'S', q, 4))
            elif ch == 'T': csteps.append(('DIAG', 't', q, None))
            elif ch == 'X': csteps.append(('PAULI_X', q))
    else:
        csteps.append(st)
R_c = sum(1 for s in csteps if s[0] == 'DIAG')
nbits = sum(s[3] for s in csteps if s[0] == 'MB') + 3*sum(1 for s in csteps if s[0] == 'MB_CNOT')
nH = sum(1 for s in csteps if s[0] == 'MB' and s[1] == 'H')
print(f"compiled: steps={len(csteps)}  MB-H blocks={nH}  bits={nbits}  R=T-count={R_c}")

# ---- builders ----
BLOCK_U2 = dict(BLOCK_U); BLOCK_U2['H'] = H2

def compiled_circuit(flips=None):
    qc = QuantumCircuit(ND); fi = 0
    for st in csteps:
        if st[0] == 'MB': qc.append(UnitaryGate(BLOCK_U2[st[1]], label=st[1]), [st[2]])
        elif st[0] == 'MB_CNOT': qc.cx(*st[1])
        elif st[0] == 'PAULI_X': qc.x(st[1])
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qc.s(st[2])
        else:
            sgn = 1 if flips is None else flips[fi]; fi += 1
            (qc.t if (st[1] == 't') == (sgn > 0) else qc.tdg)(st[2])
    return qc

def mbqc_compiled(corrections):
    qr, cr = QuantumRegister(ND+1), ClassicalRegister(nbits, 'm')
    qc = QuantumCircuit(qr, cr); idx = 0
    for st in csteps:
        if st[0] == 'MB':
            fn = {'SX': G.gate_sqrt_X, 'S': G.gate_S, 'HS': G.gate_HS, 'H': G.gate_H}[st[1]]
            fn(qc, ANC, st[2], cr, idx, apply_pauli_corrections=corrections)
            idx += st[3]
        elif st[0] == 'MB_CNOT':
            c, t = st[1]
            G.gate_CNOT(qc, c, ANC, t, cr, idx, apply_pauli_corrections=corrections); idx += 3
        elif st[0] == 'PAULI_X': qc.x(st[1])
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qc.s(st[2])
        else:
            (qc.t if st[1] == 't' else qc.tdg)(st[2])
    assert idx == nbits
    return qc

def pauli_bits(U, P):
    Q = U @ P @ U.conj().T
    for (x, z), M in {(0,0): np.eye(2), (1,0): Xm, (1,1): 1j*Xm@Zm, (0,1): Zm}.items():
        r = Q @ np.linalg.inv(M)
        if (abs(abs(r[0,0])-1) < 1e-8 and abs(r[0,1]) < 1e-8 and abs(r[1,0]) < 1e-8
                and abs(r[0,0]-r[1,1]) < 1e-8):
            return (x, z)
    raise RuntimeError

CONJ = {k: (pauli_bits(U, Xm), pauli_bits(U, Zm)) for k, U in BLOCK_U2.items()}
CONJ['S1'] = (pauli_bits(Sm, Xm), pauli_bits(Sm, Zm))

def track(bits):
    x = np.zeros(ND, np.uint8); z = np.zeros(ND, np.uint8)
    flips = []; ptr = 0
    for st in csteps:
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
            elif which == 'HS':
                bx, bz = 1 ^ (b[1]^b[2]), 1 ^ (b[0]^b[1]^b[3])
            else:  # H: X if parity else Z
                par = b[0]^b[1]^b[2]; bx, bz = par, 1 ^ par
            x[q] ^= bx; z[q] ^= bz
        elif st[0] == 'MB_CNOT':
            c, t = st[1]; b = bits[ptr:ptr+3]; ptr += 3
            x[t] ^= x[c]; z[c] ^= z[t]
            x[t] ^= b[0]^b[2]; z[c] ^= b[1]
        elif st[0] == 'PAULI_X':
            pass
        elif st[0] == 'CLIFF_DIAG':
            k, q = st[1], st[2]
            for _ in range(k):
                (xX, zX), (xZ, zZ) = CONJ['S1']
                nx = (x[q] & xX) ^ (z[q] & xZ); nz = (x[q] & zX) ^ (z[q] & zZ)
                x[q], z[q] = nx, nz
        else:
            flips.append(-1 if x[st[2]] else 1)
    return flips, x

# ---- run: uncorrected trajectories vs tracker (theorem with R = T-count) --
worst = 0
qc_u = mbqc_compiled(False)
for seed in (2, 3):
    t0 = time.time()
    sim = AerSimulator(method='statevector', seed_simulator=seed)
    qq = qc_u.copy(); qq.save_statevector()
    res = sim.run(qq, shots=1).result()
    sv = np.asarray(res.get_statevector())
    key = list(res.get_counts().keys())[0].replace(' ', '')
    bits = np.array([int(b) for b in key[::-1]], dtype=np.uint8)
    p_traj = np.abs(sv.reshape(2, D, order='C')).__pow__(2).sum(axis=0)
    t1 = time.time()
    flips, mask = track(bits)
    p_eff = np.abs(Statevector.from_instruction(compiled_circuit(flips)).data)**2
    m = int(sum(int(mask[i]) << i for i in range(ND)))
    err = np.max(np.abs(p_traj - p_eff[np.arange(D) ^ m]))
    worst = max(worst, err)
    print(f"seed={seed}: traj sim {t1-t0:.0f}s, track+eff {time.time()-t1:.0f}s, "
          f"#T-flips={sum(f<0 for f in flips)}/{R_c}, max err={err:.2e}")
print(f"COMPILED-CIRCUIT THEOREM: worst err {worst:.2e} with R = {R_c} T gates")

# ---- corrected compiled run (fidelity vs compiled ideal), time-guarded ----
sv_comp_ideal = Statevector.from_instruction(compiled_circuit()).data
t0 = time.time()
import signal
def _to(sig, frm): raise TimeoutError('corrected run exceeded 700s')
signal.signal(signal.SIGALRM, _to)
signal.alarm(700)
try:
    qc_c = mbqc_compiled(True)
    sim = AerSimulator(method='statevector', seed_simulator=1)
    qq = qc_c.copy(); qq.save_statevector()
    res = sim.run(qq, shots=1).result()
    sv = np.asarray(res.get_statevector())
    psi = sv.reshape(2, D, order='C')
    Fc = float(np.sum(np.abs(psi @ sv_comp_ideal.conj())**2))
    print(f"corrected compiled 12+1 vs compiled ideal: 1-F = {1-Fc:.2e} ({time.time()-t0:.0f}s)")
except Exception as ex:
    print("corrected compiled run skipped:", ex)
signal.alarm(0)

# ---- regenerate figure with |XEB deviation| ----
rows = list(csv.DictReader(open('tradeoff.csv')))
for r in rows:
    for k in r: r[k] = float(r[k])
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
xeb_self = 1.0174874  # recompute properly below
sv_ideal = Statevector.from_instruction(compiled_circuit()).data  # placeholder
# recompute self-XEB exactly from uncompiled ideal
from tcount_tradeoff import circuit_from
sv_id = Statevector.from_instruction(circuit_from(steps0)).data
p_id = np.abs(sv_id)**2
xeb_self = float(D*np.sum(p_id**2) - 1)
eps_a = np.array([r['eps'] for r in rows])
f_a = np.array([r['one_minus_F'] for r in rows])
dxeb = np.array([abs(r['xeb'] - xeb_self)/xeb_self for r in rows])
t_a = np.array([r['T_total'] for r in rows])
N = 568
c_fit = float(np.mean(f_a/(N*eps_a**2)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6))
ax1.loglog(eps_a, f_a, 'o-', label=r'$1-F$')
ax1.loglog(eps_a, dxeb, 's--', label=r'$|\Delta F_{\mathrm{XEB}}|/F_{\mathrm{XEB}}$')
ax1.loglog(eps_a, c_fit*N*eps_a**2, 'k:', lw=1, label=rf'$0.15\,N\varepsilon^2$')
ax1.set_xlabel(r'synthesis precision $\varepsilon$')
ax1.set_ylabel('deviation from ideal'); ax1.invert_xaxis(); ax1.legend(fontsize=6.5)
ax2.semilogy(t_a, f_a, 'o-')
for r in rows:
    ax2.annotate(rf"$10^{{{int(np.log10(r['eps']))}}}$", (r['T_total'], r['one_minus_F']),
                 textcoords='offset points', xytext=(4, 4), fontsize=6)
ax2.set_xlabel(r'total $T$-count (568 rotations)')
ax2.set_ylabel(r'$1-F$')
fig.tight_layout()
fig.savefig('paper-v2/figures/tcount_tradeoff.pdf')
print(f"figure regenerated (self-XEB={xeb_self:.6f}, c_fit={c_fit:.3f})")
