"""
Clifford+T compilation trade-off for the n12_m14_s0_e0 benchmark.

For each synthesis precision eps: replace every non-Clifford Rz by its
Ross-Selinger gridsynth Clifford+T sequence, then measure against the
exact ideal circuit:
    1 - F_state, 1 - F_XEB, total/median T-count, synthesis H/S/X counts,
    implied measurement overhead (H,S as MB blocks; Paulis frame-free).
Emits tradeoff.csv and figures/tcount_tradeoff.pdf for the paper.
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, os, importlib.util, time, csv, functools, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, _REPO + '/src/tetron')
import cirq, mpmath
from multiprocessing import Pool
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
from mbqc_translated_gates import MBQCTranslatedGates as G
from pygridsynth import gridsynth_gates

ND, D = 12, 4096
SXm = 0.5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]])
Sm = np.diag([1, 1j]).astype(complex)
H2 = np.array([[1, 1], [1, -1]], dtype=complex)/np.sqrt(2)
Tm = np.diag([1, np.exp(1j*np.pi/4)])
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
BLOCK_U = {'SX': SXm, 'S': Sm, 'HS': H2 @ Sm}
MAT1 = {'H': H2, 'T': Tm, 'S': Sm, 'X': Xm}

def snap(angle):
    th = (angle + np.pi) % (2*np.pi) - np.pi
    k = round(th/(np.pi/2))
    return (k % 4) if abs(th - k*(np.pi/2)) < 1e-3 else None

@functools.lru_cache(maxsize=None)
def fsim_blocks(t, p): return G._zsx_blocks_and_cx(t, p)

def is_sx(g): return isinstance(g, cirq.XPowGate) and np.isclose(g.exponent, 0.5)
def is_sy(g): return isinstance(g, cirq.YPowGate) and np.isclose(g.exponent, 0.5)
def is_sw(g): return (isinstance(g, cirq.PhasedXPowGate)
                      and np.isclose(g.phase_exponent, 0.25) and np.isclose(g.exponent, 0.5))

def parse(circ_file):
    spec = importlib.util.spec_from_file_location('gc', circ_file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    QO, C = mod.QUBIT_ORDER, mod.CIRCUIT
    qi = {q: i for i, q in enumerate(QO)}
    steps = []
    for moment in C:
        for op in moment.operations:
            g = op.gate; qs = [qi[q] for q in op.qubits]
            if is_sx(g): steps.append(('MB', 'SX', qs[0], 4))
            elif is_sy(g):
                steps.append(('MB', 'S', qs[0], 4)); steps.append(('MB', 'HS', qs[0], 5))
            elif is_sw(g):
                steps.append(('DIAG', 'tdg', qs[0], None))
                steps.append(('MB', 'SX', qs[0], 4))
                steps.append(('DIAG', 't', qs[0], None))
            elif isinstance(g, cirq.ZPowGate):
                ang = float(np.pi*g.exponent); k = snap(ang)
                steps.append(('CLIFF_DIAG', k, qs[0]) if k is not None
                             else ('DIAG', 'rz', qs[0], ang))
            elif isinstance(g, cirq.FSimGate):
                th, ph = round(float(g.theta), 12), round(float(g.phi), 12)
                blocks, cx_dirs = fsim_blocks(th, ph)
                for layer in range(4):
                    for ql in (0, 1):
                        wire = qs[0] if ql == 0 else qs[1]
                        for nmn, ang in blocks[layer][ql]:
                            if nmn == 'rz':
                                k = snap(ang)
                                steps.append(('CLIFF_DIAG', k, wire) if k is not None
                                             else ('DIAG', 'rz', wire, ang))
                            elif nmn == 'sx': steps.append(('MB', 'SX', wire, 4))
                            else:
                                steps.append(('MB', 'SX', wire, 4))
                                steps.append(('MB', 'SX', wire, 4))
                    if layer < 3:
                        ci, _ = cx_dirs[layer]
                        steps.append(('MB_CNOT', (qs[0], qs[1]) if ci == 0 else (qs[1], qs[0])))
            else: raise ValueError(g)
    return steps

def circuit_from(steps, rz_impl=None):
    """rz_impl: dict angle_key -> gate string; None = exact rz."""
    qc = QuantumCircuit(ND)
    for st in steps:
        if st[0] == 'MB': qc.append(UnitaryGate(BLOCK_U[st[1]], label=st[1]), [st[2]])
        elif st[0] == 'MB_CNOT': qc.cx(*st[1])
        elif st[0] == 'CLIFF_DIAG':
            for _ in range(st[1]): qc.s(st[2])
        elif st[1] in ('t', 'tdg'):
            (qc.t if st[1] == 't' else qc.tdg)(st[2])
        else:
            q, ang = st[2], st[3]
            if rz_impl is None: qc.rz(ang, q)
            else:
                for ch in rz_impl[round(ang, 12)]:
                    if ch == 'H': qc.h(q)
                    elif ch == 'T': qc.t(q)
                    elif ch == 'S': qc.s(q)
                    elif ch == 'X': qc.x(q)
    return qc

def synth_one(args):
    ang, eps = args
    seq = gridsynth_gates(theta=mpmath.mpf(ang), epsilon=mpmath.mpf(eps))
    # validate: apply seq chars in order (first char applied first)
    U = np.eye(2, dtype=complex)
    for ch in seq:
        if ch in MAT1: U = MAT1[ch] @ U
    tgt = np.diag([np.exp(-0.5j*ang), np.exp(0.5j*ang)])
    ph = np.vdot(U.flatten(), tgt.flatten()); ph /= abs(ph)
    err = np.linalg.norm(U*ph - tgt, 2)
    return ang, seq, err

if __name__ == '__main__':
    steps = parse(_os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_m14_s0_e0_pEFGH.py')
    rz_steps = [st for st in steps if st[0] == 'DIAG' and st[1] == 'rz']
    native_T = sum(1 for st in steps if st[0] == 'DIAG' and st[1] in ('t', 'tdg'))
    angles = sorted({round(st[3], 12) for st in rz_steps})
    site_count = {}
    for st in rz_steps: site_count[round(st[3], 12)] = site_count.get(round(st[3], 12), 0) + 1
    print(f"non-Clifford rz sites: {len(rz_steps)}  distinct angles: {len(angles)}  native T (sqrtW): {native_T}")

    sv_ideal = Statevector.from_instruction(circuit_from(steps)).data
    p_ideal = np.abs(sv_ideal)**2
    xeb_self = D*np.sum(p_ideal**2) - 1
    print(f"ideal self-XEB = {xeb_self:.4f}")

    EPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rows = []
    for eps in EPS:
        t0 = time.time()
        with Pool(8) as pool:
            res = pool.map(synth_one, [(a, eps) for a in angles])
        impl = {a: seq for a, seq, _ in res}
        errs = np.array([e for _, _, e in res])
        assert errs.max() <= eps*1.000001, f"synthesis error {errs.max()} > eps"
        tc = {a: impl[a].count('T') for a in angles}
        nH = {a: impl[a].count('H') for a in angles}
        nS = {a: impl[a].count('S') for a in angles}
        nX = {a: impl[a].count('X') for a in angles}
        totT = sum(tc[round(st[3],12)] for st in rz_steps)
        totH = sum(nH[round(st[3],12)] for st in rz_steps)
        totS = sum(nS[round(st[3],12)] for st in rz_steps)
        totX = sum(nX[round(st[3],12)] for st in rz_steps)
        t1 = time.time()
        sv_c = Statevector.from_instruction(circuit_from(steps, impl)).data
        F = abs(np.vdot(sv_ideal, sv_c))**2
        p_c = np.abs(sv_c)**2
        xeb = D*np.sum(p_c*p_ideal) - 1
        Ks = np.array([tc[a] for a in angles])
        rows.append(dict(eps=eps, one_minus_F=1-F, one_minus_XEB=1-xeb/xeb_self,
                         xeb=xeb, T_total=totT, T_median=int(np.median(Ks)),
                         T_min=int(Ks.min()), T_max=int(Ks.max()),
                         H_total=totH, S_total=totS, X_total=totX,
                         max_synth_err=float(errs.max())))
        print(f"eps={eps:.0e}: K_med={int(np.median(Ks))} [{Ks.min()}-{Ks.max()}]  "
              f"T_tot={totT}  1-F={1-F:.3e}  XEB={xeb:.6f}  "
              f"(synth {t1-t0:.0f}s, sim {time.time()-t1:.0f}s)")

    with open('tradeoff.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # incoherence check: fit 1-F = c * N * eps^2
    N = len(rz_steps)
    c_fit = np.mean([r['one_minus_F']/(N*r['eps']**2) for r in rows if r['one_minus_F'] > 1e-13])
    print(f"\nincoherent-accumulation fit: 1-F ~ c*N*eps^2 with c = {c_fit:.2f}  (N={N})")
    coh = [(r['eps'], r['one_minus_F']/((N*r['eps'])**2)) for r in rows]
    print("coherent-bound ratio (should be <<1 if incoherent):",
          [f"{e:.0e}:{v:.1e}" for e, v in coh])

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eps_a = np.array([r['eps'] for r in rows])
    f_a = np.array([r['one_minus_F'] for r in rows])
    x_a = np.array([max(r['one_minus_XEB'], 1e-16) for r in rows])
    t_a = np.array([r['T_total'] for r in rows])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.8))
    ax1.loglog(eps_a, f_a, 'o-', label=r'$1-F$ (state)')
    ax1.loglog(eps_a, x_a, 's--', label=r'$1-F_{\mathrm{XEB}}/F_{\mathrm{XEB}}^{\mathrm{ideal}}$')
    ax1.loglog(eps_a, c_fit*N*eps_a**2, 'k:', label=rf'${c_fit:.1f}\,N\varepsilon^2$')
    ax1.set_xlabel(r'synthesis precision $\varepsilon$'); ax1.legend(fontsize=7)
    ax1.set_ylabel('infidelity'); ax1.invert_xaxis()
    ax2.semilogy(t_a, f_a, 'o-')
    for r in rows:
        ax2.annotate(rf"$\varepsilon=10^{{{int(np.log10(r['eps']))}}}$",
                     (r['T_total'], max(r['one_minus_F'], 1e-15)),
                     textcoords='offset points', xytext=(5, 4), fontsize=6)
    ax2.set_xlabel(f'total $T$-count (N={N} rotations + {native_T} native)')
    ax2.set_ylabel(r'$1-F$ (state)')
    fig.tight_layout()
    fig.savefig('paper-v2/figures/tcount_tradeoff.pdf')
    print("figure written")
