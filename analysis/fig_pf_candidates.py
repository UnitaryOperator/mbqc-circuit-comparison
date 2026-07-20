"""Item 3: Fig. 8 redesign candidates.
Generates ideal + corrected + 30 uncorrected trajectories for s0_e0 (seeded
pipeline), then renders:
  fig_pf_A.pdf  - 2 panels: (a) single-trajectory PT overlay, (b) Gamma(k)
                  convergence with exact predictions + alpha-hat inset
  fig_pf_B.pdf  - A plus (c) minimal rank- vs bitstring-average comparison
Also saves trajectories to pf_traj.npz and prints KS statistics.
"""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, os, time, math, warnings
warnings.filterwarnings('ignore')
import numpy as np
from qiskit.quantum_info import Statevector

# functions-only library derived from sweep_30
if not os.path.exists('sweep_lib.py'):
    src = open('sweep_30.py').read()
    open('sweep_lib.py', 'w').write(src[:src.index('full = sorted(glob.glob')])
import sweep_lib as S

D, ND = 4096, 12
CF = _os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_m14_s0_e0_pEFGH.py'

t0 = time.time()
QO, steps, comp, _ = S.parse(CF)
R, bits, _, _ = S.counts_of(steps)
p_ideal = np.abs(Statevector.from_instruction(S.ideal_circuit(steps)).data) ** 2
qc_c = S.mbqc_circuit(steps, bits, True)
sv_c, _ = S.run_mbqc(qc_c, seed=1, want_sv=True)
p_corr = np.abs(sv_c.reshape(2, D, order='C')).__pow__(2).sum(axis=0)
qc_u = S.mbqc_circuit(steps, bits, False)
NT = 30
P = np.zeros((NT, D))
for i in range(NT):
    sv_u, _ = S.run_mbqc(qc_u, seed=2 + i, want_sv=True)
    P[i] = np.abs(sv_u.reshape(2, D, order='C')).__pow__(2).sum(axis=0)
np.savez('pf_traj.npz', p_ideal=p_ideal, p_corr=p_corr, P=P)
print(f"data: {NT} uncorrected trajectories ({time.time()-t0:.0f}s)")

def ks_vs_exp(z):
    z = np.sort(z)
    ecdf = np.arange(1, len(z) + 1) / len(z)
    return float(np.max(np.abs(ecdf - (1 - np.exp(-z)))))

print(f"KS vs Exp(1): ideal {ks_vs_exp(D*p_ideal):.4f}  corrected {ks_vs_exp(D*p_corr):.4f}  "
      f"uncorrected[0] {ks_vs_exp(D*P[0]):.4f}   (1.36/sqrt(D) = {1.36/np.sqrt(D):.4f})")

def gamma_pdf(z, k):
    return (k**k) * z**(k-1) * np.exp(-k*z) / math.gamma(k)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'legend.fontsize': 6.8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.7})

KS_LIST = [1, 2, 5, 10, 30]
CMAP = plt.cm.viridis(np.linspace(0.05, 0.85, len(KS_LIST)))

def panel_a(ax):
    zgrid = np.linspace(0, 8, 200)
    bins = np.linspace(0, 8, 41)
    for z, lab, c, ls in [(D*p_ideal, 'ideal', '0.15', '-'),
                          (D*p_corr, 'corrected', 'C0', '--'),
                          (D*P[0], 'uncorrected (1 traj.)', 'C1', ':')]:
        h, e = np.histogram(z, bins=bins, density=True)
        ax.step(e[:-1], h, where='post', color=c, ls=ls, lw=1.1, label=lab)
    ax.plot(zgrid, np.exp(-zgrid), 'r-', lw=0.9, alpha=0.8, label=r'Porter--Thomas $e^{-z}$')
    ax.set_yscale('log'); ax.set_ylim(2e-4, 2)
    ax.set_xlabel(r'$z = D\,p(x)$'); ax.set_ylabel('density')
    ax.legend(frameon=False)

def panel_b(ax):
    zgrid = np.linspace(1e-3, 4, 300)
    bins = np.linspace(0, 4, 45)
    for k, c in zip(KS_LIST, CMAP):
        pav = P[:k].mean(axis=0)
        h, e = np.histogram(D*pav, bins=bins, density=True)
        ax.step(e[:-1], h, where='post', color=c, lw=1.0)
        ax.plot(zgrid, gamma_pdf(zgrid, k), color=c, lw=0.8, alpha=0.55)
        ax.plot([], [], color=c, lw=1.2, label=rf'$k={k}$')
    ax.set_xlim(0, 4); ax.set_ylim(0, 2.6)
    ax.set_xlabel(r'$z = D\,\bar{p}_k(x)$'); ax.set_ylabel('density')
    ax.legend(frameon=False, title=None, ncol=2, columnspacing=0.9)
    # inset: alpha-hat (=1/Var) vs k
    axi = ax.inset_axes([0.60, 0.42, 0.36, 0.5])
    ah = [1.0/np.var(D*P[:k].mean(axis=0)) for k in KS_LIST]
    axi.plot([0.8, 40], [0.8, 40], '0.6', lw=0.7)
    axi.plot(KS_LIST, ah, 'o', ms=3, color='C0')
    axi.set_xscale('log'); axi.set_yscale('log')
    axi.set_xlabel(r'$k$', fontsize=6.5, labelpad=0.5)
    axi.set_ylabel(r'$\hat{\alpha}$', fontsize=6.5, labelpad=0.5)
    axi.tick_params(labelsize=5.8, pad=1.5)

def panel_c(ax):
    ranks = np.arange(1, D + 1)
    ideal_sorted = np.sort(p_ideal)[::-1]
    rank_avg = np.sort(P, axis=1)[:, ::-1].mean(axis=0)
    bit_avg_sorted = np.sort(P.mean(axis=0))[::-1]
    ax.semilogy(ranks, D*ideal_sorted, color='0.15', lw=1.2, label='ideal (sorted)')
    ax.semilogy(ranks, D*rank_avg, color='C2', lw=1.0, ls='--',
                label=rf'rank-averaged ({NT} traj.)')
    ax.semilogy(ranks, D*bit_avg_sorted, color='C3', lw=1.0, ls=':',
                label=rf'bitstring-averaged ({NT} traj.)')
    ax.axhline(1.0, color='0.7', lw=0.6)
    ax.set_xlim(0, D); ax.set_ylim(3e-3, 12)
    ax.set_xlabel('bitstring rank'); ax.set_ylabel(r'$D\,p$ (sorted)')
    ax.legend(frameon=False)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 4.7))
panel_a(a1); panel_b(a2)
fig.tight_layout(h_pad=1.3)
fig.savefig('fig_pf_A.pdf')

fig, (b1, b2, b3) = plt.subplots(3, 1, figsize=(3.4, 6.9))
panel_a(b1); panel_b(b2); panel_c(b3)
fig.tight_layout(h_pad=1.3)
fig.savefig('fig_pf_B.pdf')
print("candidates written: fig_pf_A.pdf (2-panel), fig_pf_B.pdf (3-panel)")
