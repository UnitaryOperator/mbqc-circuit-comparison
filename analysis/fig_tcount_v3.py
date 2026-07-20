import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'legend.fontsize': 7,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.7, 'lines.markersize': 4.5})

rows = list(csv.DictReader(open('tradeoff.csv')))
for r in rows:
    for k in r: r[k] = float(r[k])
rows.sort(key=lambda r: r['eps'])

N = 568
XEB_SELF = 1.017487
C_MEAN, C_STD = 0.122, 0.011

eps = np.array([r['eps'] for r in rows])
f = np.array([r['one_minus_F'] for r in rows])
dxeb = np.array([abs(r['xeb'] - XEB_SELF) / XEB_SELF for r in rows])
T = np.array([r['T_total'] for r in rows])
K = np.array([int(r['T_median']) for r in rows])
c_pts = f / (N * eps**2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.4, 6.2),
                                    gridspec_kw={'height_ratios': [3, 1.6, 2.6]})

# (a) raw deviations vs eps, eps increasing rightward
grid = np.logspace(np.log10(eps.min()), np.log10(eps.max()), 100)
ax1.loglog(grid, C_MEAN * N * grid**2, 'k--', lw=0.9,
           label=rf'$c\,N\varepsilon^{{2}}$, $c={C_MEAN}$')
ax1.loglog(eps, f, 'o-', color='C0', label=r'$1-F$')
ax1.loglog(eps, dxeb, 's--', color='C1',
           label=r'$|F_{\mathrm{XEB}}-F_{\mathrm{XEB}}^{\mathrm{ideal}}|/F_{\mathrm{XEB}}^{\mathrm{ideal}}$')
ax1.set_xlabel(r'synthesis precision $\varepsilon$')
ax1.set_ylabel('deviation from ideal')
ax1.legend(loc='upper left', frameon=False)

# (b) compensated ratio with visible band
ax2.axhspan(C_MEAN - C_STD, C_MEAN + C_STD, color='0.8', alpha=0.8, lw=0)
ax2.axhline(C_MEAN, color='0.45', lw=0.8)
ax2.semilogx(eps, c_pts, 'o', color='C0')
ax2.set_xlabel(r'synthesis precision $\varepsilon$')
ax2.set_ylabel(r'$(1-F)/(N\varepsilon^{2})$')
ax2.set_ylim(0.08, 0.20)
ax2.text(1.4e-6, C_MEAN + C_STD + 0.006, r'$c = 0.122 \pm 0.011$ (30 instances)',
         fontsize=7)

# (c) infidelity vs total T-count
ax3.semilogy(T, f, 'o-', color='C0')
for Ti, fi, Ki in zip(T, f, K):
    ax3.annotate(rf'$K\,{{=}}\,{Ki}$', (Ti, fi), textcoords='offset points',
                 xytext=(6, 3), fontsize=7)
ax3.set_xlabel(r'total $T$-count (568 synthesized rotations)')
ax3.set_ylabel(r'circuit infidelity $1-F$')
ax3.set_xlim(T.min() - 2800, T.max() + 6800)
ax3.set_ylim(f.min() * 0.12, f.max() * 12)

fig.tight_layout(h_pad=1.2)
fig.savefig('paper-v2/figures/tcount_tradeoff.pdf')
print('3-panel figure written')
