import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))
import sys, os, math, warnings
warnings.filterwarnings('ignore')
import numpy as np

D = 4096
dat = np.load('pf_traj.npz')
p_ideal, p_corr, P = dat['p_ideal'], dat['p_corr'], dat['P']
NT = P.shape[0]

# XEB stats for V-A paragraph 2
xeb = lambda q: float(D * np.sum(q * p_ideal) - 1)
xeb_corr = xeb(p_corr)
xeb_unc = np.array([xeb(P[i]) for i in range(NT)])
print(f"XEB corrected = {xeb_corr:.3f}; uncorrected mean±std = {xeb_unc.mean():.3f} ± {xeb_unc.std():.3f}")

# TV curves
ks = np.arange(1, NT + 1)
cum = np.cumsum(P, axis=0) / ks[:, None]
tv_unif = 0.5 * np.abs(cum - 1.0 / D).sum(axis=1)
tv_ideal = 0.5 * np.abs(cum - p_ideal[None, :]).sum(axis=1)
def gamma_pdf(z, k):
    return np.exp(k*np.log(k) + (k-1)*np.log(z) - k*z - math.lgamma(k))
zg = np.linspace(1e-6, 6, 40000)
tv_pred = np.array([0.5*np.trapezoid(np.abs(zg-1)*gamma_pdf(zg, k), zg) for k in ks])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.5, 'legend.fontsize': 7,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.7})

fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.4, 4.7))

bins = np.linspace(0, 8, 41)
ctr = 0.5*(bins[:-1]+bins[1:])
h_i, _ = np.histogram(D*p_ideal, bins=bins, density=True)
h_c, _ = np.histogram(D*p_corr, bins=bins, density=True)
h_u, _ = np.histogram(D*P[0], bins=bins, density=True)
zgrid = np.linspace(0, 8, 200)
axA.plot(zgrid, np.exp(-zgrid), color='0.6', lw=2.2, alpha=0.7,
         label=r'Porter--Thomas $e^{-z}$', zorder=1)
axA.step(bins[:-1], h_i, where='post', color='0.1', lw=1.1, label='ideal', zorder=3)
axA.plot(ctr, h_c, 'o', ms=3.0, mfc='none', mec='C0', mew=0.9, label='corrected', zorder=4)
axA.step(bins[:-1], h_u, where='post', color='C1', ls='--', lw=1.0,
         label='uncorrected (1 traj.)', zorder=2)
axA.set_yscale('log'); axA.set_ylim(2e-4, 2); axA.set_xlim(0, 8)
axA.set_xlabel(r'$z = D\,p(x)$'); axA.set_ylabel('density')
axA.legend(frameon=False)

axB.plot(ks, tv_ideal, 's', ms=3, color='C3', label=r'$\mathrm{TV}(\bar{p}_k,\ \mathrm{ideal})$')
axB.axhline(1.0/np.e, color='C3', lw=0.8, ls=':')
axB.text(1.02, 1.0/np.e * 1.06, r'$1/e$', color='C3', fontsize=7)
axB.plot(ks, tv_unif, 'o', ms=3, color='C0', label=r'$\mathrm{TV}(\bar{p}_k,\ \mathrm{uniform})$')
axB.plot(ks, tv_pred, '-', color='C0', lw=0.9, alpha=0.85)
axB.set_xscale('log'); axB.set_yscale('log')
axB.set_xlabel(r'number of averaged trajectories $k$')
axB.set_ylabel('total variation distance')
axB.set_xlim(0.9, 34); axB.set_ylim(0.055, 0.62)
axB.legend(frameon=False, loc='lower left')

fig.tight_layout(h_pad=1.4)
fig.savefig('paper-v2/figures/pauli_frame.pdf')
print("final figure written")

# ---------------- tex edits ----------------
os.chdir('paper-v2')
t = open('sections/pauli_frame.tex').read()

def span_replace(t, start_marker, end_marker, new, where):
    try:
        a = t.index(start_marker)
        b = t.index(end_marker, a) + len(end_marker)
    except ValueError:
        print(f"MISSING span ({where})"); sys.exit(1)
    return t[:a] + new + t[b:]

# P2: update trajectory count and drop the TV prose (panel b owns it now)
t = span_replace(t,
"The aggregate metrics confirm the expected collapse:",
"destroys the quantum advantage signal.",
f"""The aggregate metrics confirm the expected collapse: across thirty trajectories of the uncorrected circuit, the linear XEB against Google's ideal distribution drops from $F_{{\\mathrm{{XEB}}}} = {xeb_corr:.2f}$ (corrected) to ${xeb_unc.mean():.2f} \\pm {xeb_unc.std():.2f}$ (uncorrected, indistinguishable from uniform noise by this metric). By XEB alone, the conclusion would be that removing Pauli corrections destroys the quantum advantage signal.""",
"P2")

# P3: panel (a) + KS numbers
t = span_replace(t,
"The distribution shape analysis tells a more refined story.",
"computable classically from the recorded outcomes.",
r"""The distribution-shape analysis tells a more refined story. Figure~\ref{fig:pauli_frame}(a) overlays the probability histograms---order-free densities of the $4096$ values $D\,p(x)$, with no sorting or alignment involved---for the ideal circuit, a corrected run, and a single uncorrected trajectory: all three ride the Porter--Thomas curve $e^{-z}$, with Kolmogorov--Smirnov distances from $\mathrm{Exp}(1)$ of $0.0100$, $0.0100$, and $0.0101$, each below the $5\%$ critical value $1.36/\sqrt{D} = 0.0213$. Any one uncorrected trajectory is a Porter--Thomas random-circuit output distribution; only its relation to the \emph{ideal} distribution is altered. Section~\ref{sec:pf_theory} characterizes that relation exactly: each uncorrected trajectory realizes a specific sign-flipped member of the same random-circuit ensemble, up to a bit-flip mask, with both the sign pattern and the mask computable classically from the recorded outcomes.""",
"P3")

# P4: panel (b) TV story
t = span_replace(t,
"Bitstring-indexed averaging across many trajectories",
"XEB metric was sensitive to.",
r"""Bitstring-indexed averaging across trajectories then averages over effectively independent Porter--Thomas distributions, and Fig.~\ref{fig:pauli_frame}(b) quantifies where that leads. The total variation distance of the $k$-trajectory average $\bar{p}_k$ to the \emph{uniform} distribution falls along the parameter-free $\Gamma(k,1/k)$ prediction of Sec.~\ref{sec:pf_theory} ($0.073$ predicted and measured at $k=30$), while its distance to the \emph{ideal} distribution decays from $\approx 0.5$ only to the floor $1/e \approx 0.37$---the distance of the uniform distribution itself from a Porter--Thomas instance. Frame-blind averaging thus converges to uniform at a predicted rate and never brings the samples closer to the target than pure noise; this is the convergence that the XEB metric was sensitive to.""",
"P4")

# P5: rank-average panel dropped -> two sentences, no figure ref
t = span_replace(t,
"The quantum information in each trajectory is not lost.",
"in general it is not.",
r"""The information is nevertheless not destroyed, only misaddressed: every label-free statistic of the output---the sorted probability profile, and rank statistics generally---is identical in distribution for all members of the sign-flipped ensemble and is preserved trajectory by trajectory, while label-indexed statistics collapse. Realigning a trajectory with the ideal labels is possible, and is exactly what the frame record enables (Sec.~\ref{sec:pf_theory}).""",
"P5")

# figure environment + caption
fig_a = t.index("\\begin{figure}[t]")
fig_b = t.index("\\end{figure}", fig_a) + len("\\end{figure}")
new_fig = r"""\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\columnwidth]{pauli_frame.pdf}
    \caption{Frame-uncorrected execution of the benchmark instance.
    (a)~Order-free histograms of the rescaled output probabilities $z = D\,p(x)$ for the ideal circuit (line), a corrected run (circles), and a single uncorrected trajectory (dashed), against the Porter--Thomas density $e^{-z}$; no sorting or alignment is involved, and all three are statistically indistinguishable from Porter--Thomas (Kolmogorov--Smirnov distances $\leq 0.0101$ versus the $5\%$ critical value $0.0213$).
    (b)~Total variation distance of the bitstring-indexed average $\bar{p}_k$ of $k$ uncorrected trajectories to the uniform distribution (circles; solid line: parameter-free $\Gamma(k,1/k)$ prediction) and to the ideal distribution (squares; dotted line: the $k \to \infty$ floor $1/e$, the distance of uniform from the ideal). Averaging converges to uniform and never approaches the ideal.}
    \label{fig:pauli_frame}
\end{figure}"""
# remove the TODO comment block preceding the old figure if present
pre = t[:fig_a]
todo_key = "% TODO(team): figure redesign pending"
if todo_key in pre:
    ta = pre.rindex(todo_key)
    tb = pre.index("\n", pre.index("figures pass.", ta)) + 1
    pre = pre[:ta] + pre[tb:]
t = pre + new_fig + t[fig_b:]

# V-B gamma paragraph
t = span_replace(t,
"% TODO(item 3): revisit this paragraph",
"decorrelate precisely the labels it depends on.",
r"""The trajectory-averaging behavior of Fig.~\ref{fig:pauli_frame}(b) follows directly from this picture. Fix a bitstring $x$: across independent uncorrected trajectories the rescaled values $D\,p_i(x)$ are independent $\mathrm{Exp}(1)$ variables---each trajectory is Porter--Thomas, with labelings decorrelated by its own mask and sign pattern---so the average of $k$ trajectories is $\Gamma(k,1/k)$-distributed at every bitstring. Its total variation distance to uniform, $\tfrac{1}{2}\,\mathbb{E}\lvert z - 1\rvert$ under $z \sim \Gamma(k,1/k)$, is the prediction drawn in the figure, while the distance to the ideal saturates at the uniform-to-Porter--Thomas value $1/e$. This identifies why XEB collapses under frame omission while every rank-based statistic survives: XEB is a bitstring-indexed statistic, and the trajectory-dependent labelings decorrelate precisely the labels it depends on.""",
"V-B gamma")

for bad in ["top-left", "top-right", "bottom-left", "bottom-right", "ten trajectories",
            "Fitting a Gamma", "\\alpha \\approx k", "pauli_frame_2x2"]:
    assert bad not in t, f"stale '{bad}' remains"
open('sections/pauli_frame.tex', 'w').write(t)
print("prose rewritten; no stale panel references")

td = open('TODOS.md').read()
td = td.replace("3. [IN PROGRESS] Fig. 8 redesign — 30 seeded trajectories generated; candidates A (2-panel) and B (3-panel) delivered; KS=0.010 statistic replaces residual quadrant; awaiting layout choice",
"3. [DONE — awaiting read] Fig. 8 redesign — final: panel (a) restyled overlay (markers-on-line) + panel (b) TV distances with parameter-free Gamma prediction and 1/e floor; legend decluttered; KS numbers replace residual panel; rank-average quadrant dropped, V-A/V-B prose rewritten; XEB stats updated to 30 seeded trajectories")
open('TODOS.md', 'w').write(td)
print("tracker updated")
