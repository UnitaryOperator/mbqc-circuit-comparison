"""
plot_google_xeb.py

Compute and plot linear XEB across Google's 2019 supremacy n=12, m=14 circuits.

The Google amplitudes file is already row-paired with the experimental shots:
each line is "bitstring  re  im" where (re, im) is the *ideal* amplitude for
that experimentally-observed bitstring. So:

    F_XEB = D * <|amp_i|^2>_i  -  1,    D = 2^n

No bitstring index lookup is required, and the measurements_*.txt file is
redundant for the XEB calculation (its bitstrings are the same ones already
listed column-0 of the amplitudes file).

Run from repo root:
    python notebooks/plot_google_xeb.py
or import the helpers from src/benchmarks/xeb.py.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# core helpers (good candidates to live in src/benchmarks/xeb.py)
# =====================================================================
def xeb_from_google_amplitudes(path, n_qubits=12):
    """Linear F_XEB and 1-sigma SE from a Google supremacy amplitudes file.

    Parameters
    ----------
    path : str or Path
        Path to amplitudes_n*_m*_s*_e*_p*.txt
    n_qubits : int
        Number of qubits in the circuit (sets D = 2^n_qubits).

    Returns
    -------
    F_XEB : float
    F_XEB_se : float    standard error of the mean (1-sigma)
    n_shots : int
    """
    data = np.loadtxt(path, usecols=(1, 2))
    p = data[:, 0] ** 2 + data[:, 1] ** 2
    D = 2 ** n_qubits
    F = D * p.mean() - 1.0
    se = D * p.std(ddof=1) / np.sqrt(len(p))
    return float(F), float(se), len(p)


def parse_amplitudes_filename(path):
    """Extract (n, m, s, e, pattern) from amplitudes_n*_m*_s*_e*_p*.txt"""
    name = Path(path).name
    m = re.match(r'amplitudes_n(\d+)_m(\d+)_s(\d+)_e(\d+)_p(\w+)\.txt', name)
    if not m:
        raise ValueError(f"Cannot parse filename: {name}")
    return dict(n=int(m[1]), m=int(m[2]), s=int(m[3]),
                e=int(m[4]), pattern=m[5])


def load_xeb_table(amp_dir):
    """Compute XEB for every amplitudes file under amp_dir; return a DataFrame."""
    amp_dir = Path(amp_dir)
    rows = []
    for path in sorted(amp_dir.glob('amplitudes_*.txt')):
        info = parse_amplitudes_filename(path)
        F, se, N = xeb_from_google_amplitudes(path, n_qubits=info['n'])
        rows.append({
            **info, 'F_XEB': F, 'F_XEB_se': se, 'n_shots': N,
            'circuit_type': 'full' if info['e'] == 0 else 'elided',
            'path': str(path),
        })
    return pd.DataFrame(rows)


# =====================================================================
# plotting
# =====================================================================
def plot_xeb_by_seed(df, save_to=None):
    """F_XEB vs seed, separate series for full (e=0) vs elided (e=6)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {'full':   dict(color='C0', marker='o', label='full (e=0)'),
              'elided': dict(color='C1', marker='s', label='elided (e=6)')}

    for ctype, sub in df.groupby('circuit_type'):
        sub = sub.sort_values('s')
        s = styles[ctype]
        ax.errorbar(sub['s'], sub['F_XEB'], yerr=sub['F_XEB_se'],
                    fmt=s['marker'] + '-', color=s['color'],
                    capsize=3, lw=1.2, ms=7, label=s['label'])
        ax.axhline(sub['F_XEB'].mean(), color=s['color'],
                   ls=':', alpha=0.6, lw=1,
                   label=f'{ctype} mean = {sub["F_XEB"].mean():.3f}')

    n, m = int(df['n'].iloc[0]), int(df['m'].iloc[0])
    ax.set_xlabel('PRNG seed (s)')
    ax.set_ylabel(r'$F_{\rm XEB}$')
    ax.set_title(f'Sycamore linear XEB  |  n={n}, m={m}, pattern EFGH')
    ax.set_xticks(range(int(df['s'].max()) + 1))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, dpi=150, bbox_inches='tight')
    return fig


# =====================================================================
# entrypoint
# =====================================================================
if __name__ == '__main__':
    # Resolve repo root so this works whether run from notebooks/ or the root.
    here = Path(__file__).resolve()
    repo_root = next((p for p in [here.parent, *here.parents]
                      if (p / 'data' / 'google_amplitudes').is_dir()), None)
    if repo_root is None:
        raise SystemExit("Could not find data/google_amplitudes/ above this script.")
    amp_dir = repo_root / 'data' / 'google_amplitudes'

    df = load_xeb_table(amp_dir)
    print(df[['n', 'm', 's', 'e', 'circuit_type',
              'F_XEB', 'F_XEB_se', 'n_shots']].to_string(index=False))
    print()
    print('Summary by circuit type:')
    print(df.groupby('circuit_type')['F_XEB']
            .agg(['mean', 'std', 'count']).round(4))

    plot_xeb_by_seed(df, save_to=repo_root / 'figures' / 'xeb_google_n12_m14.png')
    plt.show()
