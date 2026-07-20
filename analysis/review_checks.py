"""(b) Eq. (18) verification + (c) trade-off universality across all 30 instances."""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, glob, os, time, warnings, importlib.util
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, _REPO + '/src/tetron')
import cirq, mpmath
from multiprocessing import Pool
from qiskit.quantum_info import Statevector
from tcount_tradeoff import parse, circuit_from
from pygridsynth import gridsynth_gates

# ---------- (b) Eq. 18 check ----------
cf = _os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_m14_s0_e0_pEFGH.py'
spec = importlib.util.spec_from_file_location('gc', cf)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
QO, C = mod.QUBIT_ORDER, mod.CIRCUIT
qi = {q: i+1 for i, q in enumerate(QO)}   # Google logical labels 1..12
layers = []
for moment in C:
    pairs = sorted(tuple(sorted((qi[op.qubits[0]], qi[op.qubits[1]])))
                   for op in moment.operations if len(op.qubits) == 2)
    if pairs: layers.append(tuple(pairs))
distinct = []
for L in layers:
    if L not in distinct: distinct.append(L)
print(f"two-qubit layers: {len(layers)}, distinct patterns: {len(distinct)}")
names = ['E','F','G','H']
for nm, pat in zip(names, distinct):
    print(f"  {nm} ({len(pat)} pairs): {list(pat)}")
print("  layer sequence:", ''.join(names[distinct.index(L)] for L in layers))

EQ18 = {'E': {(3,4),(1,2),(5,6)},
        'F': {(4,11),(3,9),(1,8),(2,10),(5,7),(6,12)},
        'G': {(7,8),(1,5),(2,6),(10,12)},
        'H': {(8,11),(1,4),(2,3),(9,10)}}
for nm, pat in zip(names, distinct):
    got = {tuple(sorted(p)) for p in pat}
    ok = got == EQ18[nm]
    print(f"  Eq18 {nm}: {'MATCH' if ok else 'MISMATCH'}", '' if ok else f"paper={sorted(EQ18[nm])} circuit={sorted(got)}")

# ---------- (c) universality of 1-F ~ 0.15 N eps^2 at eps=1e-3 ----------
EPS = 1e-3
def synth(a): return a, gridsynth_gates(theta=mpmath.mpf(a), epsilon=mpmath.mpf(EPS))
files = sorted(glob.glob(_os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_*.py')) + \
        sorted(glob.glob(_os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_patch_n12_*.py'))
cs, rows = [], []
for f in files:
    name = os.path.basename(f).replace('circuit_','').replace('_pEFGH.py','')
    t0 = time.time()
    steps = parse(f)
    rzs = [st for st in steps if st[0]=='DIAG' and st[1]=='rz']
    N = len(rzs)
    angles = sorted({round(st[3],12) for st in rzs})
    with Pool(8) as p: impl = dict(p.map(synth, angles))
    Ttot = sum(impl[round(st[3],12)].count('T') for st in rzs)
    sv_i = Statevector.from_instruction(circuit_from(steps)).data
    sv_c = Statevector.from_instruction(circuit_from(steps, impl)).data
    F = abs(np.vdot(sv_i, sv_c))**2
    c = (1-F)/(N*EPS**2)
    cs.append(c); rows.append((name, N, Ttot, 1-F, c))
    print(f"{name:<22} N={N:<4} T={Ttot:<6} 1-F={1-F:.2e}  c={c:.3f}  ({time.time()-t0:.0f}s)")
cs = np.array(cs)
print(f"\nc = (1-F)/(N eps^2) across all 30: mean={cs.mean():.3f} std={cs.std():.3f} "
      f"min={cs.min():.3f} max={cs.max():.3f}")
full = cs[:20]
print(f"full instances only: {full.mean():.3f} ± {full.std():.3f}")
