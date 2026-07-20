"""Sliced seeded re-measurement. Usage:
   python3 slices.py sweep <start> <end>   -> appends to sweep_seeded.csv (+fsim_bits_seeded.txt)
   python3 slices.py c30   <start> <end>   -> appends to c30_seeded.csv
Header written when start == 0."""
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_REPO, 'src', 'tetron'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'analysis'))

import sys, os, glob, time, csv, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, _REPO + '/src/tetron')
import numpy as np, mpmath
from multiprocessing import Pool
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from pygridsynth import gridsynth_gates
from tcount_tradeoff import parse, circuit_from
lib_path = 'sweep_lib.py'
if not os.path.exists(lib_path):
    src = open('sweep_30.py').read()
    cut = src.index("full = sorted(glob.glob")
    open(lib_path, 'w').write(src[:cut])
import sweep_lib as S

MODE, A, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = sorted(glob.glob(_os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_n12_*.py')) + \
        sorted(glob.glob(_os.path.join(_REPO, 'google_supremacy_circuit_files') + '/circuit_patch_n12_*.py'))

if MODE == 'sweep':
    # reuse machinery from sweep_30 module namespace (it ran fully on import;
    # cheap side effect ~outdated csv rewrite is harmless, we use our own file)
    out = 'sweep_seeded.csv'
    fb = open('fsim_bits_seeded.txt', 'a')
    mode = 'w' if A == 0 else 'a'
    f = open(out, mode, newline='')
    w = csv.writer(f)
    if A == 0:
        w.writerow(['name','F_pub','F_int','thm','R','bits','m2','m1','sx','sy','sw','rz_nc','rz_cl','fsim'])
    for cf in files[A:B]:
        name = os.path.basename(cf).replace('circuit_','').replace('_pEFGH.py','')
        t0 = time.time()
        QO, steps, comp, fbits = S.parse(cf)
        for x in fbits: fb.write(f"{x}\n")
        R, bits, m2, m1 = S.counts_of(steps)
        sv_i = Statevector.from_instruction(S.ideal_circuit(steps)).data
        qc_c = S.mbqc_circuit(steps, bits, True)
        sv_c, _ = S.run_mbqc(qc_c, seed=1, want_sv=True)
        F_int = S.subsys_fid(sv_c, sv_i)
        amp = _os.path.join(_REPO, 'data', 'google_amplitudes',
                            f"amplitudes_n12_m14_{name.replace('n12_m14_','')}_pEFGH.txt")
        F_pub = S.subsys_fid(sv_c, S.load_pub_amps(amp)) if os.path.exists(amp) else ''
        qc_u = S.mbqc_circuit(steps, bits, False)
        sv_u, ub = S.run_mbqc(qc_u, seed=2, want_sv=True)
        p_traj = np.abs(sv_u.reshape(2, 4096, order='C')).__pow__(2).sum(axis=0)
        flips, mask = S.track(steps, ub)
        p_eff = np.abs(Statevector.from_instruction(S.ideal_circuit(steps, flips)).data)**2
        m = int(sum(int(mask[i]) << i for i in range(12)))
        thm = float(np.max(np.abs(p_traj - p_eff[np.arange(4096) ^ m])))
        w.writerow([name, F_pub, F_int, thm, R, bits, m2, m1,
                    comp['sx'], comp['sy'], comp['sw'], comp['rz_nc'], comp['rz_cl'], comp['fsim']])
        f.flush()
        print(f"{name} F_int={1-F_int:.1e} thm={thm:.1e} R={R} ({time.time()-t0:.0f}s)")
    f.close(); fb.close()

elif MODE == 'c30':
    EPS = 1e-3
    def synth(a): return a, gridsynth_gates(theta=mpmath.mpf(a), epsilon=mpmath.mpf(EPS))
    out = 'c30_seeded.csv'
    mode = 'w' if A == 0 else 'a'
    f = open(out, mode, newline='')
    w = csv.writer(f)
    if A == 0:
        w.writerow(['name','N','Ttot','one_minus_F','c'])
    for cf in files[A:B]:
        name = os.path.basename(cf).replace('circuit_','').replace('_pEFGH.py','')
        t0 = time.time()
        steps = parse(cf)
        rzs = [st for st in steps if st[0]=='DIAG' and st[1]=='rz']
        N = len(rzs)
        angles = sorted({round(st[3],12) for st in rzs})
        with Pool(8) as p: impl = dict(p.map(synth, angles))
        Ttot = sum(impl[round(st[3],12)].count('T') for st in rzs)
        sv_i = Statevector.from_instruction(circuit_from(steps)).data
        sv_c = Statevector.from_instruction(circuit_from(steps, impl)).data
        F = abs(np.vdot(sv_i, sv_c))**2
        c = (1-F)/(N*EPS**2)
        w.writerow([name, N, Ttot, 1-F, c]); f.flush()
        print(f"{name} N={N} T={Ttot} c={c:.3f} ({time.time()-t0:.0f}s)")
    f.close()
