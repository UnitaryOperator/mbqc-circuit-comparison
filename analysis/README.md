# Analysis pipeline

This directory contains the analysis and figure-generation scripts for the
paper *Random Circuit Sampling on Majorana Tetron Arrays: Translation,
Resource Costs, and the Pauli Frame as Classical Bookkeeping*. They build on
the translation engine in `src/tetron/` and reproduce every number, table, and
figure in the paper.

## Requirements

Python 3.10+ with the packages in `requirements.txt`:
`qiskit`, `qiskit-aer`, `cirq`, `pygridsynth`, `numpy`, `matplotlib`.

```
pip install -r requirements.txt
```

## Reproducibility

Every quoted result comes from a single **seeded** compilation of each
circuit. The stochastic step is the fSim two-qubit decomposition in
`src/tetron/mbqc_translated_gates.py`, pinned via `seed_transpiler=20260709`.
With the seed in place, all gate counts and fidelities are bit-for-bit
reproducible; without it, per-instance counts drift by ~0.3% between runs.

## What each script does

### Verification (paper's core equivalence and theorem results)

| Script | Produces | Notes |
|---|---|---|
| `sweep_30.py` | Table I, Appendix B (`sweep_seeded.csv`) | Main workhorse. Sweeps all 30 circuit files; per instance computes fidelity vs. Google's published amplitudes, fidelity vs. exact simulation, the Pauli-frame tracker check of Theorem 1, and all resource counts. |
| `slices.py` | same, in chunks | Runs the sweep in contiguous slices for environments with short execution windows. Auto-derives `sweep_lib.py` from `sweep_30.py` on first run. |
| `verify_full_n12.py` | full n=12 theorem check | Standalone machine-precision verification of Theorem 1 on the benchmark instance. |
| `verify_trajectory_theorem.py` | theorem check (composite) | Trajectory-decomposition verification on a composite test circuit. |
| `test_relabeling_hypothesis.py` | falsification of the naive claim | Demonstrates that the capstone's bit-flip-relabeling hypothesis fails, motivating Theorem 1. |

### Compilation (Section IV-D, Clifford+T)

| Script | Produces | Notes |
|---|---|---|
| `tcount_tradeoff.py` | Fig. 7 data (`tradeoff.csv`) | Ross–Selinger (`pygridsynth`) synthesis of every rotation at each precision ε; measures the incoherent-accumulation law `1-F ≈ c N ε²` and T-counts. |
| `endtoend_compiled.py` | compiled-circuit theorem check | Verifies Theorem 1 on the fully compiled circuit, where every non-Clifford is a T gate. |

### Figures and cross-checks

| Script | Produces |
|---|---|
| `fig_tcount_v3.py` | Fig. 7 (three-panel T-count/fidelity trade-off) |
| `fig_pf_candidates.py`, `finalize_fig8.py` | Fig. 8 (Pauli-frame two-panel: Porter–Thomas overlay + TV-distance convergence) |
| `review_checks.py` | Eq. (18) labeling reconciliation and the 30-instance universality of `c` |

### Data files (outputs, checked in as the record of results)

- `sweep_seeded.csv` — per-instance verification and resource data (all 30 files)
- `c30_seeded.csv` — Clifford+T coefficient `c` across all 30 instances at ε=1e-3
- `tradeoff.csv` — T-count/fidelity sweep for the benchmark instance across ε

## Typical workflow

```
# full verification sweep + resource counts (Table I, Appendix B)
python sweep_30.py            # or: python slices.py sweep 0 15  (then 15 30)

# Clifford+T trade-off (Fig. 7 data)
python tcount_tradeoff.py

# figures
python fig_tcount_v3.py
python fig_pf_candidates.py
```

## Note on `slices.py` / `sweep_lib.py`

`slices.py` imports a functions-only copy of `sweep_30.py` (written to
`sweep_lib.py` on first run) so it can reuse the machinery without triggering a
full sweep on import. If `sweep_30.py` is refactored, delete `sweep_lib.py` so
it regenerates.
