# mbqc-circuit-comparison

Translating Google's 2019 random circuit sampling (RCS) benchmark into the
parity-measurement–based computational model native to Majorana **tetron**
arrays, and verifying the translation against Google's published amplitudes.

This repository accompanies the paper *Random Circuit Sampling on Majorana
Tetron Arrays: Translation, Resource Costs, and the Pauli Frame as Classical
Bookkeeping* (in preparation). It contains the gate-to-measurement translation
engine, the tetron layout, loaders for Google's published circuit and amplitude
files, and the analysis scripts that reproduce every table and figure in the
paper.

## What this does

Tetron arrays compute through joint fermion-parity measurements with classical
Pauli-frame tracking rather than pulsed unitary gates. This project:

- **translates** each gate of Google's 12-qubit Sycamore circuits into
  sequences of two-qubit parity measurements with feed-forward Pauli
  corrections, on an 8×3 tetron layout that preserves nearest-neighbor
  connectivity;
- **verifies** the translated circuits against Google's published noiseless
  amplitudes (and against exact statevector simulation) across all 30 published
  12-qubit instances;
- **measures** the measurement/ancilla resource overhead, and the Clifford+T
  cost of compiling arbitrary rotations into the hardware-native gate set;
- **analyzes** what a measurement-based processor computes when Pauli-frame
  corrections are omitted (the trajectory-decomposition result of the paper).

## Repository layout

| Path | Contents |
|---|---|
| `src/tetron/` | Core library: the translation engine (`mbqc_translated_gates.py`), tetron array and parity primitives (`array.py`, `parity.py`), Pauli-frame tracking (`pauli_frame.py`), and the 12- and 53-qubit layouts (`qubit_mapping.py`, `qubit_mapping_53.py`). |
| `src/circuits/`, `src/benchmarks/` | Circuit builders and the XEB benchmark. |
| `analysis/` | Scripts reproducing the paper's tables and figures, plus the seeded result CSVs. See [`analysis/README.md`](analysis/README.md). |
| `google_supremacy_circuit_files/` | Google's published 12-qubit circuit files (`.py` and `.qasm`). |
| `google_53qubits_circuit/` | Google's 53-qubit circuit files (used for the Clifford-only scaling checks). |
| `data/google_amplitudes/` | Google's published output amplitudes for the verifiable 12-qubit instances. |
| `tests/` | Unit tests for the parity and Pauli-frame primitives. |
| `*.ipynb` (root) | Exploratory and demonstration notebooks (see note below). |

## Installation

```bash
git clone https://github.com/UnitaryOperator/mbqc-circuit-comparison
cd mbqc-circuit-comparison
pip install -r requirements.txt
```

Requires Python 3.10+. Main dependencies: `qiskit`, `qiskit-aer`, `cirq`,
`pygridsynth`, `numpy`, `matplotlib`.

## Reproducing the paper's results

All quoted numbers come from a single **seeded** compilation of each circuit
(the fSim decomposition is pinned via `seed_transpiler` in
`mbqc_translated_gates.py`), so results are bit-for-bit reproducible. Start
here:

```bash
cd analysis
python review_checks.py     # fast (~30 s) sanity check of the full pipeline
python sweep_30.py          # main verification sweep (paper Table I, Appendix B)
python tcount_tradeoff.py   # Clifford+T trade-off (paper Fig. 7)
```

See [`analysis/README.md`](analysis/README.md) for the full script-by-script
guide.

## Notebooks

The Jupyter notebooks in the repository root and `src/tetron/` are exploratory
and development notebooks from the course project that produced this work. The
maintained, paper-referenced entry points are
`google12_tetron_vs_google_amplitudes_statevector_v2.ipynb` (single-instance
equivalence check) and `src/tetron/pauli_frame_correction_study.ipynb`
(Pauli-frame analysis). The remaining notebooks are retained for reference and
may not reflect the final pipeline; the scripts in `analysis/` are the
authoritative reproduction path.

## Authors

University of Washington: Shuyun Liu, Koray Mentesoglu, Chris Moore, Xiangyu Shi.
Microsoft (mentors): Matt Brooks, Ed Chen, Adam Mills.
Instructor: Prof. Sara Mouradian.

## License

MIT — see [LICENSE](LICENSE).
