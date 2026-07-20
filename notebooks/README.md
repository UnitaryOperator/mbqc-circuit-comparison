# Notebooks

Maintained, paper-referenced notebooks. Launch Jupyter from **this directory**
(`notebooks/`) so the relative paths to the repository resolve correctly.

| Notebook | Purpose |
|---|---|
| `google12_tetron_vs_google_amplitudes_statevector_v2.ipynb` | Single-instance equivalence check: translate a Google 12-qubit circuit, simulate, and compare against the published amplitudes (paper Appendix). |
| `pauli_frame_correction_study.ipynb` | Pauli-frame analysis: corrected vs. uncorrected trajectories (paper Sec. V / Fig. 8). |
| `google12_tetron_vs_direct_statevector.ipynb` | Equivalence against exact statevector simulation of the same circuit. |
| `google53_tetron_vs_direct_stabilizer_inverse_check.ipynb` | 53-qubit Clifford-only scaling check via the stabilizer formalism. |

For batch reproduction of all tables and figures, prefer the scripts in
[`../analysis/`](../analysis/). Older development notebooks are in
[`exploratory/`](exploratory/).
