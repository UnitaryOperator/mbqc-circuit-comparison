"""
src/circuits/builder.py
------------------------
Converts a preprocessed Google 2019 circuit JSON file into a Qiskit
QuantumCircuit using the tetron MBQC gate library (src/tetron/gates.py).

Workflow
--------
1. Load a JSON file produced by preprocess_circuits.py
2. Build a qubit map: tetron label → Qiskit wire indices (data, ancilla)
3. Build an ancilla routing table: (tetron_c, tetron_t) → ancilla wire
4. Walk each moment in the JSON and call the corresponding gate function
5. Return the completed QuantumCircuit ready for Qiskit simulation

Qubit layout
------------
For n logical qubits (tetron labels from qubit_order), the Qiskit
circuit has 2n wires:
    logical qubit at position i in qubit_order:
        data qubit   → Qiskit wire 2*i
        ancilla      → Qiskit wire 2*i + 1

For the 12-qubit Google circuits: 24 Qiskit wires total (12 data + 12 ancilla).

Ancilla routing for two-qubit gates
-------------------------------------
When add_TwoQubitGate(qc, c, a, t, ...) is called, the ancilla `a`
mediates the gate between data qubits `c` and `t`.

Convention (vertical/horizontal):
    - "Vertical" edge:   control c → ancilla a  (ZX measurement c→a)
    - "Horizontal" edge: ancilla a → target t   (ZX measurement a→t)

The ancilla used is always the one adjacent to the control qubit:
    a = 2*i_c + 1
where i_c is the position of the control tetron label in qubit_order.

This is a simulation convention. On physical tetron hardware the routing
depends on the 8×3 layout adjacency — confirm with mentors before
extending to hardware-accurate simulation.

Classical register
------------------
A single ClassicalRegister of 8 bits is shared across all gates.
Bits 0–4 are used by single-qubit gates; bits 5–7 are used by add_CNOT.
Because each gate's feedforward conditional is evaluated immediately
after its measurement, bits can be safely reused across sequential gates.

Usage
-----
    from src.circuits.builder import build_circuit

    qc = build_circuit("src/circuits/processed/circuit_n12_m14_s0_e0_pEFGH_snapped.json")
    # qc is a QuantumCircuit ready for AerSimulator

    # Noiseless ideal probabilities (circuit model baseline):
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc_no_measurements)
    probs = sv.probabilities_dict()
"""

import json
import math
from pathlib import Path

from qiskit import QuantumCircuit, ClassicalRegister

from src.tetron.gates import (
    add_sqrtX, add_sqrtY, add_sqrtW,
    add_TwoQubitGate,
)

# Ideal noiseless FSimGate parameters (snapped from per-pair calibrated values)
THETA_IDEAL = math.pi / 2   # π/2
PHI_IDEAL   = math.pi / 6   # π/6

# Classical register size (must be >= 8; matches notebook)
CBIT_SIZE = 8


def _build_qubit_map(qubit_order: list[int]) -> dict[int, tuple[int, int]]:
    """
    Build mapping from tetron logical label → (data_wire, ancilla_wire).

    Parameters
    ----------
    qubit_order : list of tetron labels in position order
                  e.g. [11, 4, 3, 9, 8, 1, 2, 10, 7, 5, 6, 12]

    Returns
    -------
    dict: tetron_label → (data_wire, ancilla_wire)
    """
    return {
        label: (2 * i, 2 * i + 1)
        for i, label in enumerate(qubit_order)
    }


def _build_ancilla_map(
    qubit_order: list[int],
    qubit_map: dict[int, tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """
    Build ancilla routing table for two-qubit gates.

    For each coupling pair (tetron_c, tetron_t), determine which Qiskit
    wire is used as the shared ancilla.

    Convention: use the control qubit's ancilla (2*i_c + 1).
    This implements the vertical→horizontal routing:
        ZX(c → a_c)  then  ZX(a_c → t)

    Parameters
    ----------
    qubit_order : list of tetron labels
    qubit_map   : tetron_label → (data_wire, ancilla_wire)

    Returns
    -------
    dict: (tetron_c, tetron_t) → ancilla_wire
          Also populated for (tetron_t, tetron_c) using the same ancilla,
          so gate order in the JSON doesn't matter.
    """
    ancilla_map = {}
    for i, label_c in enumerate(qubit_order):
        _, a_c = qubit_map[label_c]
        for j, label_t in enumerate(qubit_order):
            if i != j:
                # Convention: control's ancilla always used
                ancilla_map[(label_c, label_t)] = a_c
    return ancilla_map


def build_circuit(
    json_path: str | Path,
    theta: float = THETA_IDEAL,
    phi: float   = PHI_IDEAL,
    add_measurements: bool = False,
) -> QuantumCircuit:
    """
    Build a Qiskit QuantumCircuit from a preprocessed circuit JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to a JSON file produced by preprocess_circuits.py.
        Typically: src/circuits/processed/circuit_n12_m14_s0_e0_pEFGH_snapped.json

    theta : float
        FSimGate θ parameter. Default π/2 (idealised noiseless).

    phi : float
        FSimGate φ parameter. Default π/6 (idealised noiseless).

    add_measurements : bool
        If True, append Z-basis measurements on all data qubits at the
        end of the circuit. Default False (leave unmeasured for
        Statevector / ideal probability computation).

    Returns
    -------
    QuantumCircuit
        All qubits initialised to |0⟩. Data qubits are at even wire
        indices; ancilla qubits are at odd wire indices.

    Examples
    --------
    # MBQC circuit for XEB scoring:
    qc = build_circuit("src/circuits/processed/circuit_n12_m14_s0_e0_pEFGH_snapped.json")

    # With measurements for sampling:
    qc = build_circuit(..., add_measurements=True)

    # Override FSimGate parameters (e.g. for sensitivity study):
    qc = build_circuit(..., theta=math.pi/2, phi=0.55)
    """
    path = Path(json_path)
    with open(path) as f:
        circuit = json.load(f)

    qubit_order: list[int] = circuit["qubit_order"]
    n_logical = len(qubit_order)
    n_qiskit  = 2 * n_logical          # data + ancilla wires

    # --- Qubit and ancilla maps ---
    qubit_map   = _build_qubit_map(qubit_order)
    ancilla_map = _build_ancilla_map(qubit_order, qubit_map)

    # --- Build Qiskit circuit ---
    cbit = ClassicalRegister(CBIT_SIZE, name="c")
    qc   = QuantumCircuit(n_qiskit, name=path.stem)
    qc.add_register(cbit)

    # --- Walk moments ---
    for moment in circuit["moments"]:
        for gate in moment["gates"]:
            g    = gate["gate"]
            qlabels = gate["qubits"]   # tetron logical labels

            if g == "sqrtX":
                d, a = qubit_map[qlabels[0]]
                qc = add_sqrtX(qc, d, a, cbit)

            elif g == "sqrtY":
                d, a = qubit_map[qlabels[0]]
                qc = add_sqrtY(qc, d, a, cbit)

            elif g == "sqrtW":
                d, a = qubit_map[qlabels[0]]
                qc = add_sqrtW(qc, d, a, cbit)

            elif g == "FSimGate":
                label_c, label_t = qlabels[0], qlabels[1]
                c_wire, _  = qubit_map[label_c]
                t_wire, _  = qubit_map[label_t]
                a_wire     = ancilla_map[(label_c, label_t)]

                qc = add_TwoQubitGate(
                    qc,
                    c_wire, a_wire, t_wire,
                    theta, phi,
                    0, 0, 0, 0,     # Z1, Z2, Z3, Z4 = 0 (noiseless)
                    cbit,
                )

            else:
                raise ValueError(
                    f"Unknown gate '{g}' in moment {moment['depth']} of {path.name}"
                )

    # --- Optional final measurements on data qubits ---
    if add_measurements:
        meas_reg = ClassicalRegister(n_logical, name="meas")
        qc.add_register(meas_reg)
        for i, label in enumerate(qubit_order):
            d, _ = qubit_map[label]
            qc.measure(d, meas_reg[i])

    return qc


def circuit_metadata(json_path: str | Path) -> dict:
    """
    Return circuit metadata without building the full Qiskit circuit.
    Useful for inspecting a circuit before committing to simulation.

    Returns
    -------
    dict with keys: source_file, seed, elided, is_patch, n_qubits,
                    n_moments, n_fsim, n_single_qubit, qubit_order
    """
    with open(json_path) as f:
        d = json.load(f)
    return {k: d[k] for k in [
        "source_file", "seed", "elided", "is_patch",
        "n_qubits", "n_moments", "n_fsim", "n_single_qubit", "qubit_order",
    ]}
