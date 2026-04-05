"""
tests/test_parity.py
--------------------
Smoke tests for the tetron parity measurement primitives.

Core property being tested: QND (non-destructive) measurements.

Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 is a +1 eigenstate of ZZ.
Measuring ZZ on |Φ+⟩ must:
    1. Return outcome 0 (+1 eigenvalue) with probability 1.
    2. Leave the state unchanged (up to floating point).

This is the exact demonstration Ed Chen gave in the April 2026 meeting.
"""

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Statevector

from src.tetron.parity import measure_zz, measure_xx, measure_z, measure_x


def bell_phi_plus(n_total: int = 2, q0: int = 0, q1: int = 1) -> DensityMatrix:
    """
    Prepare |Φ+⟩ on qubits q0, q1 in an n_total-qubit DensityMatrix.
    Other qubits are in |0⟩.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_total)
    qc.h(q0)
    qc.cx(q0, q1)
    return DensityMatrix(qc)


class TestZZMeasurement:
    def test_bell_state_zz_deterministic_outcome(self):
        """ZZ on |Φ+⟩ must yield outcome 0 (+1 eigenvalue) with prob 1."""
        rho = bell_phi_plus()
        rho_post, outcome = measure_zz(rho, 0, 1, forced_outcome=None)
        # Sample many times and verify always 0
        for _ in range(20):
            _, s = measure_zz(rho, 0, 1)
            assert s == 0, f"ZZ on |Φ+⟩ yielded outcome {s}, expected 0"

    def test_bell_state_zz_state_unchanged(self):
        """ZZ on |Φ+⟩ must not change the state (QND property)."""
        rho = bell_phi_plus()
        rho_post, outcome = measure_zz(rho, 0, 1, forced_outcome=0)
        fidelity = np.real(np.trace(np.array(rho.data) @ np.array(rho_post.data)))
        assert abs(fidelity - 1.0) < 1e-10, f"State changed after ZZ: fidelity={fidelity}"

    def test_product_state_zz_random(self):
        """ZZ on |+⟩|+⟩ should give random ±1 outcomes (50/50)."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        rho = DensityMatrix(qc)
        outcomes = [measure_zz(rho, 0, 1)[1] for _ in range(200)]
        frac = sum(outcomes) / len(outcomes)
        assert 0.3 < frac < 0.7, f"ZZ on |++⟩ not random: fraction={frac}"

    def test_zz_collapses_to_eigenstate(self):
        """After ZZ measurement, re-measurement must yield same outcome."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        rho = DensityMatrix(qc)
        rho_post, s1 = measure_zz(rho, 0, 1)
        _, s2 = measure_zz(rho_post, 0, 1)
        assert s1 == s2, f"Second ZZ gave different outcome: {s1} vs {s2}"


class TestXXMeasurement:
    def test_plus_plus_xx_deterministic(self):
        """XX on |++⟩ must yield outcome 0 with prob 1 (|++⟩ is +1 XX eigenstate)."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        rho = DensityMatrix(qc)
        for _ in range(20):
            _, s = measure_xx(rho, 0, 1)
            assert s == 0, f"XX on |++⟩ yielded {s}"


class TestZMeasurement:
    def test_z0_gives_0(self):
        """Z measurement on |0⟩ must give outcome 0."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1)
        rho = DensityMatrix(qc)
        _, s = measure_z(rho, 0)
        assert s == 0

    def test_z1_gives_1(self):
        """Z measurement on |1⟩ must give outcome 1."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1)
        qc.x(0)
        rho = DensityMatrix(qc)
        _, s = measure_z(rho, 0)
        assert s == 1
