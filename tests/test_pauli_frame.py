"""
tests/test_pauli_frame.py
-------------------------
Tests for the Pauli frame tracker.
"""

import pytest
from src.tetron.pauli_frame import PauliFrame


class TestPauliFrameInit:
    def test_all_zero_at_init(self):
        frame = PauliFrame(n_qubits=8)
        for q in range(8):
            assert frame.get_frame(q) == (0, 0)


class TestZZUpdate:
    def test_outcome_0_no_update(self):
        frame = PauliFrame(n_qubits=8)
        frame.update_from_zz(data_qubit=0, ancilla_qubit=4, outcome=0)
        assert frame.get_frame(0) == (0, 0)
        assert frame.get_frame(4) == (0, 0)

    def test_outcome_1_flips_z(self):
        frame = PauliFrame(n_qubits=8)
        frame.update_from_zz(data_qubit=0, ancilla_qubit=4, outcome=1)
        assert frame.get_frame(0) == (0, 1)
        assert frame.get_frame(4) == (0, 1)

    def test_double_update_cancels(self):
        """Two ZZ outcomes of 1 should cancel (XOR)."""
        frame = PauliFrame(n_qubits=8)
        frame.update_from_zz(0, 4, outcome=1)
        frame.update_from_zz(0, 4, outcome=1)
        assert frame.get_frame(0) == (0, 0)


class TestReadoutCorrection:
    def test_z_readout_no_frame(self):
        frame = PauliFrame(n_qubits=4)
        assert frame.correct_z_readout(qubit=0, raw_outcome=1) == 1

    def test_z_readout_with_z_frame(self):
        frame = PauliFrame(n_qubits=4)
        frame.update_from_zz(0, 2, outcome=1)   # sets z_frame[0] = 1
        assert frame.correct_z_readout(qubit=0, raw_outcome=1) == 0  # 1 XOR 1 = 0
        assert frame.correct_z_readout(qubit=0, raw_outcome=0) == 1  # 0 XOR 1 = 1

    def test_reset_qubit(self):
        frame = PauliFrame(n_qubits=4)
        frame.update_from_zz(0, 2, outcome=1)
        frame.reset_qubit(0)
        assert frame.get_frame(0) == (0, 0)
