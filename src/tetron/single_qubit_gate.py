"""
This module implements measurement-based single-qubit gates following the
Microsoft tetron roadmap paper 

For the supremacy single-qubit gates:
    √X ~ HSH  
    √Y ~ HSS  
    √W = T X^{1/2} T†  where W = (X+Y)/√2, and T state is prepared directly by MZMs
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.circuit.classical import expr

class TetronSingleQubitGates:
    def __init__(self):
        self.simulator = AerSimulator()
    
    @staticmethod
    def _rotate_to_z_basis(qc, qubit, pauli):
        """
        Apply gates to rotate from the Pauli eigenbasis to the Z basis to implement single qubit measurements on different 
        axis. The mapping is:
            Z -> nothing
            X -> H  
            Y -> HS†  
        """
        if pauli in ('I', 'Z'):
            pass
        elif pauli == 'X':
            qc.h(qubit)
        elif pauli == 'Y':
            qc.sdg(qubit)
            qc.h(qubit)
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")

    @staticmethod
    def _rotate_from_z_basis(qc, qubit, pauli):
        """
        Undo the basis rotation (inverse of _rotate_to_z_basis).

        Mappings:
            I -> nothing
            Z -> nothing
            X -> H
            Y -> SH
        """
        if pauli in ('I', 'Z'):
            pass
        elif pauli == 'X':
            qc.h(qubit)
        elif pauli == 'Y':
            qc.h(qubit)
            qc.s(qubit)
        else:
            raise ValueError(f"Unknown Pauli: {pauli}")
    
    # ---
    #  Two-qubit parity measurement M^{PQ}                                #
    # ----

    @staticmethod
    def measure_parity(qc, ancilla, data, pauli_ancilla, pauli_data,
                       creg, creg_idx):
        """
        Implement a parity measurement M^{PQ} where P acts on the ancilla
        and Q acts on the data qubit.

        Parameters:
            qc:             QuantumCircuit
            ancilla:        index of ancilla qubit
            data:           index of data qubit
            pauli_ancilla:  'X', 'Y', 'Z', or 'I' — Pauli on ancilla
            pauli_data:     'X', 'Y', 'Z', or 'I' — Pauli on data
            creg:           ClassicalRegister for measurement result
            creg_idx:       bit index in creg to store the outcome

        When pauli_data='I' the measurement is a single-qubit measurement on
        the ancilla only — no CNOT is applied and the data qubit is untouched.
        This is used for ancilla initialisation/readout steps (e.g. M^{XI}).

        The measurement outcome s ∈ {0, 1} maps to eigenvalue +1 (s=0) or -1 (s=1).
        """
        TetronSingleQubitGates._rotate_to_z_basis(qc, ancilla, pauli_ancilla)

        if pauli_data != 'I':
            TetronSingleQubitGates._rotate_to_z_basis(qc, data, pauli_data)
            qc.cx(data, ancilla)

        qc.measure(ancilla, creg[creg_idx])

        if pauli_data != 'I':
            qc.cx(data, ancilla)
            TetronSingleQubitGates._rotate_from_z_basis(qc, data, pauli_data)

        TetronSingleQubitGates._rotate_from_z_basis(qc, ancilla, pauli_ancilla)
    
    @staticmethod
    def gate_H(qc, ancilla, data, creg, start_idx=0):
        """
        Implement [H] gate.

        Measurement sequence: M^{XI}, M^{ZY}, M^{YI}, M^{XI}

        Correction on data qubit — Y^{(1+s0*s1*s2)/2} · X:

            parity_012 = c0 XOR c1 XOR c2
            parity odd  (s0*s1*s2 = -1, y_exp=0): apply X
            parity even (s0*s1*s2 = +1, y_exp=1): apply Z  (Y·X = iZ up to phase)
        """
        c = creg
        i = start_idx

        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 1)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 2)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 3)

        # Pauli correction
        parity_012 = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 2])
        with qc.if_test(parity_012):           
            qc.x(data)
        with qc.if_test(expr.logic_not(parity_012)): 
            qc.z(data)

        qc.reset(ancilla)

        return qc
    
    @staticmethod
    def gate_S(qc, ancilla, data, creg, start_idx=0):
        """
        Implement [S].

        Measurement sequence: M^{XI}, M^{ZZ}, M^{YI}, M^{XI}

        Correction on data qubit — Z^{(1+s0*s1*s2)/2}:
            parity_012 = c0 XOR c1 XOR c2
            even parity (s0*s1*s2 = +1, exponent=1): apply Z
        """
        c = creg
        i = start_idx

        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 1)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 2)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 3)

        # Pauli correction
        parity_012 = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 2])
        with qc.if_test(expr.logic_not(parity_012)):  
            qc.z(data)

        qc.reset(ancilla)

        return qc
    
    @staticmethod
    def gate_HSH(qc, ancilla, data, creg, start_idx=0):
        """
        Implement [HSH] gate.

        Measurement sequence: M^{XI}, M^{ZZ}, M^{ZY}, M^{XI}

        Correction on data qubit — Y^{(1-s0*s3)/2} · X^{(1+s1*s2)/2}:
            y_par   = c0 XOR c3          → y_exp=1 when 1
            x12_par = c1 XOR c2          → x_exp_raw=1 when 0
            z_exp = y_par  → apply Z when y_par=1
            x_exp = NOT(y_par XOR x12_par) = XNOR → apply X when NOT(y_par XOR x12_par)
        """
        c = creg
        i = start_idx

        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 1)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 2)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 3)

        # Pauli correction
        y_par   = expr.bit_xor(c[i], c[i + 3])
        x12_par = expr.bit_xor(c[i + 1], c[i + 2])
        with qc.if_test(y_par):                                           # apply Z
            qc.z(data)
        with qc.if_test(expr.logic_not(expr.bit_xor(y_par, x12_par))):   # apply X
            qc.x(data)

        qc.reset(ancilla)

        return qc

    @staticmethod
    def gate_SH(qc, ancilla, data, creg, start_idx=0):
        """
        Implement [SH] gate.
        Measurement sequence: M^{XI}, M^{ZZ}, M^{ZY}, M^{YI}, M^{XI}

        Correction on data qubit — Y^{(1-s0*s2*s3)/2} · Z^{(1-s1*s2)/2}:
            x_exp = y_exp → apply X when c0 XOR c2 XOR c3 = 1
            z_exp = y_exp XOR z_exp_raw = c0 XOR c1 XOR c3  (c2 cancels)
        """
        c = creg
        i = start_idx

        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 1)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 2)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 3)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 4)

        # Pauli correction
        y_par = expr.bit_xor(expr.bit_xor(c[i], c[i + 2]), c[i + 3])
        z_par = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 3])
        with qc.if_test(y_par):   # apply X (x_exp = y_exp)
            qc.x(data)
        with qc.if_test(z_par):   # apply Z
            qc.z(data)

        qc.reset(ancilla)

        return qc
    
    @staticmethod
    def gate_HS(qc, ancilla, data, creg, start_idx=0):
        """
        Implement [HS] gate via 5 measurements + inline Pauli correction.

        Measurement sequence: M^{XI}, M^{ZY}, M^{ZZ}, M^{YI}, M^{XI}

        Correction on data qubit — X^{(1+s1*s2)/2} · Z^{(1+s0*s1*s3)/2}:
            apply X when c1 XOR c2 = 0  (even parity → product = +1)
            apply Z when c0 XOR c1 XOR c3 = 0
        """
        c = creg
        i = start_idx

        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 1)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 2)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 3)
        TetronSingleQubitGates.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 4)

        # Pauli correction
        x12_par  = expr.bit_xor(c[i + 1], c[i + 2])
        z013_par = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 3])
        with qc.if_test(expr.logic_not(x12_par)):   # even parity → apply X
            qc.x(data)
        with qc.if_test(expr.logic_not(z013_par)):  # even parity → apply Z
            qc.z(data)

        qc.reset(ancilla)

        return qc
    
    @staticmethod
    def gate_sqrt_X(qc, ancilla, data, creg, start_idx=0):
        """
        Implement √X gate via the Clifford decomposition HSH.
        """
        TetronSingleQubitGates.gate_HSH(qc, ancilla, data, creg, start_idx)
        return qc
    
    @staticmethod
    def gate_sqrt_Y(qc, ancilla, data, creg, start_idx=0):
        """
        Implement √Y gate via the Clifford decomposition HSS = (HS)·S.
        """
        # Apply [S] first (4 measurements)
        TetronSingleQubitGates.gate_S(qc, ancilla, data, creg, start_idx)
        # Then apply [HS] (5 measurements)
        TetronSingleQubitGates.gate_HS(qc, ancilla, data, creg, start_idx + 4)
        return qc
    
    @staticmethod
    def gate_sqrt_W(qc, ancilla, data, creg, start_idx=0):
        """
        Implement √W gate where W = (X+Y)/√2.
        √W = T · √X · T†

        Since the T state can be prepared directly using MZMs (roadmap paper,
        Appendix D), we treat T as a directly available unitary gate in
        simulation. 
        Circuit on data qubit:
            T† → [HSH] → T

        Total measurements: 4 (from [HSH] = [√X])
        """
        # Apply T† on data qubit (directly available)
        qc.tdg(data)
        TetronSingleQubitGates.gate_HSH(qc, ancilla, data, creg, start_idx)
        # Apply T on data qubit (directly available)
        qc.t(data)

        return qc
    
