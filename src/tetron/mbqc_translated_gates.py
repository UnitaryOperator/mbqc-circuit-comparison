"""
mbqc_translated_gates.py

Adds an `apply_pauli_corrections` flag to every gate method. When True
(default) the gate behaves identically to the original implementation —
mid-circuit measurements are followed by feed-forward Pauli corrections
that compensate for the random byproducts of each parity measurement.
When False, the parity measurements still happen (so the state's
trajectory is identical up to the corrections), but no compensating
X/Z is applied to the data qubit. This lets you study what the output
distribution looks like when the Pauli frame is left uncorrected.

The Rz/S rotations, T/Tdg wrappers in sqrt_W, and the Z1..Z4 rotations
in gate_two_qubit are gate-definition rotations (not byproduct
corrections) and are always applied regardless of the flag.
"""
import numpy as np
from scipy.linalg import sqrtm
from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.circuit.library import UnitaryGate, CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.compiler import transpile


# =========================================================================
#  Module-level helpers (matrices)
# =========================================================================
def fSim_matrix(theta, phi):
    """fSim(theta, phi) unitary as a 4x4 numpy array."""
    return np.array([
        [1, 0,                       0,                       0],
        [0, np.cos(theta),          -1j * np.sin(theta),      0],
        [0, -1j * np.sin(theta),     np.cos(theta),           0],
        [0, 0,                       0,                       np.exp(-1j * phi)],
    ], dtype=complex)


def sqrt_W_matrix():
    """sqrt(W) where W = (X + Y)/sqrt(2)."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    W = (X + Y) / np.sqrt(2)
    return sqrtm(W)


class MBQCTranslatedGates:
    """
    Measurement-based single- and two-qubit gates.

    All single-qubit gate methods have signature
        gate(qc, ancilla, data, creg, start_idx=0, apply_pauli_corrections=True)
    matching TetronSingleQubitGates. All two-qubit gate methods have
    signature
        gate(qc, control, ancilla, target, creg, start_idx=0[, ...params],
             apply_pauli_corrections=True)

    The number of classical bits consumed by each gate can be queried via
    the *_n_bits class attributes (single-qubit) or via the n_bits_two_qubit
    helper (two-qubit, which depends on the synthesis). The bit count
    does NOT depend on apply_pauli_corrections — the measurements still
    happen either way; only the conditional X/Z byproduct gates are
    suppressed when the flag is False.
    """

    # ---------------------------------------------------------------------
    # number of classical bits used
    # ---------------------------------------------------------------------
    N_BITS = {
        'H':       4,
        'S':       4,
        'HSH':     4,
        'SH':      5,
        'HS':      5,
        'sqrt_X':  4,
        'sqrt_Y':  9,
        'sqrt_W':  4,
    }

    # classical bits of CNOT
    CNOT_N_BITS = 3

    # =====================================================================
    # rotations
    # =====================================================================
    @staticmethod
    def _rotate_to_z_basis(qc, qubit, pauli):
        """Rotate from Pauli eigenbasis to Z basis."""
        if pauli in ('I', 'Z'):
            return
        if pauli == 'X':
            qc.h(qubit)
        elif pauli == 'Y':
            qc.sdg(qubit)
            qc.h(qubit)
        else:
            raise ValueError(f"unknown Pauli: {pauli!r}")

    @staticmethod
    def _rotate_from_z_basis(qc, qubit, pauli):
        """Inverse of _rotate_to_z_basis."""
        if pauli in ('I', 'Z'):
            return
        if pauli == 'X':
            qc.h(qubit)
        elif pauli == 'Y':
            qc.h(qubit)
            qc.s(qubit)
        else:
            raise ValueError(f"unknown Pauli: {pauli!r}")

    # =====================================================================
    # two-qubit parity measurement  M^{PQ}
    # =====================================================================
    @staticmethod
    def measure_parity(qc, ancilla, data, pauli_ancilla, pauli_data,
                       creg, creg_idx):
        """
        M^{PQ} with P on ancilla, Q on data, outcome stored in creg[creg_idx].
        pauli_data='I' means a single-qubit measurement on the ancilla.

        This is a pure measurement — it does not depend on the
        apply_pauli_corrections flag, since it is the source of byproducts,
        not a correction step.
        """
        cls = MBQCTranslatedGates
        cls._rotate_to_z_basis(qc, ancilla, pauli_ancilla)

        if pauli_data != 'I':
            cls._rotate_to_z_basis(qc, data, pauli_data)
            qc.cx(data, ancilla)

        qc.measure(ancilla, creg[creg_idx])

        if pauli_data != 'I':
            qc.cx(data, ancilla)
            cls._rotate_from_z_basis(qc, data, pauli_data)

        cls._rotate_from_z_basis(qc, ancilla, pauli_ancilla)

    # =====================================================================
    # single-qubit gates
    # =====================================================================
    @staticmethod
    def gate_H(qc, ancilla, data, creg, start_idx=0,
               apply_pauli_corrections=True):
        """
        Measurements: M^{XI}, M^{ZY}, M^{YI}, M^{XI}
        Correction: parity_012 odd -> X, parity_012 even -> Z
                    (equivalent to Y^{(1+s0 s1 s2)/2} * X)

        If apply_pauli_corrections=False, the measurements still happen
        but the conditional X/Z byproduct gates are omitted.
        """
        c = creg
        i = start_idx
        cls = MBQCTranslatedGates
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 1)
        cls.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 2)
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 3)

        if apply_pauli_corrections:
            parity_012 = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 2])
            with qc.if_test(parity_012):
                qc.x(data)
            with qc.if_test(expr.logic_not(parity_012)):
                qc.z(data)

        return qc

    @staticmethod
    def gate_S(qc, ancilla, data, creg, start_idx=0,
               apply_pauli_corrections=True):
        """
        Measurements: M^{XI}, M^{ZZ}, M^{YI}, M^{XI}
        Correction: Z^{(1 + s0 s1 s2)/2}  (apply Z on even parity)
        """
        c = creg
        i = start_idx
        cls = MBQCTranslatedGates
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 1)
        cls.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 2)
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 3)

        if apply_pauli_corrections:
            parity_012 = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 2])
            with qc.if_test(expr.logic_not(parity_012)):
                qc.z(data)

        return qc

    @staticmethod
    def gate_HSH(qc, ancilla, data, creg, start_idx=0,
                 apply_pauli_corrections=True):
        """
        Measurements: M^{XI}, M^{ZZ}, M^{ZY}, M^{XI}
        Correction: Y^{(1 - s0 s3)/2} * X^{(1 + s1 s2)/2}
        """
        c = creg
        i = start_idx
        cls = MBQCTranslatedGates
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 1)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 2)
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 3)

        if apply_pauli_corrections:
            y_par   = expr.bit_xor(c[i], c[i + 3])
            x12_par = expr.bit_xor(c[i + 1], c[i + 2])
            with qc.if_test(y_par):
                qc.z(data)
            with qc.if_test(expr.logic_not(expr.bit_xor(y_par, x12_par))):
                qc.x(data)

        return qc

    @staticmethod
    def gate_SH(qc, ancilla, data, creg, start_idx=0,
                apply_pauli_corrections=True):
        """
        Measurements: M^{XI}, M^{ZZ}, M^{ZY}, M^{YI}, M^{XI}
        Correction: Y^{(1 - s0 s2 s3)/2} * Z^{(1 - s1 s2)/2}
        """
        c = creg
        i = start_idx
        cls = MBQCTranslatedGates
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 1)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 2)
        cls.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 3)
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 4)

        if apply_pauli_corrections:
            y_par = expr.bit_xor(expr.bit_xor(c[i], c[i + 2]), c[i + 3])
            z_par = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 3])
            with qc.if_test(y_par):
                qc.x(data)
            with qc.if_test(z_par):
                qc.z(data)

        return qc

    @staticmethod
    def gate_HS(qc, ancilla, data, creg, start_idx=0,
                apply_pauli_corrections=True):
        """
        Measurements: M^{XI}, M^{ZY}, M^{ZZ}, M^{YI}, M^{XI}
        Correction: X^{(1 + s1 s2)/2} * Z^{(1 + s0 s1 s3)/2}
        """
        c = creg
        i = start_idx
        cls = MBQCTranslatedGates
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Y', c, i + 1)
        cls.measure_parity(qc, ancilla, data, 'Z', 'Z', c, i + 2)
        cls.measure_parity(qc, ancilla, data, 'Y', 'I', c, i + 3)
        cls.measure_parity(qc, ancilla, data, 'X', 'I', c, i + 4)

        if apply_pauli_corrections:
            x12_par  = expr.bit_xor(c[i + 1], c[i + 2])
            z013_par = expr.bit_xor(expr.bit_xor(c[i], c[i + 1]), c[i + 3])
            with qc.if_test(expr.logic_not(x12_par)):
                qc.x(data)
            with qc.if_test(expr.logic_not(z013_par)):
                qc.z(data)

        return qc

    # ---------------------------------------------------------------------
    # supremacy single-qubit gates
    # ---------------------------------------------------------------------
    @staticmethod
    def gate_sqrt_X(qc, ancilla, data, creg, start_idx=0,
                    apply_pauli_corrections=True):
        """sqrt(X) = HSH. 4 measurements."""
        return MBQCTranslatedGates.gate_HSH(
            qc, ancilla, data, creg, start_idx,
            apply_pauli_corrections=apply_pauli_corrections,
        )

    @staticmethod
    def gate_sqrt_Y(qc, ancilla, data, creg, start_idx=0,
                    apply_pauli_corrections=True):
        """sqrt(Y) = HSS = (HS) * S. 9 measurements (4 + 5)."""
        MBQCTranslatedGates.gate_S(
            qc, ancilla, data, creg, start_idx,
            apply_pauli_corrections=apply_pauli_corrections,
        )
        MBQCTranslatedGates.gate_HS(
            qc, ancilla, data, creg, start_idx + 4,
            apply_pauli_corrections=apply_pauli_corrections,
        )
        return qc

    @staticmethod
    def gate_sqrt_W(qc, ancilla, data, creg, start_idx=0,
                    apply_pauli_corrections=True):
        """
        sqrt(W) with W = (X+Y)/sqrt(2), implemented as T * sqrt(X) * T_dag.
        T is treated as directly available (T-state via MZMs). 4 measurements.

        The T and Tdg are gate-definition rotations (not Pauli byproduct
        corrections) and are always applied.
        """
        qc.tdg(data)
        MBQCTranslatedGates.gate_HSH(
            qc, ancilla, data, creg, start_idx,
            apply_pauli_corrections=apply_pauli_corrections,
        )
        qc.t(data)
        return qc

    # =====================================================================
    # CNOT via three single-qubit measurements
    # =====================================================================
    @staticmethod
    def gate_CNOT(qc, control, ancilla, target, creg, start_idx=0,
                  apply_pauli_corrections=True):
        """
        Measurement-based CNOT from control -> target using a shared ancilla.

        Uses 3 measurement outcomes stored in creg[start_idx .. start_idx+2]:
            creg[start_idx]   = P1 (ZX on (control, ancilla))
            creg[start_idx+1] = P2 (ZX on (ancilla,  target))
            creg[start_idx+2] = M  (X measurement on ancilla)

        Pauli corrections:
            X_target  if  P1 XOR M = 1
            Z_control if  P2 = 1

        The ancilla is reset before and after.
        """
        c = creg
        i = start_idx
        cls = MBQCTranslatedGates

        qc.reset(ancilla)

        cls.measure_parity(qc, ancilla, control, 'X', 'Z', c, i)
        cls.measure_parity(qc, target, ancilla, 'X', 'Z', c, i + 1)

        qc.h(ancilla)
        qc.measure(ancilla, c[i + 2])
        qc.h(ancilla)

        if apply_pauli_corrections:
            P1 = c[i]
            P2 = c[i + 1]
            M  = c[i + 2]

            with qc.if_test(expr.bit_xor(P1, M)):
                qc.x(target)
            with qc.if_test((P2, 1)):
                qc.z(control)

        return qc

    # =====================================================================
    # two-qubit fSim gate
    # =====================================================================
    @staticmethod
    def _apply_rz_or_s(qc, qubit, angle):
        """Rz(angle) to Clifford S^k when angle is a multiple of pi/2."""
        theta = (angle + np.pi) % (2 * np.pi) - np.pi
        k = round(theta / (np.pi / 2))
        if abs(theta - k * (np.pi / 2)) < 1e-3:
            k_mod = k % 4
            for _ in range(k_mod):
                qc.s(qubit)
        else:
            qc.rz(theta, qubit)
        return qc

    @staticmethod
    def _zsx_blocks_and_cx(theta, phi):
        qc = QuantumCircuit(2)
        qc.append(UnitaryGate(fSim_matrix(theta, phi)), [0, 1])
        U_target = Operator(qc).data

        decomposer = TwoQubitBasisDecomposer(CXGate(), euler_basis='ZSX')
        synth = decomposer(U_target)
        synth = transpile(synth, basis_gates=['rz', 'sx', 'x', 'cx'],
                          optimization_level=3, seed_transpiler=20260709)

        blocks = []
        cx_dirs = []
        current = {0: [], 1: []}

        for instr in synth.data:
            inst = instr.operation
            qargs = instr.qubits
            name = inst.name

            if name == 'cx':
                blocks.append(current)
                current = {0: [], 1: []}
                ctrl = synth.find_bit(qargs[0]).index
                targ = synth.find_bit(qargs[1]).index
                cx_dirs.append((ctrl, targ))
                continue

            q = synth.find_bit(qargs[0]).index
            if name == 'rz':
                current[q].append(('rz', float(inst.params[0])))
            elif name == 'sx':
                current[q].append(('sx', None))
            elif name == 'x':
                current[q].append(('x', None))

        blocks.append(current)
        return blocks, cx_dirs

    @staticmethod
    def n_bits_two_qubit(theta, phi):
        blocks, cx_dirs = MBQCTranslatedGates._zsx_blocks_and_cx(theta, phi)
        cls = MBQCTranslatedGates
        n = 0
        for block in blocks:
            for q in (0, 1):
                for name, _ in block[q]:
                    if name == 'sx':
                        n += cls.N_BITS['sqrt_X']
                    elif name == 'x':
                        n += 2 * cls.N_BITS['sqrt_X']
        n += len(cx_dirs) * cls.CNOT_N_BITS
        return n

    @staticmethod
    def gate_two_qubit(qc, control, ancilla, target, creg,
                       theta, phi, Z1=0.0, Z2=0.0, Z3=0.0, Z4=0.0,
                       start_idx=0, apply_pauli_corrections=True):
        """
        Generic two-qubit gate Rz(Z3)_c Rz(Z4)_t * fSim(theta, phi) * Rz(Z1)_c Rz(Z2)_t.

        Uses a shared ancilla `ancilla` distinct from both `control` and `target`.
        Classical bits consumed: n_bits_two_qubit(theta, phi). Caller must size
        creg accordingly (use n_bits_two_qubit to find out).

        The Z1..Z4 wrappers are convenient when reproducing Cirq-style FSim
        moments that absorb arbitrary single-qubit Z rotations on either side.
        Set them all to 0 for a bare fSim. These Rz rotations are gate-
        definition rotations and are always applied regardless of
        apply_pauli_corrections.

        When apply_pauli_corrections=False, the byproduct X/Z corrections
        inside every internal gate_sqrt_X and gate_CNOT call are suppressed.
        """
        cls = MBQCTranslatedGates

        qc.rz(Z1, control)
        qc.rz(Z2, target)

        blocks, cx_dirs = cls._zsx_blocks_and_cx(theta, phi)
        idx = start_idx

        for layer in range(4):
            block = blocks[layer]

            for q_local in (0, 1):
                wire = control if q_local == 0 else target
                for name, angle in block[q_local]:
                    if name == 'rz':
                        cls._apply_rz_or_s(qc, wire, angle)
                    elif name == 'sx':
                        cls.gate_sqrt_X(
                            qc, ancilla, wire, creg, start_idx=idx,
                            apply_pauli_corrections=apply_pauli_corrections,
                        )
                        idx += cls.N_BITS['sqrt_X']
                    elif name == 'x':
                        cls.gate_sqrt_X(
                            qc, ancilla, wire, creg, start_idx=idx,
                            apply_pauli_corrections=apply_pauli_corrections,
                        )
                        idx += cls.N_BITS['sqrt_X']
                        cls.gate_sqrt_X(
                            qc, ancilla, wire, creg, start_idx=idx,
                            apply_pauli_corrections=apply_pauli_corrections,
                        )
                        idx += cls.N_BITS['sqrt_X']

            # CX between the data qubits (with the correct direction)
            if layer < 3:
                ctrl_idx, _ = cx_dirs[layer]
                if ctrl_idx == 0:
                    cls.gate_CNOT(
                        qc, control, ancilla, target, creg, start_idx=idx,
                        apply_pauli_corrections=apply_pauli_corrections,
                    )
                else:
                    cls.gate_CNOT(
                        qc, target, ancilla, control, creg, start_idx=idx,
                        apply_pauli_corrections=apply_pauli_corrections,
                    )
                idx += cls.CNOT_N_BITS

        qc.rz(Z3, control)
        qc.rz(Z4, target)

        expected = cls.n_bits_two_qubit(theta, phi)
        assert idx - start_idx == expected, (
            f"two-qubit gate bit-count mismatch: used {idx - start_idx}, "
            f"expected {expected}"
        )
        return qc
