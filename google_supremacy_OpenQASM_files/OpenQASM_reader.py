import cirq
import numpy as np
from circuit_n12_m14_s0_e0_pEFGH import CIRCUIT

TOL = 0.05  # tolerance for snapping


def snap_fsim(gate):
    target_theta = np.pi / 2
    target_phi = np.pi / 6

    if (abs(gate.theta - target_theta) < TOL and
        abs(gate.phi - target_phi) < TOL):
        return cirq.FSimGate(theta=target_theta, phi=target_phi)

    return gate


def snap_single_qubit(gate):
    # XPow
    if isinstance(gate, cirq.XPowGate):
        if abs(gate.exponent - 0.5) < TOL:
            return cirq.XPowGate(exponent=0.5)
        return gate

    # YPow
    if isinstance(gate, cirq.YPowGate):
        if abs(gate.exponent - 0.5) < TOL:
            return cirq.YPowGate(exponent=0.5)
        return gate

    # PhasedX (W)
    if isinstance(gate, cirq.PhasedXPowGate):
        if (abs(gate.exponent - 0.5) < TOL and
            abs(gate.phase_exponent - 0.25) < TOL):
            return cirq.PhasedXPowGate(
                phase_exponent=0.25,
                exponent=0.5
            )
        return gate

    return gate


def snap_gate(gate):
    if isinstance(gate, cirq.FSimGate):
        return snap_fsim(gate)
    else:
        return snap_single_qubit(gate)


def snap_circuit(circuit):
    new_moments = []

    for moment in circuit:
        new_ops = []
        for op in moment.operations:
            new_gate = snap_gate(op.gate)
            new_ops.append(new_gate.on(*op.qubits))
        new_moments.append(cirq.Moment(new_ops))

    return cirq.Circuit(new_moments)


# ---- APPLY ----
snapped_circuit = snap_circuit(CIRCUIT)

print("\n=== ORIGINAL ===")
print(CIRCUIT[:5])

print("\n=== SNAPPED ===")
print(snapped_circuit[:5])