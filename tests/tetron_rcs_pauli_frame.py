"""
Tetron-based Random Circuit Sampling (RCS) with Pauli Frame Tracking
=====================================================================

Architecture: 12x2 tetrons (12 logical + 12 ancilla)
Gate set: Random single-qubit gates from {sqrt(X), sqrt(Y), sqrt(W)}
Paradigm: Measurement-based braiding on tetron qubits

pip install qiskit qiskit-aer

Reference:
  - Microsoft Quantum, arXiv:2502.12252v3 (2025), Sec. III & App. E
  - Google, Nature 574, 505-510 (2019), Fig. 3
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from dataclasses import dataclass, field
from collections import Counter


# ============================================================================
# 1. Gate set: {sqrt(X), sqrt(Y), sqrt(W)} for RCS
# ============================================================================

SQRT_X = Operator(np.array([
    [(1+1j)/2, (1-1j)/2],
    [(1-1j)/2, (1+1j)/2]
], dtype=complex))

SQRT_Y = Operator(np.array([
    [ np.cos(np.pi/4), -np.sin(np.pi/4)],
    [ np.sin(np.pi/4),  np.cos(np.pi/4)]
], dtype=complex))

# W = (X + Y) / sqrt(2), then sqrt via eigendecomposition
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
W = (X + Y) / np.sqrt(2)
eigvals, eigvecs = np.linalg.eigh(W)
SQRT_W = Operator(eigvecs @ np.diag(np.sqrt(eigvals.astype(complex))) @ eigvecs.conj().T)

GATES = {'sX': SQRT_X, 'sY': SQRT_Y, 'sW': SQRT_W}


# ============================================================================
# 2. Pauli Frame Tracker
# ============================================================================

@dataclass
class PauliFrame:
    """
    Tracks accumulated Pauli byproducts from measurement-based braiding.

    In tetron MBQC, each gate produces random Pauli byproducts depending on
    parity measurement outcomes. These are tracked classically (not physically
    corrected) and applied as bit-flips to the final readout.

    Per-qubit: (x_bit, z_bit) -> I(0,0), X(1,0), Z(0,1), Y(1,1)
    """
    n: int
    xf: np.ndarray = field(default=None)
    zf: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.xf is None: self.xf = np.zeros(self.n, dtype=int)
        if self.zf is None: self.zf = np.zeros(self.n, dtype=int)

    def label(self, q):
        return {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}[(self.xf[q], self.zf[q])]

    def apply_byproduct(self, q, p):
        self.xf[q] ^= (1 if p in ('X','Y') else 0)
        self.zf[q] ^= (1 if p in ('Z','Y') else 0)

    def conjugate_clifford(self, q, c):
        """Update frame under C P C_dag: H swaps X<->Z; S maps X->Y, Y->-X."""
        x, z = self.xf[q], self.zf[q]
        if c == 'H':
            self.xf[q], self.zf[q] = z, x
        elif c == 'S':
            if (x,z) == (1,0):   self.zf[q] = 1   # X -> Y
            elif (x,z) == (1,1): self.zf[q] = 0   # Y -> -X

    def cz_propagate(self, q1, q2):
        """CZ: X on q1 spawns Z on q2 and vice versa."""
        x1, x2 = self.xf[q1], self.xf[q2]
        if x1: self.apply_byproduct(q2, 'Z')
        if x2: self.apply_byproduct(q1, 'Z')

    def frame_string(self):
        return ''.join(self.label(q) for q in range(self.n))

    def correction_mask(self):
        """Bitmask: qubits with X or Y need bit-flip at Z-basis readout."""
        mask = 0
        for q in range(self.n):
            if self.label(q) in ('X','Y'):
                mask |= (1 << q)  # qiskit bit ordering: q0 = LSB
        return mask


# ============================================================================
# 3. Measurement-Based Braiding Simulation
# ============================================================================

def braiding_gate(gate_key, q, frame, rng):
    """
    Simulate measurement-based implementation of a gate on a (comp, ancilla)
    tetron pair.

    From arXiv:2502.12252v3 Eq.(E1)-(E2):
      [S] via M^{XI} M^{YI} M^{ZZ} M^{XI}: byproduct = Z^{(1+s0*s1*s2)/2}
      [H] via M^{XI} M^{YI} M^{ZY} M^{XI}: byproduct = Y^{(1+s0*s1*s2)/2} * X

    Decompositions:
      sqrt(X) ~ H*S       -> 2 steps,  8 parity measurements
      sqrt(Y) ~ S*H*S     -> 3 steps, 12 parity measurements
      sqrt(W) ~ H*S*H*S   -> 4 steps, 16 parity measurements
    """
    decomp = {
        'sX': ['S', 'H'],
        'sY': ['S', 'H', 'S'],
        'sW': ['S', 'H', 'S', 'H'],
    }

    steps = []
    for cliff in decomp[gate_key]:
        # 4 parity measurements on (comp_tetron, ancilla_tetron)
        s = [rng.choice([-1, 1]) for _ in range(4)]
        exp = (1 + s[0]*s[1]*s[2]) // 2

        if cliff == 'S':
            byp = 'Z' if exp % 2 == 1 else 'I'
            mseq = 'XI->YI->ZZ->XI'
        else:
            byp = 'Y' if exp % 2 == 1 else 'X'
            mseq = 'XI->YI->ZY->XI'

        if byp != 'I':
            frame.apply_byproduct(q, byp)
        frame.conjugate_clifford(q, cliff)

        steps.append({'cliff': cliff, 'meas': mseq, 'outcomes': s,
                      'byproduct': byp, 'frame': frame.label(q)})

    return {'gate': gate_key, 'qubit': q, 'steps': steps, 'final': frame.label(q)}


# ============================================================================
# 4. Build Qiskit RCS Circuit + Track Pauli Frame
# ============================================================================

def build_rcs(n_qubits=12, n_cycles=8, seed=42):
    """
    Build a Qiskit RCS circuit AND simultaneously track the Pauli frame
    that a real tetron device would accumulate.

    Returns: (qiskit_circuit, pauli_frame, measurement_records)
    """
    rng = np.random.default_rng(seed)

    qc = QuantumCircuit(n_qubits, n_qubits)
    frame = PauliFrame(n_qubits)

    gate_keys = list(GATES.keys())
    prev = [None] * n_qubits
    all_recs = []
    frame_hist = []
    gate_hist = []

    for cyc in range(n_cycles):
        cyc_recs = []
        cyc_gates = []

        # --- Single-qubit layer: random {sqrt(X), sqrt(Y), sqrt(W)} ---
        for q in range(n_qubits):
            avail = [g for g in gate_keys if g != prev[q]]
            chosen = rng.choice(avail)
            prev[q] = chosen
            cyc_gates.append(chosen)

            # Ideal gate on Qiskit circuit
            qc.unitary(GATES[chosen], [q], label=chosen)

            # Measurement-based braiding on tetron (Pauli frame tracking)
            rec = braiding_gate(chosen, q, frame, rng)
            cyc_recs.append(rec)

        qc.barrier()

        # --- Two-qubit layer: CZ in alternating tiling ---
        for i in range(cyc % 2, n_qubits - 1, 2):
            qc.cz(i, i + 1)
            frame.cz_propagate(i, i + 1)

        qc.barrier()

        all_recs.append(cyc_recs)
        frame_hist.append(frame.frame_string())
        gate_hist.append(cyc_gates)

    qc.measure(range(n_qubits), range(n_qubits))

    return qc, frame, all_recs, frame_hist, gate_hist


# ============================================================================
# 5. Run & Display
# ============================================================================

def main():
    N = 12
    CYCLES = 8
    SHOTS = 2048
    SEED = 42

    print("=" * 80)
    print("  TETRON RCS: MEASUREMENT-BASED RANDOM CIRCUIT SAMPLING")
    print("  12 Logical Tetrons + 12 Ancilla Tetrons = 24 Total")
    print("=" * 80)
    print(f"\n  Gate set:  {{sqrt(X), sqrt(Y), sqrt(W)}}  (no sequential repeats)")
    print(f"  Two-qubit: CZ (alternating even/odd nearest-neighbor)")
    print(f"  Cycles: {CYCLES}   Shots: {SHOTS}   Seed: {SEED}\n")

    # Build circuit + frame
    qc, frame, all_recs, frame_hist, gate_hist = build_rcs(N, CYCLES, SEED)

    # ---- Gate selections ----
    print("-" * 80)
    print("  GATE SELECTIONS PER CYCLE")
    print("-" * 80)
    print("  Cycle " + " ".join(f"q{q:02d}" for q in range(N)))
    for c, gates in enumerate(gate_hist):
        print(f"    {c:3d}   " + "  ".join(f"{g:>2s}" for g in gates))

    # ---- Frame evolution ----
    print(f"\n{'-'*80}")
    print("  PAULI FRAME EVOLUTION (byproducts from measurement outcomes)")
    print("-" * 80)
    print(f"  Cycle  Frame (q00..q11)     Non-I")
    for c, fs in enumerate(frame_hist):
        ni = sum(1 for ch in fs if ch != 'I')
        print(f"    {c:3d}   {'  '.join(fs)}       {ni:2d}/12")

    # ---- Braiding detail, cycle 0 ----
    print(f"\n{'-'*80}")
    print("  MEASUREMENT-BASED BRAIDING DETAIL (Cycle 0, first 4 qubits)")
    print("-" * 80)
    print("  Each gate = parity measurements on (comp, ancilla) tetron pair")
    print("  Outcomes s_i in {+1,-1} determine Pauli byproduct\n")

    for rec in all_recs[0][:4]:
        q = rec['qubit']
        print(f"  q{q:02d}: {rec['gate']}  "
              f"(decomp: {' -> '.join(s['cliff'] for s in rec['steps'])})")
        for step in rec['steps']:
            o = step['outcomes']
            print(f"    [{step['cliff']}] Meas: {step['meas']}")
            print(f"         s=[{o[0]:+d},{o[1]:+d},{o[2]:+d},{o[3]:+d}]"
                  f"  byproduct={step['byproduct']}  frame={step['frame']}")
        print(f"    >> Accumulated frame: {rec['final']}\n")
    print("  ... (q04-q11 omitted)\n")

    # ---- XY frame analysis ----
    print(f"{'-'*80}")
    print("  XY FRAME ANALYSIS (correction for Z-basis readout)")
    print("-" * 80)
    fs = frame.frame_string()
    xq = [q for q in range(N) if frame.label(q) == 'X']
    yq = [q for q in range(N) if frame.label(q) == 'Y']
    zq = [q for q in range(N) if frame.label(q) == 'Z']
    iq = [q for q in range(N) if frame.label(q) == 'I']
    flip_q = sorted(xq + yq)
    mask = frame.correction_mask()

    print(f"  Final frame:  {fs}")
    print(f"  I (no correction):             {['q'+str(q).zfill(2) for q in iq]}")
    print(f"  X (bit-flip at readout):       {['q'+str(q).zfill(2) for q in xq]}")
    print(f"  Z (phase only, no Z-meas fix): {['q'+str(q).zfill(2) for q in zq]}")
    print(f"  Y (bit-flip + phase):          {['q'+str(q).zfill(2) for q in yq]}")
    print(f"\n  Qubits needing bit-flip:  {['q'+str(q).zfill(2) for q in flip_q]}")
    print(f"  Correction bitmask:       {format(mask, f'0{N}b')}")

    # ---- Simulate ----
    print(f"\n{'-'*80}")
    print("  QISKIT SIMULATION (AerSimulator)")
    print("-" * 80)

    sim = AerSimulator()
    result = sim.run(qc, shots=SHOTS).result()
    raw_counts = result.get_counts()

    # Apply frame correction
    corrected = Counter()
    for bs, cnt in raw_counts.items():
        # qiskit bitstring: "q11 q10 ... q01 q00" (MSB=q11, LSB=q00)
        bits = int(bs, 2)
        corrected[format(bits ^ mask, f'0{N}b')] += cnt

    print(f"\n  {'#':>3s}  {'Raw':>14s} {'Cnt':>5s}  |  {'Corrected':>14s} {'Cnt':>5s}")
    raw_top = sorted(raw_counts.items(), key=lambda x: -x[1])[:10]
    cor_top = sorted(corrected.items(), key=lambda x: -x[1])[:10]
    for i in range(10):
        rb, rc = raw_top[i] if i < len(raw_top) else ('', 0)
        cb, cc = cor_top[i] if i < len(cor_top) else ('', 0)
        print(f"  {i+1:3d}  {rb:>14s} {rc:5d}  |  {cb:>14s} {cc:5d}")

    print(f"\n  Unique raw bitstrings:       {len(raw_counts)}")
    print(f"  Unique corrected bitstrings: {len(corrected)}")

    # ---- Summary ----
    tot_meas = sum(len(r['steps'])*4 for cr in all_recs for r in cr)
    tot_cz = sum(len(range(c%2, N-1, 2)) for c in range(CYCLES))

    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  Logical tetrons:            12")
    print(f"  Ancilla tetrons:            12")
    print(f"  Total tetrons:              24")
    print(f"  RCS cycles:                 {CYCLES}")
    print(f"  Total 1-qubit gates:        {N * CYCLES}")
    print(f"  Total CZ gates:             {tot_cz}")
    print(f"  Total parity measurements:  {tot_meas}")
    print(f"  Frame corrections needed:   {len(flip_q)}/{N} qubits")
    print(f"  Circuit depth:              {qc.depth()}")
    print()
    print("  The Pauli byproducts from measurement-based braiding are NEVER")
    print("  physically corrected. They accumulate in the classical Pauli frame.")
    print("  At Z-basis readout, X/Y frame entries flip the bit; Z entries don't.")
    print()


if __name__ == '__main__':
    main()
