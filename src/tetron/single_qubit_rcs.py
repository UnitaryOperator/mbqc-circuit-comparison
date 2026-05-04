"""
single_qubit_rcs.py

Random single-qubit MBQC circuit sampling on N data/ancilla pairs.

Now with linear XEB (cross-entropy benchmarking) fidelity, consistent with
the standard estimator F_XEB = D <p_U(x)>_exp - 1 used in
[Boixo+ 2018; Arute+ 2019] and the depolarizing-channel framework
rho_U = eps_m |psi_U><psi_U| + (1-eps_m) I/D.

Usage
-----
    from single_qubit_rcs import SingleQubitRCS

    # Clifford mode: gate set {sqrt_X, sqrt_Y}
    rcs = SingleQubitRCS(mode='clifford', seed=42)
    results = rcs.sweep([10, 50, 100, 200])

    # Supremacy mode: gate set {sqrt_X, sqrt_Y, sqrt_W}
    rcs = SingleQubitRCS(mode='supremacy', seed=42)
    results = rcs.sweep([2, 4, 8, 12])
"""
import time
import numpy as np
from collections import defaultdict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from single_qubit_gate import TetronSingleQubitGates


class SingleQubitRCS:
    """
    Random single-qubit MBQC circuit sampling on N disjoint data/ancilla pairs.

    Layout (2N qubits total):
        index:  0   1   2   3   ...  2N-2   2N-1
        role:   d0  a0  d1  a1  ...  d_{N-1} a_{N-1}

    Each ancilla is initialised to |+>. Each data qubit gets a uniformly random
    gate drawn from the gate set corresponding to the chosen mode.

    No-repeat constraint
    --------------------
    Across consecutive layers, the same gate is never applied back-to-back on
    the same data qubit.
    """

    _ALL_GATE_FNS = {
        'H':       (TetronSingleQubitGates.gate_H,      4),
        'S':       (TetronSingleQubitGates.gate_S,      4),
        'HSH':     (TetronSingleQubitGates.gate_HSH,    4),
        'SH':      (TetronSingleQubitGates.gate_SH,     5),
        'HS':      (TetronSingleQubitGates.gate_HS,     5),
        'sqrt_X':  (TetronSingleQubitGates.gate_sqrt_X, 4),
        'sqrt_Y':  (TetronSingleQubitGates.gate_sqrt_Y, 9),
        'sqrt_W':  (TetronSingleQubitGates.gate_sqrt_W, 4),
    }

    CLIFFORD_GATES  = ['sqrt_X', 'sqrt_Y']
    SUPREMACY_GATES = ['sqrt_X', 'sqrt_Y', 'sqrt_W']

    LABELS = {
        'H': '[H]', 'S': '[S]', 'HSH': '[HSH]', 'SH': '[SH]', 'HS': '[HS]',
        'sqrt_X': '\u221aX', 'sqrt_Y': '\u221aY', 'sqrt_W': '\u221aW',
    }

    def __init__(self, mode='clifford', seed=42):
        if mode not in ('clifford', 'supremacy'):
            raise ValueError(f"mode must be 'clifford' or 'supremacy', got {mode!r}")

        self.mode = mode
        self.rng  = np.random.default_rng(seed)

        if mode == 'clifford':
            self.gate_set  = self.CLIFFORD_GATES
            self.simulator = AerSimulator(method='stabilizer')
        else:
            self.gate_set  = self.SUPREMACY_GATES
            self.simulator = AerSimulator()

        self.bits_per_pair = max(self._ALL_GATE_FNS[g][1] for g in self.gate_set)

    # =====================================================================
    # random gate sampling with no-repeat constraint
    # =====================================================================
    def _sample_gates(self, n_pairs, depth):
        gates = [self.rng.choice(self.gate_set, size=n_pairs).tolist()]
        for _ in range(1, depth):
            prev = gates[-1]
            new_layer = [
                str(self.rng.choice([g for g in self.gate_set if g != p]))
                for p in prev
            ]
            gates.append(new_layer)
        return gates

    @staticmethod
    def _validate_no_repeat(gates):
        depth = len(gates)
        if depth < 2:
            return
        n_pairs = len(gates[0])
        for t in range(1, depth):
            for i in range(n_pairs):
                assert gates[t][i] != gates[t - 1][i], (
                    f"no-repeat constraint violated at layer {t}, qubit {i}: "
                    f"gate {gates[t][i]!r} repeats from layer {t - 1}"
                )

    # =====================================================================
    # gate-based equivalents
    # =====================================================================
    def _apply_direct_gate(self, qc, gate_name, qubit):
        if gate_name == 'H':
            qc.h(qubit)
        elif gate_name == 'S':
            qc.s(qubit)
        elif gate_name in ('HSH', 'sqrt_X'):
            qc.sx(qubit)
        elif gate_name == 'SH':
            qc.h(qubit); qc.s(qubit)
        elif gate_name == 'HS':
            qc.s(qubit); qc.h(qubit)
        elif gate_name == 'sqrt_Y':
            qc.s(qubit); qc.s(qubit); qc.h(qubit)
        elif gate_name == 'sqrt_W':
            qc.tdg(qubit); qc.sx(qubit); qc.t(qubit)
        else:
            raise ValueError(f"unknown gate: {gate_name}")

    # =====================================================================
    # circuit builders
    # =====================================================================
    def build_mbqc_verify(self, gate_name, init_state='0'):
        gate_fn, n_int_bits = self._ALL_GATE_FNS[gate_name]
        q     = QuantumRegister(2, 'q')
        c_int = ClassicalRegister(n_int_bits, 'c_int')
        c_out = ClassicalRegister(1, 'c_out')
        qc    = QuantumCircuit(q, c_int, c_out)

        if init_state == '1':
            qc.x(q[1])
        elif init_state == '+':
            qc.h(q[1])

        qc.h(q[0])
        gate_fn(qc, q[0], q[1], c_int, start_idx=0)
        qc.measure(q[1], c_out[0])
        return qc

    def build_direct_verify(self, gate_name, init_state='0'):
        q  = QuantumRegister(1, 'q')
        c  = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q, c)

        if init_state == '1':
            qc.x(q[0])
        elif init_state == '+':
            qc.h(q[0])

        self._apply_direct_gate(qc, gate_name, q[0])
        qc.measure(q[0], c[0])
        return qc

    def build_mbqc_rcs(self, n_pairs, depth=1, gates=None):
        if gates is None:
            gates = self._sample_gates(n_pairs, depth)
        else:
            assert len(gates) == depth
            assert all(len(row) == n_pairs for row in gates)
            self._validate_no_repeat(gates)

        total_int = n_pairs * self.bits_per_pair * depth
        qr     = QuantumRegister(n_pairs * 2, 'q')
        c_int  = ClassicalRegister(total_int, 'c_int')
        c_data = ClassicalRegister(n_pairs,   'c_data')
        qc     = QuantumCircuit(qr, c_int, c_data)

        for layer in range(depth):
            for i, g in enumerate(gates[layer]):
                data_q    = qr[i * 2]
                ancilla_q = qr[i * 2 + 1]
                qc.h(ancilla_q)
                start_idx = (layer * n_pairs + i) * self.bits_per_pair
                self._ALL_GATE_FNS[g][0](qc, ancilla_q, data_q, c_int,
                                         start_idx=start_idx)

        for i in range(n_pairs):
            qc.measure(qr[i * 2], c_data[i])
        return qc, gates

    def build_direct_rcs(self, gates):
        depth   = len(gates)
        n_pairs = len(gates[0])
        qr = QuantumRegister(n_pairs, 'q')
        cr = ClassicalRegister(n_pairs, 'c')
        qc = QuantumCircuit(qr, cr)
        for layer in range(depth):
            for i, g in enumerate(gates[layer]):
                self._apply_direct_gate(qc, g, qr[i])
        qc.measure(qr, cr)
        return qc

    # =====================================================================
    # verification
    # =====================================================================
    def verify_gate(self, gate_name, init_state='0', shots=2048):
        if gate_name not in self.gate_set:
            raise ValueError(f"{gate_name} not in gate set for mode={self.mode!r}")

        mbqc_qc   = self.build_mbqc_verify(gate_name, init_state)
        direct_qc = self.build_direct_verify(gate_name, init_state)

        mbqc_raw = self.simulator.run(
            transpile(mbqc_qc, self.simulator, optimization_level=0),
            shots=shots
        ).result().get_counts()
        direct_raw = self.simulator.run(
            transpile(direct_qc, self.simulator),
            shots=shots
        ).result().get_counts()

        mbqc_out = defaultdict(int)
        for bs, cnt in mbqc_raw.items():
            mbqc_out[bs.split(' ')[0]] += cnt
        mbqc_out = dict(mbqc_out)

        p0_m = mbqc_out.get('0', 0)   / shots
        p0_d = direct_raw.get('0', 0) / shots
        return {
            'gate': gate_name, 'init_state': init_state,
            'p0_mbqc': p0_m, 'p0_direct': p0_d, 'tvd': abs(p0_m - p0_d),
            'mbqc_counts': mbqc_out, 'direct_counts': direct_raw,
        }

    def verify_all(self, init_states=('0', '1', '+'), shots=2048, verbose=True):
        results = []
        if verbose:
            print(f'{"gate":<8} {"init":<6} {"p0_mbqc":>8} {"p0_direct":>10} {"TVD":>8}')
            print('-' * 45)
        for gate in self.gate_set:
            for init in init_states:
                r = self.verify_gate(gate, init, shots)
                results.append(r)
                if verbose:
                    print(f'{self.LABELS[gate]:<8} |{init}>    '
                          f'{r["p0_mbqc"]:>8.3f} {r["p0_direct"]:>10.3f} {r["tvd"]:>8.4f}')
        if verbose:
            max_tvd = max(r['tvd'] for r in results)
            print('-' * 45)
            print(f'Max TVD: {max_tvd:.4f}  (expected ~0 within shot noise)')
        return results

    # =====================================================================
    # ideal probabilities & XEB
    # =====================================================================
    def _ideal_marginals(self, gates):
        """
        Exact per-qubit ideal output probabilities for the direct circuit.

        The RCS circuit is a tensor product of independent single-qubit
        sequences (no entangling gates between data qubits), so the ideal
        joint distribution factorises:

            p_U(x_0 x_1 ... x_{N-1}) = prod_i p_i(x_i)

        We compute each qubit's 2-element marginal exactly via Statevector.
        This keeps the ideal side cheap even for large N in clifford mode.

        Returns
        -------
        marginals : list of np.ndarray, length n_pairs
            marginals[i] = [p_i(0), p_i(1)].
        """
        depth   = len(gates)
        n_pairs = len(gates[0])
        marginals = []
        for i in range(n_pairs):
            qc = QuantumCircuit(1)
            for layer in range(depth):
                self._apply_direct_gate(qc, gates[layer][i], 0)
            sv = Statevector.from_instruction(qc)
            marginals.append(np.abs(sv.data) ** 2)
        return marginals

    def compute_xeb(self, sampled_counts, gates, shots):
        """
        Linear cross-entropy benchmarking fidelity.

            F_XEB = D * <p_U(x)>_{x ~ P_exp} - 1,    D = 2^N

        where p_U(x) = |<x|U|0>|^2 is the *ideal* probability and the
        expectation is over experimentally sampled bitstrings.

        Under the depolarizing-channel framework (Eq. 2),
        F_XEB = eps_m * (D <p_U^2> - 1). For Porter-Thomas D <p_U^2> -> 2
        and F_XEB -> eps_m (-> 1 noiseless). Single-qubit-only circuits do
        not reach Porter-Thomas, so F_XEB tracks (4/3)^N - 1 in supremacy
        mode and is identically 0 in clifford mode. Use TVD as the primary
        diagnostic in clifford mode.
        """
        n_pairs   = len(gates[0])
        D         = 2 ** n_pairs
        marginals = self._ideal_marginals(gates)

        mean_p = 0.0
        for bitstring, cnt in sampled_counts.items():
            p_x = 1.0
            for i in range(n_pairs):
                xi = int(bitstring[n_pairs - 1 - i])  # qiskit: bit -1 is qubit 0
                p_x *= marginals[i][xi]
            mean_p += (cnt / shots) * p_x

        return D * mean_p - 1.0

    # =====================================================================
    # scaling RCS
    # =====================================================================
    @staticmethod
    def _qubit_marginal(counts, qubit_idx, n_bits):
        pos = n_bits - 1 - qubit_idx
        out = defaultdict(int)
        for bs, cnt in counts.items():
            out[bs[pos]] += cnt
        return dict(out)

    def run_rcs(self, n_pairs, depth=1, shots=512):
        """
        Run one scaling point. Returns dict with:
            n, depth, qubits_mbqc, t_mbqc, t_direct, speedup,
            tvd_mean, tvd_max, f_xeb, gates
        """
        qc_mbqc, gates = self.build_mbqc_rcs(n_pairs, depth=depth)
        qc_direct      = self.build_direct_rcs(gates)

        t0 = time.time()
        res_mbqc = self.simulator.run(
            transpile(qc_mbqc, self.simulator, optimization_level=0),
            shots=shots
        ).result()
        t_mbqc = time.time() - t0

        t0 = time.time()
        res_direct = self.simulator.run(
            transpile(qc_direct, self.simulator),
            shots=shots
        ).result()
        t_direct = time.time() - t0

        mbqc_raw   = res_mbqc.get_counts()
        direct_raw = res_direct.get_counts()
        mbqc_data  = defaultdict(int)
        for bs, cnt in mbqc_raw.items():
            mbqc_data[bs.split(' ')[0]] += cnt
        mbqc_data = dict(mbqc_data)

        tvds = []
        for i in range(n_pairs):
            m = self._qubit_marginal(mbqc_data, i, n_pairs)
            d = self._qubit_marginal(direct_raw, i, n_pairs)
            keys = set(m) | set(d)
            tvd  = 0.5 * sum(abs(m.get(k, 0)/shots - d.get(k, 0)/shots) for k in keys)
            tvds.append(tvd)

        f_xeb = self.compute_xeb(mbqc_data, gates, shots)

        return {
            'n': n_pairs,
            'depth': depth,
            'qubits_mbqc': 2 * n_pairs,
            't_mbqc':   t_mbqc,
            't_direct': t_direct,
            'speedup':  t_mbqc / t_direct if t_direct > 0 else float('inf'),
            'tvd_mean': float(np.mean(tvds)),
            'tvd_max':  float(np.max(tvds)),
            'f_xeb':    f_xeb,
            'gates':    gates,
        }

    def sweep(self, n_list, depth=1, shots=512, verbose=True):
        """Run run_rcs over each N in n_list. Returns list of result dicts."""
        results = []
        if verbose:
            print(f'{"N":>5s}  {"depth":>5s}  {"qubits":>7s}  {"t_MBQC":>10s}  '
                  f'{"t_direct":>10s}  {"mean TVD":>9s}  {"max TVD":>8s}  '
                  f'{"F_XEB":>8s}')
            print('-' * 82)
        for n in n_list:
            r = self.run_rcs(n, depth=depth, shots=shots)
            results.append(r)
            if verbose:
                print(f'{r["n"]:>5d}  {r["depth"]:>5d}  {r["qubits_mbqc"]:>7d}  '
                      f'{r["t_mbqc"]:>8.3f}s  {r["t_direct"]:>8.3f}s  '
                      f'{r["tvd_mean"]:>9.4f}  {r["tvd_max"]:>8.4f}  '
                      f'{r["f_xeb"]:>8.4f}')
        return results
