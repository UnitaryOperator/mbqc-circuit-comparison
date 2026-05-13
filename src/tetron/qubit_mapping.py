# ============================================================
# Google 12-qubit Cirq-grid index -> tetron logical mapping
# ============================================================
#
# Physical device: 8-row × 3-column tetron array (T1 … T24)
# Center-out indexing (T1 at center, spiraling outward):
#
#   T21  T16  T17      anc   8   anc
#   T15  T10  T11       11  anc    7
#   T9   T2   T3       anc   1   anc
#   T8   T1   T4         4  anc    5
#   T7   T6   T5       anc   2   anc
#   T14  T13  T12        3  anc    6
#   T20  T19  T18      anc  10   anc
#   T24  T23  T22        9  anc   12
#
# Qiskit 0-based index: q[i] = T_{i+1},  so T_k -> q[k-1]
#
# Google circuit uses cirq.GridQubit(row, col)
# Active patch:
#   (3,3) (3,4) (3,5) (3,6)
#   (4,3) (4,4) (4,5) (4,6)
#   (5,3) (5,4) (5,5) (5,6)
#
# Cirq grid -> tetron logical label:
#   11   4   3   9
#    8   1   2  10
#    7   5   6  12

GRID_TO_LOGICAL = {
    (3, 3): 11,
    (3, 4):  4,
    (3, 5):  3,
    (3, 6):  9,

    (4, 3):  8,
    (4, 4):  1,
    (4, 5):  2,
    (4, 6): 10,

    (5, 3):  7,
    (5, 4):  5,
    (5, 5):  6,
    (5, 6): 12,
}

# ============================================================
# Logical label -> physical tetron site T_i  (1-based)
# ============================================================
# Only logical (data) qubits are listed here.
# Qiskit index = site - 1.

LOGICAL_TO_TETRON_SITE = {
     8: 16,
    11: 15,
     7: 11,
     1:  2,
     4:  8,
     5:  4,
     2:  6,
     3: 14,
     6: 12,
    10: 19,
     9: 24,
    12: 22,
}

# ============================================================
# Logical label -> ancilla site for SINGLE-QUBIT gates
# ============================================================
# Each logical qubit uses the ancilla directly below it in the
# 8×3 grid (same column, one row down).  For the bottom-row
# qubits L9 and L12, which have no ancilla below, T23 (row7,
# col1) is the nearest unoccupied ancilla in the same row.
#
#   L8  (T16, r0,c1) -> T10 (r1,c1)
#   L11 (T15, r1,c0) -> T9  (r2,c0)
#   L7  (T11, r1,c2) -> T3  (r2,c2)
#   L1  (T2,  r2,c1) -> T1  (r3,c1)
#   L4  (T8,  r3,c0) -> T7  (r4,c0)
#   L5  (T4,  r3,c2) -> T5  (r4,c2)
#   L2  (T6,  r4,c1) -> T13 (r5,c1)
#   L3  (T14, r5,c0) -> T20 (r6,c0)
#   L6  (T12, r5,c2) -> T18 (r6,c2)
#   L10 (T19, r6,c1) -> T23 (r7,c1)
#   L9  (T24, r7,c0) -> T23 (r7,c1)  [same row, no below]
#   L12 (T22, r7,c2) -> T23 (r7,c1)  [same row, no below]

LOGICAL_TO_SQ_ANCILLA_SITE = {
     8: 10,
    11:  9,
     7:  3,
     1:  1,
     4:  7,
     5:  5,
     2: 13,
     3: 20,
     6: 18,
    10: 23,
     9: 23,
    12: 23,
}

# ============================================================
# Logical edge -> ancilla site for TWO-QUBIT gates
# ============================================================
# The ancilla site is the tetron physically between the two
# coupled logical qubits in the 8×3 grid.
# Keys are sorted tuples so (a,b) and (b,a) are the same edge.

EDGE_TO_ANCILLA_SITE = {
    # H-layer / left-diagonal: use upper-left outer ancilla
    tuple(sorted(( 8, 11))): 21,
    tuple(sorted(( 1,  4))):  9,
    tuple(sorted(( 2,  3))):  7,
    tuple(sorted(( 9, 10))): 20,

    # G-layer / right-diagonal: use upper-right outer ancilla
    tuple(sorted(( 7,  8))): 17,
    tuple(sorted(( 1,  5))):  3,
    tuple(sorted(( 2,  6))):  5,
    tuple(sorted((10, 12))): 18,

    # F-layer / vertical same-column: middle ancilla between the pair
    tuple(sorted((11,  4))):  9,
    tuple(sorted(( 3,  9))): 20,
    tuple(sorted(( 8,  1))): 10,
    tuple(sorted(( 2, 10))): 13,
    tuple(sorted(( 7,  5))):  3,
    tuple(sorted(( 6, 12))): 18,

    # E-layer / vertical type-2: middle ancilla
    tuple(sorted(( 4,  3))):  7,
    tuple(sorted(( 1,  2))):  1,
    tuple(sorted(( 5,  6))):  5,
}

# ============================================================
# Helper: T_i (1-based) -> Qiskit 0-based index
# ============================================================

def tetron_site_to_qiskit_index(site_1based: int) -> int:
    """T_i notation -> Qiskit 0-based index q[i-1]."""
    return site_1based - 1


# ============================================================
# Grid / logical -> Qiskit data-qubit index
# ============================================================

def grid_to_logical(row: int, col: int) -> int:
    return GRID_TO_LOGICAL[(row, col)]


def grid_to_tetron_site(row: int, col: int) -> int:
    logical = grid_to_logical(row, col)
    return LOGICAL_TO_TETRON_SITE[logical]


def grid_to_qiskit_index(row: int, col: int) -> int:
    """Return the Qiskit wire index of the DATA qubit at Cirq (row, col)."""
    return tetron_site_to_qiskit_index(grid_to_tetron_site(row, col))


# ============================================================
# Single-qubit ancilla lookup
# ============================================================

def logical_to_sq_ancilla_index(logical_label: int) -> int:
    """
    Return the Qiskit wire index of the ancilla used for single-qubit
    gates on the given logical qubit label.

    The ancilla is the tetron directly below the logical qubit in the
    8×3 physical layout (or same-row neighbour for bottom-row qubits).

    Usage with TetronSingleQubitGates (note: ancilla arg comes first):
        q_data = logical_to_qiskit_index(label)
        q_anc  = logical_to_sq_ancilla_index(label)
        TetronSingleQubitGates.gate_sqrt_X(qc, q_anc, q_data, creg)
    """
    site = LOGICAL_TO_SQ_ANCILLA_SITE[logical_label]
    return tetron_site_to_qiskit_index(site)


def grid_to_sq_ancilla_index(row: int, col: int) -> int:
    """
    Return the Qiskit wire index of the single-qubit ancilla for the
    logical qubit at Cirq grid position (row, col).
    """
    logical = grid_to_logical(row, col)
    return logical_to_sq_ancilla_index(logical)


# ============================================================
# Two-qubit ancilla lookup
# ============================================================

def logical_edge_to_ancilla_site(q1_logical: int, q2_logical: int) -> int:
    edge = tuple(sorted((q1_logical, q2_logical)))
    if edge not in EDGE_TO_ANCILLA_SITE:
        raise ValueError(
            f"Logical edge {edge} is not a valid EFGH coupling "
            f"in the 8×3 tetron layout."
        )
    return EDGE_TO_ANCILLA_SITE[edge]


def logical_edge_to_qiskit_indices(q1_logical: int, q2_logical: int):
    """
    Return (data_qubit_1, data_qubit_2, ancilla_qubit) as Qiskit
    0-based indices for a two-qubit gate between two logical labels.

    Usage with add_TwoQubitGate:
        d1, d2, anc = logical_edge_to_qiskit_indices(8, 11)
        add_TwoQubitGate(qc, d1, anc, d2, theta, phi, 0,0,0,0, cbit)
    """
    site1    = LOGICAL_TO_TETRON_SITE[q1_logical]
    site2    = LOGICAL_TO_TETRON_SITE[q2_logical]
    anc_site = logical_edge_to_ancilla_site(q1_logical, q2_logical)
    return (
        tetron_site_to_qiskit_index(site1),
        tetron_site_to_qiskit_index(site2),
        tetron_site_to_qiskit_index(anc_site),
    )


def grid_edge_to_qiskit_indices(q1_grid, q2_grid):
    """
    Return (data_qubit_1, data_qubit_2, ancilla_qubit) as Qiskit
    0-based indices for a two-qubit gate between two Cirq GridQubits.

    q1_grid, q2_grid: tuples like (3, 4), (3, 5).

    Example:
        grid_edge_to_qiskit_indices((3,4), (3,5))  # E-layer edge (4,3)
        # -> (7, 13, 6)  i.e. T8, T14, ancilla T7
    """
    q1_logical = GRID_TO_LOGICAL[q1_grid]
    q2_logical = GRID_TO_LOGICAL[q2_grid]
    return logical_edge_to_qiskit_indices(q1_logical, q2_logical)
