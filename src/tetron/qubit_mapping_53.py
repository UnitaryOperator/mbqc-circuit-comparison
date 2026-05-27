"""
qubit_mapping_53.py

Mapping utilities for translating Google Sycamore 53-qubit Cirq-grid
circuits into the 53-data + 53-ancilla tetron layout.

This corrected version swaps the lower-right logical labels 41 and 47
relative to the first draft so that the Google edge (47, 51) has a local
tetron ancilla connection.

Conventions
-----------
* Google circuit qubits are addressed as ``cirq.GridQubit(row, col)``.
* ``GRID_TO_LOGICAL`` maps those grid coordinates to the Google logical
  qubit labels shown in the uploaded grid-index figure.
* ``LOGICAL_TO_TETRON_SITE`` maps logical labels 1..53 to tetron sites
  T1..T106 in the code-index figure.
* Qiskit wire index convention is unchanged from ``qubit_mapping.py``:
  ``q[i] = T_{i+1}``; equivalently, tetron site ``T_k`` is Qiskit wire
  ``k - 1``.
* The file only provides wire/ancilla lookup tables. The circuit translator
  can decide to keep only Clifford single-qubit gates ``I``, ``sqrt_X``,
  ``sqrt_Y`` and the Clifford ``fSim(theta=pi/2, phi=0)`` approximation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

Grid = Tuple[int, int]
LogicalEdge = Tuple[int, int]
TetronCoord = Tuple[int, int]

DEFAULT_FSIM_THETA = np.pi / 2
DEFAULT_FSIM_PHI = 0.0

# ============================================================
# Google 53-qubit Cirq-grid index -> Google logical label
# ============================================================
# Top-left grid cell is (0, 0). Only occupied Sycamore cells are listed.

GRID_TO_LOGICAL: Dict[Grid, int] = {
    (0, 5): 38,
    (0, 6): 39,

    (1, 4): 36,
    (1, 5): 27,
    (1, 6): 28,
    (1, 7): 30,

    (2, 4): 14,
    (2, 5): 13,
    (2, 6): 17,
    (2, 7): 23,
    (2, 8): 33,

    (3, 2): 35,
    (3, 3): 11,
    (3, 4):  4,
    (3, 5):  3,
    (3, 6):  9,
    (3, 7): 19,
    (3, 8): 34,
    (3, 9): 45,

    (4, 1): 37,
    (4, 2): 22,
    (4, 3):  8,
    (4, 4):  1,
    (4, 5):  2,
    (4, 6): 10,
    (4, 7): 20,
    (4, 8): 43,
    (4, 9): 49,

    (5, 0): 52,
    (5, 1): 32,
    (5, 2): 21,
    (5, 3):  7,
    (5, 4):  5,
    (5, 5):  6,
    (5, 6): 12,
    (5, 7): 41,
    (5, 8): 50,

    (6, 1): 31,
    (6, 2): 24,
    (6, 3): 18,
    (6, 4): 15,
    (6, 5): 16,
    (6, 6): 51,
    (6, 7): 47,

    (7, 2): 29,
    (7, 3): 26,
    (7, 4): 25,
    (7, 5): 42,
    (7, 6): 46,

    (8, 3): 40,
    (8, 4): 44,
    (8, 5): 48,

    (9, 4): 53,
}

LOGICAL_TO_GRID: Dict[int, Grid] = {v: k for k, v in GRID_TO_LOGICAL.items()}

# ============================================================
# Tetron layout coordinate -> (T_site, logical_label_or_None)
# ============================================================
# None means this tetron is used as an ancilla site.

TETRON_LAYOUT: Dict[TetronCoord, Tuple[int, int | None]] = {
    ( 0,  4): (100,   52),

    ( 1,  3): ( 84,   37),
    ( 1,  4): ( 85, None),

    ( 2,  3): ( 60, None),
    ( 2,  4): ( 61,   32),
    ( 2,  5): ( 62, None),
    ( 2,  6): ( 63,   31),

    ( 3,  2): ( 51, None),
    ( 3,  3): ( 36,   22),
    ( 3,  4): ( 37, None),
    ( 3,  5): ( 38,   21),
    ( 3,  6): ( 39, None),
    ( 3,  7): ( 64,   29),

    ( 4,  2): ( 50,   35),
    ( 4,  3): ( 21, None),
    ( 4,  4): ( 16,    8),
    ( 4,  5): ( 17, None),
    ( 4,  6): ( 33,   24),
    ( 4,  7): ( 65, None),
    ( 4,  8): ( 86,   40),

    ( 5,  2): ( 49, None),
    ( 5,  3): ( 15,   11),
    ( 5,  4): ( 10, None),
    ( 5,  5): ( 11,    7),
    ( 5,  6): ( 34, None),
    ( 5,  7): ( 66,   26),
    ( 5,  8): ( 87, None),

    ( 6,  3): (  9, None),
    ( 6,  4): (  2,    1),
    ( 6,  5): (  3, None),
    ( 6,  6): ( 35,   18),
    ( 6,  7): ( 67, None),
    ( 6,  8): ( 88,   44),
    ( 6,  9): (101, None),

    ( 7,  0): ( 83, None),
    ( 7,  1): ( 59,   36),
    ( 7,  2): ( 28, None),
    ( 7,  3): (  8,    4),
    ( 7,  4): (  1, None),
    ( 7,  5): (  4,    5),
    ( 7,  6): ( 29, None),
    ( 7,  7): ( 68,   25),
    ( 7,  8): ( 89, None),
    ( 7,  9): (102,   53),

    ( 8,  0): ( 82,   38),
    ( 8,  1): ( 58, None),
    ( 8,  2): ( 27,   14),
    ( 8,  3): (  7, None),
    ( 8,  4): (  6,    2),
    ( 8,  5): (  5, None),
    ( 8,  6): ( 30,   15),
    ( 8,  7): ( 69, None),
    ( 8,  8): ( 90,   48),
    ( 8,  9): (103, None),

    ( 9,  0): ( 81, None),
    ( 9,  1): ( 57,   27),
    ( 9,  2): ( 26, None),
    ( 9,  3): ( 14,    3),
    ( 9,  4): ( 13, None),
    ( 9,  5): ( 12,    6),
    ( 9,  6): ( 31, None),
    ( 9,  7): ( 70,   42),
    ( 9,  8): ( 91, None),

    (10,  0): ( 80,   39),
    (10,  1): ( 56, None),
    (10,  2): ( 25,   13),
    (10,  3): ( 20, None),
    (10,  4): ( 19,   10),
    (10,  5): ( 18, None),
    (10,  6): ( 32,   16),
    (10,  7): ( 71, None),
    (10,  8): ( 92,   46),

    (11,  0): ( 79, None),
    (11,  1): ( 55,   28),
    (11,  2): ( 40, None),
    (11,  3): ( 24,    9),
    (11,  4): ( 23, None),
    (11,  5): ( 22,   12),
    (11,  6): ( 52, None),
    (11,  7): ( 72,   51),
    (11,  8): ( 93, None),

    (12,  0): ( 78,   30),
    (12,  1): ( 54, None),
    (12,  2): ( 41,   17),
    (12,  3): ( 44, None),
    (12,  4): ( 43,   20),
    (12,  5): ( 42, None),
    (12,  6): ( 53,   47),
    (12,  7): ( 73, None),

    (13,  0): ( 77, None),
    (13,  1): ( 48,   23),
    (13,  2): ( 47, None),
    (13,  3): ( 46,   19),
    (13,  4): ( 45, None),
    (13,  5): ( 75,   41),
    (13,  6): ( 74, None),

    (14,  0): ( 76,   33),
    (14,  1): ( 99, None),
    (14,  2): ( 98,   34),
    (14,  3): ( 97, None),
    (14,  4): ( 96,   43),
    (14,  5): ( 95, None),
    (14,  6): ( 94,   50),

    (15,  3): (106,   45),
    (15,  4): (105, None),
    (15,  5): (104,   49),
}

TETRON_COORD_TO_SITE: Dict[TetronCoord, int] = {
    coord: site for coord, (site, _logical) in TETRON_LAYOUT.items()
}

TETRON_SITE_TO_COORD: Dict[int, TetronCoord] = {
    site: coord for coord, site in TETRON_COORD_TO_SITE.items()
}

TETRON_COORD_TO_LOGICAL: Dict[TetronCoord, int | None] = {
    coord: logical for coord, (_site, logical) in TETRON_LAYOUT.items()
}

LOGICAL_TO_TETRON_SITE: Dict[int, int] = {
     1:   2,
     2:   6,
     3:  14,
     4:   8,
     5:   4,
     6:  12,
     7:  11,
     8:  16,
     9:  24,
    10:  19,
    11:  15,
    12:  22,
    13:  25,
    14:  27,
    15:  30,
    16:  32,
    17:  41,
    18:  35,
    19:  46,
    20:  43,
    21:  38,
    22:  36,
    23:  48,
    24:  33,
    25:  68,
    26:  66,
    27:  57,
    28:  55,
    29:  64,
    30:  78,
    31:  63,
    32:  61,
    33:  76,
    34:  98,
    35:  50,
    36:  59,
    37:  84,
    38:  82,
    39:  80,
    40:  86,
    41:  75,
    42:  70,
    43:  96,
    44:  88,
    45: 106,
    46:  92,
    47:  53,
    48:  90,
    49: 104,
    50:  94,
    51:  72,
    52: 100,
    53: 102,
}

LOGICAL_TO_TETRON_COORD: Dict[int, TetronCoord] = {
    logical: TETRON_SITE_TO_COORD[site]
    for logical, site in LOGICAL_TO_TETRON_SITE.items()
}

ANCILLA_TETRON_SITES = frozenset(
    site for coord, (site, logical) in TETRON_LAYOUT.items() if logical is None
)

DATA_TETRON_SITES = frozenset(LOGICAL_TO_TETRON_SITE.values())

# ============================================================
# Logical label -> ancilla site for SINGLE-QUBIT gates
# ============================================================
# Deterministic rule used to construct this table:
# prefer the ancilla directly below the data tetron. If unavailable, use the
# nearest same-row right/left ancilla. This differs from the old 8x3-only file
# for bottom-row central qubits because the 53-qubit layout has extra lower
# ancillas available.

LOGICAL_TO_SQ_ANCILLA_SITE: Dict[int, int] = {
     1:   1,
     2:  13,
     3:  20,
     4:   7,
     5:   5,
     6:  18,
     7:   3,
     8:  10,
     9:  44,
    10:  23,
    11:   9,
    12:  42,
    13:  40,
    14:  26,
    15:  31,
    16:  52,
    17:  47,
    18:  29,
    19:  97,
    20:  45,
    21:  17,
    22:  21,
    23:  99,
    24:  34,
    25:  69,
    26:  67,
    27:  56,
    28:  54,
    29:  65,
    30:  77,
    31:  39,
    32:  37,
    33:  99,
    34:  97,
    35:  49,
    36:  58,
    37:  60,
    38:  81,
    39:  79,
    40:  87,
    41:  95,
    42:  71,
    43: 105,
    44:  89,
    45: 105,
    46:  93,
    47:  74,
    48:  91,
    49: 105,
    50:  95,
    51:  73,
    52:  85,
    53: 103,
}

# ============================================================
# Google hardware edges and logical edge -> ancilla site
# ============================================================
# GOOGLE_GRID_EDGES are the 86 unique fSim couplers appearing in the uploaded
# circuit_n53_m12_s0_e0_pABCDCDAB.py file. EDGE_TO_ANCILLA_SITE is keyed by
# sorted logical-label pairs.

GOOGLE_GRID_EDGES = frozenset({
    ((0, 5), (0, 6)),
    ((0, 5), (1, 5)),
    ((0, 6), (1, 6)),
    ((1, 4), (1, 5)),
    ((1, 4), (2, 4)),
    ((1, 5), (1, 6)),
    ((1, 5), (2, 5)),
    ((1, 6), (1, 7)),
    ((1, 6), (2, 6)),
    ((1, 7), (2, 7)),
    ((2, 4), (2, 5)),
    ((2, 4), (3, 4)),
    ((2, 5), (2, 6)),
    ((2, 5), (3, 5)),
    ((2, 6), (2, 7)),
    ((2, 6), (3, 6)),
    ((2, 7), (2, 8)),
    ((2, 7), (3, 7)),
    ((2, 8), (3, 8)),
    ((3, 2), (3, 3)),
    ((3, 2), (4, 2)),
    ((3, 3), (3, 4)),
    ((3, 3), (4, 3)),
    ((3, 4), (3, 5)),
    ((3, 4), (4, 4)),
    ((3, 5), (3, 6)),
    ((3, 5), (4, 5)),
    ((3, 6), (3, 7)),
    ((3, 6), (4, 6)),
    ((3, 7), (3, 8)),
    ((3, 7), (4, 7)),
    ((3, 8), (3, 9)),
    ((3, 8), (4, 8)),
    ((3, 9), (4, 9)),
    ((4, 1), (4, 2)),
    ((4, 1), (5, 1)),
    ((4, 2), (4, 3)),
    ((4, 2), (5, 2)),
    ((4, 3), (4, 4)),
    ((4, 3), (5, 3)),
    ((4, 4), (4, 5)),
    ((4, 4), (5, 4)),
    ((4, 5), (4, 6)),
    ((4, 5), (5, 5)),
    ((4, 6), (4, 7)),
    ((4, 6), (5, 6)),
    ((4, 7), (4, 8)),
    ((4, 7), (5, 7)),
    ((4, 8), (4, 9)),
    ((4, 8), (5, 8)),
    ((5, 0), (5, 1)),
    ((5, 1), (5, 2)),
    ((5, 1), (6, 1)),
    ((5, 2), (5, 3)),
    ((5, 2), (6, 2)),
    ((5, 3), (5, 4)),
    ((5, 3), (6, 3)),
    ((5, 4), (5, 5)),
    ((5, 4), (6, 4)),
    ((5, 5), (5, 6)),
    ((5, 5), (6, 5)),
    ((5, 6), (5, 7)),
    ((5, 6), (6, 6)),
    ((5, 7), (5, 8)),
    ((5, 7), (6, 7)),
    ((6, 1), (6, 2)),
    ((6, 2), (6, 3)),
    ((6, 2), (7, 2)),
    ((6, 3), (6, 4)),
    ((6, 3), (7, 3)),
    ((6, 4), (6, 5)),
    ((6, 4), (7, 4)),
    ((6, 5), (6, 6)),
    ((6, 5), (7, 5)),
    ((6, 6), (6, 7)),
    ((6, 6), (7, 6)),
    ((7, 2), (7, 3)),
    ((7, 3), (7, 4)),
    ((7, 3), (8, 3)),
    ((7, 4), (7, 5)),
    ((7, 4), (8, 4)),
    ((7, 5), (7, 6)),
    ((7, 5), (8, 5)),
    ((8, 3), (8, 4)),
    ((8, 4), (8, 5)),
    ((8, 4), (9, 4)),
})

# After swapping logical labels 41 and 47 in the lower-right patch, every
# Google hardware edge has a local data-ancilla-data connection.
QUESTIONABLE_EDGE_TO_ANCILLA_SITE: Dict[LogicalEdge, int] = {}

EDGE_TO_ANCILLA_SITE: Dict[LogicalEdge, int] = {
    tuple(sorted(( 1,  2))):   1,
    tuple(sorted(( 1,  4))):   9,
    tuple(sorted(( 1,  5))):   3,
    tuple(sorted(( 1,  8))):  10,
    tuple(sorted(( 2,  3))):   7,
    tuple(sorted(( 2,  6))):   5,
    tuple(sorted(( 2, 10))):  13,
    tuple(sorted(( 3,  4))):   7,
    tuple(sorted(( 3,  9))):  20,
    tuple(sorted(( 3, 13))):  26,
    tuple(sorted(( 4, 11))):   9,
    tuple(sorted(( 4, 14))):  28,
    tuple(sorted(( 5,  6))):   5,
    tuple(sorted(( 5,  7))):   3,
    tuple(sorted(( 5, 15))):  29,
    tuple(sorted(( 6, 12))):  18,
    tuple(sorted(( 6, 16))):  31,
    tuple(sorted(( 7,  8))):  17,
    tuple(sorted(( 7, 18))):  34,
    tuple(sorted(( 7, 21))):  17,
    tuple(sorted(( 8, 11))):  21,
    tuple(sorted(( 8, 22))):  37,
    tuple(sorted(( 9, 10))):  20,
    tuple(sorted(( 9, 17))):  40,
    tuple(sorted(( 9, 19))):  44,
    tuple(sorted((10, 12))):  18,
    tuple(sorted((10, 20))):  23,
    tuple(sorted((11, 35))):  21,
    tuple(sorted((12, 41))):  42,
    tuple(sorted((12, 51))):  52,
    tuple(sorted((13, 14))):  26,
    tuple(sorted((13, 17))):  40,
    tuple(sorted((13, 27))):  26,
    tuple(sorted((14, 36))):  28,
    tuple(sorted((15, 16))):  31,
    tuple(sorted((15, 18))):  29,
    tuple(sorted((15, 25))):  29,
    tuple(sorted((16, 42))):  31,
    tuple(sorted((16, 51))):  71,
    tuple(sorted((17, 23))):  54,
    tuple(sorted((17, 28))):  40,
    tuple(sorted((18, 24))):  34,
    tuple(sorted((18, 26))):  34,
    tuple(sorted((19, 20))):  44,
    tuple(sorted((19, 23))):  47,
    tuple(sorted((19, 34))):  47,
    tuple(sorted((20, 41))):  42,
    tuple(sorted((20, 43))):  45,
    tuple(sorted((21, 22))):  37,
    tuple(sorted((21, 24))):  39,
    tuple(sorted((21, 32))):  62,
    tuple(sorted((22, 35))):  51,
    tuple(sorted((22, 37))):  60,
    tuple(sorted((23, 30))):  54,
    tuple(sorted((23, 33))):  77,
    tuple(sorted((24, 29))):  39,
    tuple(sorted((24, 31))):  39,
    tuple(sorted((25, 26))):  67,
    tuple(sorted((25, 42))):  69,
    tuple(sorted((25, 44))):  67,
    tuple(sorted((26, 29))):  65,
    tuple(sorted((26, 40))):  65,
    tuple(sorted((27, 28))):  56,
    tuple(sorted((27, 36))):  58,
    tuple(sorted((27, 38))):  58,
    tuple(sorted((28, 30))):  79,
    tuple(sorted((28, 39))):  56,
    tuple(sorted((31, 32))):  62,
    tuple(sorted((32, 37))):  85,
    tuple(sorted((32, 52))):  85,
    tuple(sorted((33, 34))):  99,
    tuple(sorted((34, 43))):  97,
    tuple(sorted((34, 45))):  97,
    tuple(sorted((38, 39))):  81,
    tuple(sorted((40, 44))):  87,
    tuple(sorted((41, 47))):  42,
    tuple(sorted((41, 50))):  74,
    tuple(sorted((42, 46))):  91,
    tuple(sorted((42, 48))):  69,
    tuple(sorted((43, 49))):  95,
    tuple(sorted((43, 50))):  95,
    tuple(sorted((44, 48))):  89,
    tuple(sorted((44, 53))): 101,
    tuple(sorted((45, 49))): 105,
    tuple(sorted((46, 51))):  71,
    tuple(sorted((47, 51))):  52,
}

# ============================================================
# Basic helpers: T_i -> Qiskit wire index
# ============================================================

def tetron_site_to_qiskit_index(site_1based: int) -> int:
    """T_i notation -> Qiskit 0-based wire index q[i-1]."""
    if site_1based < 1:
        raise ValueError(f"Tetron site must be 1-based and positive, got {site_1based}")
    return site_1based - 1


def qiskit_index_to_tetron_site(qiskit_index: int) -> int:
    """Qiskit 0-based wire index -> T_i notation."""
    if qiskit_index < 0:
        raise ValueError(f"Qiskit index must be non-negative, got {qiskit_index}")
    return qiskit_index + 1


# ============================================================
# Grid / logical -> data-qubit wire lookup
# ============================================================

def grid_to_logical(row: int, col: int) -> int:
    """Return the Google logical label at Cirq grid coordinate (row, col)."""
    return GRID_TO_LOGICAL[(row, col)]


def logical_to_grid(logical_label: int) -> Grid:
    """Return the Cirq grid coordinate for a Google logical label."""
    return LOGICAL_TO_GRID[logical_label]


def logical_to_tetron_site(logical_label: int) -> int:
    """Return the DATA tetron site T_i for a logical label."""
    return LOGICAL_TO_TETRON_SITE[logical_label]


def grid_to_tetron_site(row: int, col: int) -> int:
    """Return the DATA tetron site T_i for a Cirq grid coordinate."""
    return logical_to_tetron_site(grid_to_logical(row, col))


def logical_to_qiskit_index(logical_label: int) -> int:
    """Return the Qiskit wire index of the DATA qubit for a logical label."""
    return tetron_site_to_qiskit_index(logical_to_tetron_site(logical_label))


def grid_to_qiskit_index(row: int, col: int) -> int:
    """Return the Qiskit wire index of the DATA qubit at Cirq (row, col)."""
    return tetron_site_to_qiskit_index(grid_to_tetron_site(row, col))


# ============================================================
# Single-qubit ancilla lookup
# ============================================================

def logical_to_sq_ancilla_site(logical_label: int) -> int:
    """Return tetron site T_i of the ancilla for single-qubit gates."""
    return LOGICAL_TO_SQ_ANCILLA_SITE[logical_label]


def logical_to_sq_ancilla_index(logical_label: int) -> int:
    """Return Qiskit wire index of the single-qubit ancilla."""
    return tetron_site_to_qiskit_index(logical_to_sq_ancilla_site(logical_label))


def grid_to_sq_ancilla_site(row: int, col: int) -> int:
    """Return tetron site T_i of the single-qubit ancilla for Cirq (row, col)."""
    return logical_to_sq_ancilla_site(grid_to_logical(row, col))


def grid_to_sq_ancilla_index(row: int, col: int) -> int:
    """Return Qiskit wire index of the single-qubit ancilla for Cirq (row, col)."""
    return tetron_site_to_qiskit_index(grid_to_sq_ancilla_site(row, col))


# ============================================================
# Two-qubit ancilla lookup
# ============================================================

def _sorted_edge(q1_logical: int, q2_logical: int) -> LogicalEdge:
    return tuple(sorted((q1_logical, q2_logical)))  # type: ignore[return-value]


def logical_edge_to_ancilla_site(q1_logical: int, q2_logical: int) -> int:
    """Return tetron site T_i of the ancilla for a logical two-qubit edge."""
    edge = _sorted_edge(q1_logical, q2_logical)
    if edge not in EDGE_TO_ANCILLA_SITE:
        raise ValueError(
            f"Logical edge {edge} is not in the 53-qubit Google/tetron edge table."
        )
    return EDGE_TO_ANCILLA_SITE[edge]


def logical_edge_to_qiskit_indices(q1_logical: int, q2_logical: int):
    """
    Return (data_qubit_1, data_qubit_2, ancilla_qubit) as Qiskit 0-based
    wire indices for a two-qubit gate between two logical labels.

    Usage with MBQCTranslatedGates.gate_two_qubit:
        d1, d2, anc = logical_edge_to_qiskit_indices(51, 47)
        MBQCTranslatedGates.gate_two_qubit(
            qc, d1, anc, d2, creg, DEFAULT_FSIM_THETA, DEFAULT_FSIM_PHI,
            start_idx=idx,
        )
    """
    site1 = logical_to_tetron_site(q1_logical)
    site2 = logical_to_tetron_site(q2_logical)
    anc_site = logical_edge_to_ancilla_site(q1_logical, q2_logical)
    return (
        tetron_site_to_qiskit_index(site1),
        tetron_site_to_qiskit_index(site2),
        tetron_site_to_qiskit_index(anc_site),
    )


def grid_edge_to_logical_edge(q1_grid: Grid, q2_grid: Grid) -> LogicalEdge:
    """Translate a Cirq-grid edge into a sorted pair of logical labels."""
    return _sorted_edge(GRID_TO_LOGICAL[q1_grid], GRID_TO_LOGICAL[q2_grid])


def grid_edge_to_qiskit_indices(q1_grid: Grid, q2_grid: Grid):
    """
    Return (data_qubit_1, data_qubit_2, ancilla_qubit) as Qiskit 0-based
    wire indices for a two-qubit gate between two Cirq grid coordinates.
    """
    q1_logical = GRID_TO_LOGICAL[q1_grid]
    q2_logical = GRID_TO_LOGICAL[q2_grid]
    return logical_edge_to_qiskit_indices(q1_logical, q2_logical)


# ============================================================
# Optional validation helpers
# ============================================================

def is_questionable_edge(q1_logical: int, q2_logical: int) -> bool:
    """True for edges whose ancilla assignment should be manually checked."""
    return _sorted_edge(q1_logical, q2_logical) in QUESTIONABLE_EDGE_TO_ANCILLA_SITE


def validate_mapping() -> None:
    """Run internal consistency checks for the 53-qubit mapping tables."""
    if set(GRID_TO_LOGICAL.values()) != set(range(1, 54)):
        raise AssertionError("GRID_TO_LOGICAL must contain logical labels 1..53 exactly once")
    if set(LOGICAL_TO_TETRON_SITE) != set(range(1, 54)):
        raise AssertionError("LOGICAL_TO_TETRON_SITE must contain logical labels 1..53")
    if DATA_TETRON_SITES & ANCILLA_TETRON_SITES:
        raise AssertionError("A tetron site cannot be both data and ancilla")
    if len(DATA_TETRON_SITES) != 53 or len(ANCILLA_TETRON_SITES) != 53:
        raise AssertionError("Expected 53 data tetrons and 53 ancilla tetrons")
    for logical, site in LOGICAL_TO_SQ_ANCILLA_SITE.items():
        if site not in ANCILLA_TETRON_SITES:
            raise AssertionError(f"Single-qubit ancilla for L{logical} is not an ancilla: T{site}")
    if len(GOOGLE_GRID_EDGES) != 86:
        raise AssertionError("Expected 86 Google hardware/circuit edges")
    for edge, site in EDGE_TO_ANCILLA_SITE.items():
        if site not in ANCILLA_TETRON_SITES:
            raise AssertionError(f"Two-qubit ancilla for edge {edge} is not an ancilla: T{site}")


validate_mapping()
