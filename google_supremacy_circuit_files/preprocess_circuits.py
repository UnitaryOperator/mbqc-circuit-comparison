"""
src/circuits/preprocess_circuits.py
-------------------------------------
Preprocess Google 2019 Sycamore n12_m14 circuit .py files into a clean
JSON format suited for tetron MBQC simulation.

What this script does
---------------------
1. Parses every circuit_*.py file in the source directory using regex
   (no cirq dependency required).
2. Maps Cirq GridQubit(r,c) → tetron logical label (1–12) using the
   qubit map established by the team (Week 4 update, page 4).
3. Snaps all FSimGate(θ, φ) to the idealised FSimGate(π/2, π/6).
4. Classifies single-qubit gates:
     (cirq.X ** 0.5)                           → "sqrtX"
     (cirq.Y ** 0.5)                           → "sqrtY"
     PhasedXPowGate(phase_exponent=0.25, ...)  → "sqrtW"
     cirq.Rz(...)                              → DROPPED (virtual frame rotation)
5. Strips moments that become empty after dropping Rz gates and
   re-indexes depth from 0.
6. Outputs one JSON file per source circuit into src/circuits/processed/.

Filename anatomy (Google convention)
--------------------------------------
circuit[_patch]_n12_m14_s{seed}_e{elided}_pEFGH.py

  n12   = 12 qubits
  m14   = 14 two-qubit gate cycles
  s{N}  = random seed (s0-s9 for full circuits, s10-s19 for patch)
  e{N}  = number of FSimGate applications elided from the circuit:
            e0  = full circuit (no elisions) -- use for ideal simulation
            e6  = 6 gates elided (Google's classical cross-check variant)
            e18 = 18 gates elided (patch circuits only, for classical sim)
  patch = circuit split into two 6-qubit halves for Google's verification

Coupling pattern mapping (EFGH -> tetron ABCD)
-----------------------------------------------
Google EFGH labels refer to which subset of Sycamore edges are active
in a given moment. Mapped to tetron logical label pairs per Week 4 PDF:

  E -> tetron pairs: (4,3), (1,2), (5,6)
  F -> tetron pairs: (11,4), (3,9), (8,1), (2,10), (7,5), (6,12)
  G -> tetron pairs: (8,7), (1,5), (2,6), (10,12)
  H -> tetron pairs: (11,8), (4,1), (3,2), (9,10)

Qubit map (Cirq GridQubit -> tetron logical label)
---------------------------------------------------
Source: Week 4 update, page 4 (confirmed by Shuyun).
  (3,3)->11  (3,4)->4   (3,5)->3   (3,6)->9
  (4,3)->8   (4,4)->1   (4,5)->2   (4,6)->10
  (5,3)->7   (5,4)->5   (5,5)->6   (5,6)->12

Usage
-----
  python src/circuits/preprocess_circuits.py \\
      --src  path/to/n12_m14/          \\
      --out  src/circuits/processed/

  # Only process full circuits (e0):
  python src/circuits/preprocess_circuits.py \\
      --src n12_m14/ --out src/circuits/processed/ --elided 0
"""

import re
import json
import math
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THETA_IDEAL = math.pi / 2        # pi/2  ~= 1.5708
PHI_IDEAL   = math.pi / 6        # pi/6  ~= 0.5236

# Cirq GridQubit (row, col) -> tetron logical label (1-12)
# Source: Week 4 team update, page 4.
CIRQ_TO_TETRON: dict = {
    (3, 3): 11,  (3, 4): 4,   (3, 5): 3,   (3, 6): 9,
    (4, 3): 8,   (4, 4): 1,   (4, 5): 2,   (4, 6): 10,
    (5, 3): 7,   (5, 4): 5,   (5, 5): 6,   (5, 6): 12,
}

# EFGH Google pattern -> tetron logical label pairs (A,B,C,D in 8x3 layout)
# Source: Week 4 team update, page 4.
EFGH_TO_TETRON_PAIRS: dict = {
    "E": [(4, 3),  (1, 2),  (5, 6)],
    "F": [(11, 4), (3, 9),  (8, 1), (2, 10), (7, 5), (6, 12)],
    "G": [(8, 7),  (1, 5),  (2, 6), (10, 12)],
    "H": [(11, 8), (4, 1),  (3, 2), (9, 10)],
}

RE_GRIDQUBIT = re.compile(r'cirq\.GridQubit\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_FILENAME  = re.compile(
    r'circuit(_patch)?_n(\d+)_m(\d+)_s(\d+)_e(\d+)_p(\w+)\.py'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_qubits(text: str) -> list:
    """Extract all GridQubit(r,c) pairs from a text snippet."""
    return [(int(r), int(c)) for r, c in RE_GRIDQUBIT.findall(text)]


def gridqubit_to_tetron(q: tuple) -> int:
    return CIRQ_TO_TETRON.get(q)


def extract_balanced(text: str, start: int) -> str:
    """
    From position `start` (which should be just after an opening paren),
    extract content up to the matching closing paren.
    Handles arbitrary nesting depth.
    """
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
        i += 1
    return text[start:i - 1]


def find_on_qubits(text: str, gate_end: int) -> list:
    """
    From gate_end (position after a gate declaration), find the next
    .on(...) call and return the tetron labels of all GridQubits inside.
    """
    on_match = re.search(r'\.on\(', text[gate_end:gate_end + 200])
    if not on_match:
        return []
    on_start = gate_end + on_match.end()
    content = extract_balanced(text, on_start)
    qubits = parse_qubits(content)
    labels = []
    for q in qubits:
        t = gridqubit_to_tetron(q)
        if t is not None:
            labels.append(t)
    return labels


def parse_filename(path: Path) -> dict:
    m = RE_FILENAME.match(path.name)
    if not m:
        return None
    return {
        "is_patch": m.group(1) is not None,
        "n_qubits": int(m.group(2)),
        "n_cycles": int(m.group(3)),
        "seed":     int(m.group(4)),
        "elided":   int(m.group(5)),
        "pattern":  m.group(6),
    }


# ---------------------------------------------------------------------------
# Gate detection patterns
# ---------------------------------------------------------------------------

# Each pattern finds the START of a gate declaration.
# We then call find_on_qubits() starting from the match end.

RE_FSIMGATE = re.compile(
    r'cirq\.FSimGate\(\s*theta\s*=\s*([\d.e+\-]+)\s*,\s*phi\s*=\s*([\d.e+\-]+)\s*\)',
    re.DOTALL
)
RE_SQRT_X   = re.compile(r'\(\s*cirq\.X\s*\*\*\s*0\.5\s*\)')
RE_SQRT_Y   = re.compile(r'\(\s*cirq\.Y\s*\*\*\s*0\.5\s*\)')
RE_PHASED_X = re.compile(r'cirq\.PhasedXPowGate\([^)]*phase_exponent\s*=\s*0\.25[^)]*\)')
RE_RZ       = re.compile(r'cirq\.Rz\(')
RE_MOMENT   = re.compile(r'cirq\.Moment\(')


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_moment(mtext: str) -> list:
    """
    Parse one cirq.Moment(...) text block into a list of gate dicts.
    Rz gates are silently dropped.
    Returns list of gate dicts.
    """
    gates = []
    used_positions = set()

    def already_used(pos: int) -> bool:
        for s, e in used_positions:
            if s <= pos < e:
                return True
        return False

    # --- FSimGate ---
    for m in RE_FSIMGATE.finditer(mtext):
        if already_used(m.start()):
            continue
        theta_orig = float(m.group(1))
        phi_orig   = float(m.group(2))
        qubits     = find_on_qubits(mtext, m.end())
        gates.append({
            "gate":           "FSimGate",
            "qubits":         qubits,
            "theta":          THETA_IDEAL,
            "phi":            PHI_IDEAL,
            "theta_original": theta_orig,
            "phi_original":   phi_orig,
        })
        used_positions.add((m.start(), m.end()))

    # --- sqrtX ---
    for m in RE_SQRT_X.finditer(mtext):
        if already_used(m.start()):
            continue
        qubits = find_on_qubits(mtext, m.end())
        if qubits:
            gates.append({"gate": "sqrtX", "qubits": qubits})
            used_positions.add((m.start(), m.end()))

    # --- sqrtY ---
    for m in RE_SQRT_Y.finditer(mtext):
        if already_used(m.start()):
            continue
        qubits = find_on_qubits(mtext, m.end())
        if qubits:
            gates.append({"gate": "sqrtY", "qubits": qubits})
            used_positions.add((m.start(), m.end()))

    # --- sqrtW (PhasedXPowGate with phase_exponent=0.25) ---
    for m in RE_PHASED_X.finditer(mtext):
        if already_used(m.start()):
            continue
        qubits = find_on_qubits(mtext, m.end())
        if qubits:
            gates.append({"gate": "sqrtW", "qubits": qubits})
            used_positions.add((m.start(), m.end()))

    # Rz: not added (virtual frame rotation, dropped)

    return gates


def parse_circuit_py(path: Path) -> dict:
    """
    Parse a Cirq circuit Python file into a clean structured dict.

    Output schema
    -------------
    {
        "source_file":       str,
        "is_patch":          bool,
        "n_qubits":          int,
        "n_cycles":          int,
        "seed":              int,
        "elided":            int,      # 0, 6, or 18
        "pattern":           str,      # "EFGH"
        "theta_ideal":       float,    # pi/2
        "phi_ideal":         float,    # pi/6
        "qubit_map":         dict,     # "r,c" -> tetron_label
        "qubit_order":       list[int],# tetron labels in QUBIT_ORDER sequence
        "efgh_tetron_pairs": dict,     # EFGH pattern -> list of (a,b) tetron pairs
        "moments": [
            {
                "depth": int,          # 0-indexed, Rz-only moments removed
                "gates": [
                    {
                        "gate":   "sqrtX"|"sqrtY"|"sqrtW"|"FSimGate",
                        "qubits": [int],  # tetron logical labels
                        # FSimGate only:
                        "theta":          float,  # pi/2 (snapped)
                        "phi":            float,  # pi/6 (snapped)
                        "theta_original": float,
                        "phi_original":   float,
                    }, ...
                ]
            }, ...
        ],
        "n_moments_raw":    int,
        "n_moments":        int,
        "n_fsim":           int,
        "n_single_qubit":   int,
        "unmapped_qubits":  list[str],
    }
    """
    src  = path.read_text()
    meta = parse_filename(path)
    if meta is None:
        raise ValueError(f"Cannot parse filename: {path.name}")

    # ---- QUBIT_ORDER --------------------------------------------------------
    qo_match = re.search(r'QUBIT_ORDER\s*=\s*\[(.*?)\]', src, re.DOTALL)
    raw_qubit_order = parse_qubits(qo_match.group(1)) if qo_match else []

    unmapped = []
    qubit_order_tetron = []
    for q in raw_qubit_order:
        t = gridqubit_to_tetron(q)
        if t is None:
            unmapped.append(f"{q[0]},{q[1]}")
            qubit_order_tetron.append(-1)
        else:
            qubit_order_tetron.append(t)

    qubit_map = {
        f"{q[0]},{q[1]}": gridqubit_to_tetron(q)
        for q in raw_qubit_order
        if gridqubit_to_tetron(q) is not None
    }

    # ---- CIRCUIT body -------------------------------------------------------
    cb_match = re.search(
        r'CIRCUIT\s*=\s*cirq\.Circuit\(\s*\[(.*)\]\s*\)',
        src, re.DOTALL
    )
    if not cb_match:
        raise ValueError(f"Cannot find CIRCUIT in {path.name}")
    circuit_body = cb_match.group(1)

    # Split into per-moment text blocks
    positions = [m.start() for m in RE_MOMENT.finditer(circuit_body)]
    moment_texts = []
    for i, start in enumerate(positions):
        end = positions[i+1] if i+1 < len(positions) else len(circuit_body)
        moment_texts.append(circuit_body[start:end])

    # ---- Parse each moment --------------------------------------------------
    moments_raw = [parse_moment(mt) for mt in moment_texts]

    # ---- Drop empty (Rz-only) moments and re-index --------------------------
    moments_clean = []
    for raw_gates in moments_raw:
        if raw_gates:
            moments_clean.append({
                "depth": len(moments_clean),
                "gates": raw_gates,
            })

    # ---- Counters -----------------------------------------------------------
    n_fsim         = sum(1 for m in moments_clean for g in m["gates"] if g["gate"] == "FSimGate")
    n_single_qubit = sum(1 for m in moments_clean for g in m["gates"] if g["gate"] != "FSimGate")

    return {
        "source_file":       path.name,
        "is_patch":          meta["is_patch"],
        "n_qubits":          len(raw_qubit_order),
        "n_cycles":          meta["n_cycles"],
        "seed":              meta["seed"],
        "elided":            meta["elided"],
        "pattern":           meta["pattern"],
        "theta_ideal":       THETA_IDEAL,
        "phi_ideal":         PHI_IDEAL,
        "qubit_map":         qubit_map,
        "qubit_order":       qubit_order_tetron,
        "efgh_tetron_pairs": EFGH_TO_TETRON_PAIRS,
        "moments":           moments_clean,
        "n_moments_raw":     len(moments_raw),
        "n_moments":         len(moments_clean),
        "n_fsim":            n_fsim,
        "n_single_qubit":    n_single_qubit,
        "unmapped_qubits":   list(set(unmapped)),
    }


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def output_filename(meta: dict) -> str:
    patch = "_patch" if meta["is_patch"] else ""
    return (
        f"circuit{patch}_n{meta['n_qubits']}_m{meta['n_cycles']}"
        f"_s{meta['seed']}_e{meta['elided']}_p{meta['pattern']}_snapped.json"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Google 2019 Sycamore circuit .py files -> snapped JSON."
    )
    parser.add_argument("--src", type=Path, required=True,
        help="Directory containing circuit_*.py files (e.g. n12_m14/)")
    parser.add_argument("--out", type=Path, default=Path("src/circuits/processed"),
        help="Output directory for JSON files (default: src/circuits/processed)")
    parser.add_argument("--elided", type=int, nargs="*", default=None,
        help="Only process circuits with these elision counts (e.g. --elided 0). "
             "Default: process all.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    py_files  = sorted(args.src.glob("circuit*.py"))
    processed = skipped = 0
    errors    = []

    for path in py_files:
        meta = parse_filename(path)
        if meta is None:
            print(f"  SKIP (unrecognised filename): {path.name}")
            skipped += 1
            continue
        if args.elided is not None and meta["elided"] not in args.elided:
            skipped += 1
            continue

        try:
            result   = parse_circuit_py(path)
            out_name = output_filename(meta)
            out_path = args.out / out_name
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            tag = "[PATCH]" if meta["is_patch"] else "      "
            print(
                f"  OK {tag} s{meta['seed']} e{meta['elided']} | "
                f"{result['n_moments']} moments "
                f"({result['n_moments_raw']} raw) | "
                f"{result['n_fsim']} FSimGates | "
                f"{result['n_single_qubit']} sq gates | "
                f"-> {out_name}"
            )
            if result["unmapped_qubits"]:
                print(f"     WARNING unmapped GridQubits: {result['unmapped_qubits']}")
            processed += 1

        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            errors.append((path.name, str(e)))

    print(f"\nDone: {processed} processed, {skipped} skipped, {len(errors)} errors.")
    if errors:
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
