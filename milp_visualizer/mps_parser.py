"""MPS file parser for Mixed Integer Linear Programs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MPS:
    name: str = ""
    # row_type: row_name -> 'N'|'L'|'G'|'E'
    row_types: dict[str, str] = field(default_factory=dict)
    objective: str = ""
    # coefficients: row_name -> {col_name -> float}
    coefficients: dict[str, dict[str, float]] = field(default_factory=lambda: {})
    # rhs: row_name -> float
    rhs: dict[str, float] = field(default_factory=dict)
    # ranges: row_name -> float
    ranges: dict[str, float] = field(default_factory=dict)
    # bounds: col_name -> {'lb': float, 'ub': float, 'type': str}
    bounds: dict[str, dict] = field(default_factory=dict)
    # integer_vars: set of column names declared inside INTORG/INTEND markers
    integer_vars: set[str] = field(default_factory=set)
    # binary_vars: set of column names declared BV
    binary_vars: set[str] = field(default_factory=set)

    @property
    def variables(self) -> list[str]:
        cols: set[str] = set()
        for row_coeffs in self.coefficients.values():
            cols.update(row_coeffs)
        return sorted(cols)

    @property
    def constraints(self) -> list[str]:
        return [r for r in self.row_types if r != self.objective]


def parse(path: str | Path) -> MPS:
    model = MPS()
    section = None
    in_integer_block = False

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")

            if not line.strip() or line.startswith("$"):
                continue

            if line[0] != " " and not line[0].isdigit():
                keyword = line.split()[0].upper()
                if keyword == "ENDATA":
                    break
                section = keyword
                if section == "NAME":
                    parts = line.split()
                    model.name = parts[1] if len(parts) > 1 else ""
                continue

            tokens = line.split()
            if not tokens:
                continue

            if section == "ROWS":
                row_type, row_name = tokens[0], tokens[1]
                model.row_types[row_name] = row_type
                if row_type == "N" and not model.objective:
                    model.objective = row_name

            elif section == "COLUMNS":
                if tokens[0] == "MARKER":
                    marker_type = tokens[-1].strip("'")
                    in_integer_block = marker_type == "INTORG"
                    continue

                col = tokens[0]
                if in_integer_block:
                    model.integer_vars.add(col)

                i = 1
                while i + 1 < len(tokens):
                    row, val = tokens[i], float(tokens[i + 1])
                    model.coefficients.setdefault(row, {})[col] = val
                    i += 2

            elif section == "RHS":
                i = 1
                while i + 1 < len(tokens):
                    row, val = tokens[i], float(tokens[i + 1])
                    model.rhs[row] = val
                    i += 2

            elif section == "RANGES":
                i = 1
                while i + 1 < len(tokens):
                    row, val = tokens[i], float(tokens[i + 1])
                    model.ranges[row] = val
                    i += 2

            elif section == "BOUNDS":
                bound_type = tokens[0].upper()
                col = tokens[2]
                val = float(tokens[3]) if len(tokens) > 3 else 0.0

                if col not in model.bounds:
                    model.bounds[col] = {"lb": 0.0, "ub": math.inf}

                if bound_type == "LO":
                    model.bounds[col]["lb"] = val
                elif bound_type == "UP":
                    model.bounds[col]["ub"] = val
                elif bound_type == "FX":
                    model.bounds[col]["lb"] = val
                    model.bounds[col]["ub"] = val
                elif bound_type == "FR":
                    model.bounds[col]["lb"] = -math.inf
                    model.bounds[col]["ub"] = math.inf
                elif bound_type == "MI":
                    model.bounds[col]["lb"] = -math.inf
                elif bound_type == "PL":
                    model.bounds[col]["ub"] = math.inf
                elif bound_type == "BV":
                    model.bounds[col]["lb"] = 0.0
                    model.bounds[col]["ub"] = 1.0
                    model.binary_vars.add(col)
                    model.integer_vars.add(col)
                elif bound_type == "LI":
                    model.bounds[col]["lb"] = val
                    model.integer_vars.add(col)
                elif bound_type == "UI":
                    model.bounds[col]["ub"] = val
                    model.integer_vars.add(col)

    return model
