"""LP file parser for Mixed Integer Linear Programs."""

from __future__ import annotations

import math
import re
from pathlib import Path

from .mps_parser import MPS

# Variable names: letter/underscore start, then letters/digits/common LP special chars
_VAR_PAT = r'[A-Za-z_!#$%&@][A-Za-z0-9_,.\[\](){}!#$%&@\']*'
_NUM_PAT = r'[+-]?\s*\d+(?:\.\d*)?(?:[eE][+-]?\d+)?'
_SENSE_RE = re.compile(r'(<=|>=|=)')

_SECTION_MAP = {
    'minimize': 'objective', 'minimize\n': 'objective',
    'min': 'objective', 'minimum': 'objective',
    'maximize': 'objective', 'max': 'objective', 'maximum': 'objective',
    'subject to': 'constraints', 'st': 'constraints',
    's.t.': 'constraints', 'such that': 'constraints',
    'bounds': 'bounds', 'bound': 'bounds',
    'generals': 'generals', 'general': 'generals', 'gen': 'generals',
    'integers': 'generals', 'integer': 'generals',
    'binaries': 'binaries', 'binary': 'binaries', 'bin': 'binaries',
    'end': 'end',
}


def _detect_section(line: str) -> str | None:
    return _SECTION_MAP.get(line.strip().lower())


def _parse_expr(text: str) -> dict[str, float]:
    """Extract {var: coeff} from an LP expression string."""
    # Space-separate +/- that aren't part of scientific notation
    text = re.sub(r'(?<![eE])([+-])', r' \1 ', text)
    result: dict[str, float] = {}
    pending_sign = 1.0
    pending_coeff: float | None = None

    for tok in text.split():
        if tok == '+':
            pending_sign = 1.0
        elif tok == '-':
            pending_sign = -1.0
        elif tok == '*':
            pass
        else:
            try:
                pending_coeff = float(tok) * pending_sign
                pending_sign = 1.0
            except ValueError:
                coeff = pending_coeff if pending_coeff is not None else pending_sign
                result[tok] = result.get(tok, 0.0) + coeff
                pending_coeff = None
                pending_sign = 1.0
    return result


def _parse_bound_line(line: str, model: MPS) -> None:
    stripped = line.strip()
    tokens = stripped.split()

    if len(tokens) >= 2 and tokens[-1].lower() == 'free':
        var = tokens[0]
        model.bounds.setdefault(var, {'lb': 0.0, 'ub': math.inf})
        model.bounds[var]['lb'] = -math.inf
        model.bounds[var]['ub'] = math.inf
        return
    if len(tokens) >= 2 and tokens[0].lower() == 'free':
        var = tokens[1]
        model.bounds.setdefault(var, {'lb': 0.0, 'ub': math.inf})
        model.bounds[var]['lb'] = -math.inf
        model.bounds[var]['ub'] = math.inf
        return

    parts = _SENSE_RE.split(stripped)
    parts = [p.strip() for p in parts if p.strip()]

    def _to_float(s: str) -> float:
        try:
            return float(s)
        except ValueError:
            return -math.inf if s.startswith('-') else math.inf

    def _is_num(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return 'inf' in s.lower()

    if len(parts) == 3:
        lhs, sense, rhs = parts
        if _is_num(lhs) and not _is_num(rhs):
            var, num = rhs, _to_float(lhs)
            model.bounds.setdefault(var, {'lb': 0.0, 'ub': math.inf})
            if sense == '<=':
                model.bounds[var]['lb'] = num
            elif sense == '>=':
                model.bounds[var]['ub'] = num
        elif not _is_num(lhs):
            var, num = lhs, _to_float(rhs)
            model.bounds.setdefault(var, {'lb': 0.0, 'ub': math.inf})
            if sense == '<=':
                model.bounds[var]['ub'] = num
            elif sense == '>=':
                model.bounds[var]['lb'] = num
            elif sense == '=':
                model.bounds[var]['lb'] = model.bounds[var]['ub'] = num

    elif len(parts) == 5:
        lb_str, _s1, var, _s2, ub_str = parts
        model.bounds.setdefault(var, {'lb': 0.0, 'ub': math.inf})
        model.bounds[var]['lb'] = _to_float(lb_str)
        model.bounds[var]['ub'] = _to_float(ub_str)


def parse_lp(path: str | Path) -> MPS:
    """Parse an LP format file into an MPS dataclass."""
    model = MPS()
    section: str | None = None

    with open(path) as f:
        raw_lines = f.readlines()

    # Strip comments, empty lines
    lines: list[str] = []
    for line in raw_lines:
        line = line.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith('\\') or stripped.startswith('//'):
            continue
        # Inline backslash comment (not in scientific notation context)
        line = re.split(r'\s+\\', line)[0].rstrip()
        if line.strip():
            lines.append(line)

    # Accumulate multi-line constraint/objective blocks
    # Strategy: collect all lines per section, then parse
    obj_lines: list[str] = []
    constraint_lines: list[str] = []
    bound_lines: list[str] = []
    general_lines: list[str] = []
    binary_lines: list[str] = []

    for line in lines:
        sec = _detect_section(line)
        if sec == 'end':
            break
        if sec is not None:
            section = sec
            continue

        if section == 'objective':
            obj_lines.append(line.strip())
        elif section == 'constraints':
            constraint_lines.append(line.strip())
        elif section == 'bounds':
            bound_lines.append(line.strip())
        elif section == 'generals':
            general_lines.append(line.strip())
        elif section == 'binaries':
            binary_lines.append(line.strip())

    # --- Objective ---
    obj_text = ' '.join(obj_lines)
    if ':' in obj_text:
        name_part, expr_part = obj_text.split(':', 1)
        obj_name = name_part.strip()
    else:
        obj_name, expr_part = 'obj', obj_text
    model.objective = obj_name
    model.row_types[obj_name] = 'N'
    for var, coeff in _parse_expr(expr_part).items():
        model.coefficients.setdefault(obj_name, {})[var] = coeff

    # --- Constraints ---
    # Join continuation lines: a new constraint starts when the line contains ':'
    # before any sense operator
    blocks: list[str] = []
    buf: list[str] = []
    for line in constraint_lines:
        colon_pos = line.find(':')
        sense_pos = min(
            (line.find(s) for s in ('<=', '>=', '=') if s in line),
            default=len(line),
        )
        is_new = colon_pos != -1 and colon_pos < sense_pos
        if is_new and buf:
            blocks.append(' '.join(buf))
            buf = []
        buf.append(line)
    if buf:
        blocks.append(' '.join(buf))

    for block in blocks:
        colon_pos = block.find(':')
        sense_match = _SENSE_RE.search(block)
        if not sense_match:
            continue

        if colon_pos != -1 and colon_pos < sense_match.start():
            row_name = block[:colon_pos].strip()
            rest = block[colon_pos + 1:]
        else:
            row_name = f'c{len(model.row_types)}'
            rest = block

        parts = _SENSE_RE.split(rest, maxsplit=1)
        if len(parts) != 3:
            continue
        lhs_expr, sense, rhs_str = parts
        sense_map = {'<=': 'L', '>=': 'G', '=': 'E'}
        model.row_types[row_name] = sense_map.get(sense, 'L')
        try:
            model.rhs[row_name] = float(rhs_str.strip())
        except ValueError:
            model.rhs[row_name] = 0.0
        for var, coeff in _parse_expr(lhs_expr).items():
            model.coefficients.setdefault(row_name, {})[var] = coeff

    # --- Bounds ---
    for line in bound_lines:
        _parse_bound_line(line, model)

    # --- Generals / integers ---
    for line in general_lines:
        for tok in line.split():
            model.integer_vars.add(tok)

    # --- Binaries ---
    for line in binary_lines:
        for tok in line.split():
            model.binary_vars.add(tok)
            model.integer_vars.add(tok)
            model.bounds.setdefault(tok, {'lb': 0.0, 'ub': 1.0})

    return model
