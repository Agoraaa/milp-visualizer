"""TSP solved with iterative DFJ subtour elimination using OR-Tools CBC."""

import argparse
import math
import random

from ortools.linear_solver import pywraplp

from milp_visualizer import visualize


def build_tsp(n: int, seed: int):
    rng = random.Random(seed)
    coords = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]

    solver = pywraplp.Solver.CreateSolver("CBC")

    x = {
        (i, j): solver.BoolVar(f"x_{i}_{j}")
        for i in range(n) for j in range(n) if i != j
    }

    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(n) if j != i) == 1, f"out_{i}")
        solver.Add(sum(x[j, i] for j in range(n) if j != i) == 1, f"in_{i}")

    objective = solver.Objective()
    for (i, j), var in x.items():
        dist = math.dist(coords[i], coords[j])
        objective.SetCoefficient(var, dist)
    objective.SetMinimization()

    return solver, x, coords, n


def find_subtours(n: int, x: dict, solver) -> list[list[int]]:
    succ = {
        i: j
        for i in range(n) for j in range(n)
        if i != j and x[i, j].solution_value() > 0.5
    }
    visited: set[int] = set()
    tours: list[list[int]] = []
    for start in range(n):
        if start in visited:
            continue
        tour, cur = [], start
        while cur not in visited:
            visited.add(cur)
            tour.append(cur)
            cur = succ.get(cur, start)
        tours.append(tour)
    return tours


def _node_categories(
    n: int,
    x: dict,
    current_sec: set[int] | None,
    prev_sec_vars: set[str],
) -> dict[str, str]:
    current_vars = (
        {f"x_{i}_{j}" for i in current_sec for j in current_sec if i != j}
        if current_sec else set()
    )
    cats = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            name = f"x_{i}_{j}"
            if name in current_vars:
                cats[name] = "current_sec"
            elif name in prev_sec_vars:
                cats[name] = "prev_sec"
            elif x[i, j].solution_value() > 0.5:
                cats[name] = "active"
            else:
                cats[name] = "idle"
    return cats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    solver, x, coords, n = build_tsp(args.cities, args.seed)

    cut_idx = 0
    prev_sec_vars: set[str] = set()
    while True:
        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            print(f"solver status: {status} — stopping")
            break

        obj = solver.Objective().Value()
        tours = find_subtours(n, x, solver)
        print(f"\n--- iteration {cut_idx}  obj={obj:.2f}  subtours={len(tours)} ---")
        for t in tours:
            print(f"  {t}")

        if len(tours) == 1:
            print(f"\nvalid tour: {tours[0] + [tours[0][0]]}")
            visualize(solver, f"visualizations/tsp_iter_{cut_idx:02d}.html",
                      mode="variables", label_nodes=False,
                      node_categories=_node_categories(n, x, None, prev_sec_vars))
            break

        s = min(tours, key=len)
        current_vars = {f"x_{i}_{j}" for i in s for j in s if i != j}
        if cut_idx > 2:
            visualize(solver, f"visualizations/tsp_iter_{cut_idx:02d}.html",
                    mode="variables", label_nodes=False,
                    node_categories=_node_categories(n, x, set(s), prev_sec_vars))

        cut = solver.Constraint(0.0, len(s) - 1.0, f"sec_{cut_idx}_{min(s)}")
        for i in s:
            for j in s:
                if i != j:
                    cut.SetCoefficient(x[i, j], 1.0)
        print(f"  SEC {cut_idx}: {s} <= {len(s) - 1}")
        prev_sec_vars = current_vars
        cut_idx += 1


if __name__ == "__main__":
    main()
