"""Visualize CVRP variable co-occurrence for MTZ and DL formulations (no solve)."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).parent.parent))
from milp_visualizer import visualize

# ---------------------------------------------------------------------------
# CVRP instance
# ---------------------------------------------------------------------------

@dataclass
class CVRPInstance:
    locations: list[tuple[float, float]]  # index 0 = depot
    demands: list[int]                    # demands[0] = 0
    capacity: int
    n_vehicles: int

    @property
    def n_nodes(self) -> int:
        return len(self.locations)

    @property
    def n_customers(self) -> int:
        return self.n_nodes - 1

    def dist(self, i: int, j: int) -> float:
        xi, yi = self.locations[i]
        xj, yj = self.locations[j]
        return math.hypot(xi - xj, yi - yj)


def generate(
    n_customers: int,
    *,
    capacity: int | None = None,
    n_vehicles: int | None = None,
    demand_range: tuple[int, int] = (1, 10),
    grid: tuple[float, float] = (100.0, 100.0),
    seed: int | None = None,
) -> CVRPInstance:
    rng = random.Random(seed)
    depot = (grid[0] / 2.0, grid[1] / 2.0)
    customers = [(rng.uniform(0, grid[0]), rng.uniform(0, grid[1])) for _ in range(n_customers)]
    locations = [depot] + customers
    demands = [0] + [rng.randint(demand_range[0], demand_range[1]) for _ in range(n_customers)]
    total_demand = sum(demands)
    if capacity is None:
        capacity = math.ceil(total_demand / max(1, math.ceil(n_customers / 5)))
    if n_vehicles is None:
        n_vehicles = math.ceil(total_demand / capacity) + 1
    return CVRPInstance(locations=locations, demands=demands, capacity=capacity, n_vehicles=n_vehicles)


# ---------------------------------------------------------------------------
# Gurobi model builder
# ---------------------------------------------------------------------------

def build_model(instance: CVRPInstance, *, formulation: str = "mtz") -> gp.Model:
    """Build CVRP model without solving.

    formulation : "mtz" or "dl"
      mtz -- Miller-Tucker-Zemlin:
               u[i,k] - u[j,k] + Q * x[i,j,k] <= Q - d[j]
      dl  -- Desrochers-Laporte (strengthened, adds reverse arc):
               u[i,k] - u[j,k] + (Q-d[i]) x[i,j,k] - (Q-d[j]) x[j,i,k] <= Q - d[i] - d[j]
    """
    n = instance.n_nodes
    K = range(instance.n_vehicles)
    V = range(n)
    C = range(1, n)
    Q = instance.capacity
    d = instance.demands

    m = gp.Model(f"CVRP_{formulation.upper()}")
    m.Params.OutputFlag = 0

    arcs = [(i, j, k) for i in V for j in V for k in K if i != j]
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = m.addVars(V, K, lb=0.0, ub=float(Q), name="u")

    m.setObjective(gp.quicksum(instance.dist(i, j) * x[i, j, k] for (i, j, k) in arcs), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum(x[i, j, k] for j in V for k in K if j != i) == 1 for i in C), name="visit")
    m.addConstrs(
        (gp.quicksum(x[j, i, k] for j in V if j != i) == gp.quicksum(x[i, j, k] for j in V if j != i)
         for i in V for k in K),
        name="flow",
    )
    m.addConstrs((gp.quicksum(x[0, j, k] for j in C) <= 1 for k in K), name="depart")

    if formulation == "mtz":
        m.addConstrs(
            (u[i, k] - u[j, k] + Q * x[i, j, k] <= Q - d[j] for i in V for j in C for k in K if i != j),
            name="mtz",
        )
    elif formulation == "dl":
        m.addConstrs(
            (u[i, k] - u[j, k] + (Q - d[i]) * x[i, j, k] - (Q - d[j]) * x[j, i, k] <= Q - d[i] - d[j]
             for i in V for j in C for k in K if i != j),
            name="dl",
        )
    else:
        raise ValueError(f"unknown formulation {formulation!r}, use 'mtz' or 'dl'")

    m.addConstrs((u[0, k] == 0.0 for k in K), name="depot_load")
    m.addConstrs((u[i, k] >= d[i] for i in C for k in K), name="load_lb")
    m.update()
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INSTANCE_SEED = 2640
N_CUSTOMERS = 60
N_VEHICLES = 2


def main() -> None:
    inst = generate(n_customers=N_CUSTOMERS, n_vehicles=N_VEHICLES, seed=INSTANCE_SEED)
    print(f"instance: {inst.n_customers} customers  capacity={inst.capacity}  vehicles={inst.n_vehicles}")
    print(f"demands: {inst.demands[1:]}\n")

    for form in ("mtz", "dl"):
        print(f"--- {form.upper()} ---")
        m = build_model(inst, formulation=form)
        visualize(m, mode="variables", output=f"cvrp_{form}.html", label_nodes=True)
        print()


if __name__ == "__main__":
    main()
