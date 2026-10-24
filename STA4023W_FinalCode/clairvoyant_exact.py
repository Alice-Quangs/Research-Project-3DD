"""Offline geometric MILP: containment, non-overlap, six rotations and bin cost.
Adapted from the supplied research_benchmark.py; no support/weight/blocking constraints.
"""
from __future__ import annotations
import importlib, math, time
from typing import Any

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan

def unique_orientations(item: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    l, w, h = item
    return sorted({
        (l, w, h), (l, h, w), (w, l, h),
        (w, h, l), (h, l, w), (h, w, l),
    })

def non_dominated_bin_types(bin_types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove a type only when another type is no smaller on every fixed axis
    and costs no more. This is safe for the exact model used here."""
    kept = []
    for index, candidate in enumerate(bin_types):
        dominated = False
        for other_index, other in enumerate(bin_types):
            if index == other_index:
                continue
            no_smaller = (
                other["length"] >= candidate["length"]
                and other["width"] >= candidate["width"]
                and other["height"] >= candidate["height"]
            )
            no_costlier = other["cost"] <= candidate["cost"]
            strictly_better = (
                other["length"] > candidate["length"]
                or other["width"] > candidate["width"]
                or other["height"] > candidate["height"]
                or other["cost"] < candidate["cost"]
            )
            if no_smaller and no_costlier and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(dict(candidate))
    return kept

def item_fits_type(item: tuple[int, int, int], bin_type: dict[str, Any]) -> bool:
    dimensions = (bin_type["length"], bin_type["width"], bin_type["height"])
    return any(all(orientation[d] <= dimensions[d] for d in range(3))
               for orientation in unique_orientations(item))

def build_bin_copies(
    items: list[tuple[int, int, int]],
    bin_types: list[dict[str, Any]],
    heuristic_upper_bound: float,
) -> list[dict[str, Any]]:
    n = len(items)
    copies: list[dict[str, Any]] = []
    for type_index, bin_type in enumerate(non_dominated_bin_types(bin_types)):
        cost = float(bin_type["cost"])
        maximum = n if cost <= 0 else min(n, int(heuristic_upper_bound // cost))
        if maximum == 0 and any(item_fits_type(item, bin_type) for item in items):
            maximum = 1
        for copy_index in range(maximum):
            copy = dict(bin_type)
            copy["type_index"] = type_index
            copy["copy_index"] = copy_index
            copies.append(copy)
    return copies

def solve_exact(
    items: list[tuple[int, int, int]],
    bin_types: list[dict[str, Any]],
    heuristic_upper_bound: float,
    time_limit_seconds: int,
) -> dict[str, Any]:
    try:
        pywraplp = importlib.import_module("ortools.linear_solver.pywraplp")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OR-Tools is required. Install it with: python -m pip install ortools"
        ) from exc

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools could not create the SCIP solver.")
    solver.set_time_limit(int(time_limit_seconds * 1000))

    n = len(items)
    rotations = [unique_orientations(item) for item in items]
    bins = build_bin_copies(items, bin_types, heuristic_upper_bound)
    if not bins:
        raise ValueError("No candidate bins remain after safe upper-bound pruning.")

    max_bin = [max(b[("length", "width", "height")[d]] for b in bins) for d in range(3)]
    max_item = [max(rotation[d] for item_rotations in rotations for rotation in item_rotations)
                for d in range(3)]
    big_m = [max_bin[d] + max_item[d] for d in range(3)]

    q = [solver.IntVar(0, 1, f"q_{j}") for j in range(len(bins))]
    x: dict[tuple[int, int], Any] = {}
    for i, item in enumerate(items):
        for j, bin_type in enumerate(bins):
            upper = 1 if item_fits_type(item, bin_type) else 0
            x[i, j] = solver.IntVar(0, upper, f"x_{i}_{j}")

    rotation_vars: dict[tuple[int, int], Any] = {}
    for i, item_rotations in enumerate(rotations):
        for r in range(len(item_rotations)):
            rotation_vars[i, r] = solver.IntVar(0, 1, f"rot_{i}_{r}")

    position = {
        (i, d): solver.NumVar(0, max_bin[d], f"pos_{i}_{d}")
        for i in range(n) for d in range(3)
    }

    oriented_size = {
        (i, d): sum(rotations[i][r][d] * rotation_vars[i, r]
                    for r in range(len(rotations[i])))
        for i in range(n) for d in range(3)
    }

    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(len(bins))) == 1)
        solver.Add(sum(rotation_vars[i, r] for r in range(len(rotations[i]))) == 1)

    for j in range(len(bins)):
        for i in range(n):
            solver.Add(x[i, j] <= q[j])
        solver.Add(q[j] <= sum(x[i, j] for i in range(n)))

        bin_volume = bins[j]["length"] * bins[j]["width"] * bins[j]["height"]
        solver.Add(
            sum((items[i][0] * items[i][1] * items[i][2]) * x[i, j] for i in range(n))
            <= bin_volume * q[j]
        )

    # Identical-bin symmetry: copy k+1 may not be used before copy k.
    by_type: dict[int, list[int]] = {}
    for j, bin_type in enumerate(bins):
        by_type.setdefault(bin_type["type_index"], []).append(j)
    for indices in by_type.values():
        indices.sort(key=lambda j: bins[j]["copy_index"])
        for previous, current in zip(indices, indices[1:]):
            solver.Add(q[previous] >= q[current])

    axis_names = ("length", "width", "height")
    for i in range(n):
        for j, bin_type in enumerate(bins):
            for d, axis in enumerate(axis_names):
                solver.Add(
                    position[i, d] + oriented_size[i, d]
                    <= bin_type[axis] + big_m[d] * (1 - x[i, j])
                )

    # Six disjunctive separation choices for each item pair in each bin.
    for i in range(n):
        for k in range(i + 1, n):
            for j in range(len(bins)):
                separation = [solver.IntVar(0, 1, f"sep_{i}_{k}_{j}_{direction}")
                              for direction in range(6)]
                for variable in separation:
                    solver.Add(variable <= x[i, j])
                    solver.Add(variable <= x[k, j])
                solver.Add(sum(separation) >= x[i, j] + x[k, j] - 1)

                for d in range(3):
                    solver.Add(
                        position[i, d] + oriented_size[i, d]
                        <= position[k, d] + big_m[d] * (1 - separation[2 * d])
                    )
                    solver.Add(
                        position[k, d] + oriented_size[k, d]
                        <= position[i, d] + big_m[d] * (1 - separation[2 * d + 1])
                    )

    objective_expression = sum(float(bins[j]["cost"]) * q[j] for j in range(len(bins)))
    solver.Add(objective_expression <= float(heuristic_upper_bound))
    solver.Minimize(objective_expression)

    start = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - start

    status_names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status_name = status_names.get(status, f"UNKNOWN_{status}")
    has_solution = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    incumbent = float(solver.Objective().Value()) if has_solution else math.nan
    try:
        best_bound = float(solver.Objective().BestBound())
    except (AttributeError, RuntimeError):
        best_bound = incumbent if status == pywraplp.Solver.OPTIMAL else math.nan

    if status == pywraplp.Solver.OPTIMAL:
        best_bound = incumbent

    return {
        "solver_status": status_name,
        "is_optimal": status == pywraplp.Solver.OPTIMAL,
        "incumbent_cost": incumbent,
        "best_bound": best_bound,
        "solver_gap_pct": (
            100.0 * safe_div(incumbent - best_bound, incumbent)
            if has_solution and math.isfinite(best_bound) else math.nan
        ),
        "exact_time_seconds": elapsed,
        "candidate_bins": len(bins),
        "variables": solver.NumVariables(),
        "constraints": solver.NumConstraints(),
    }
