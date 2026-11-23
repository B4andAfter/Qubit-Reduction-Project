"""Classical baselines: exhaustive search and Branch & Cut (PuLP + CBC).

Both solve the integer model directly (Eqs. 5-8 after taking the log),
not the QUBO. They give the ground truth the quantum runs are scored against.

Run:  python classical.py
"""
import itertools
import math
import time

import pulp
from pyomo.environ import *
from pyomo.future import solver_factory

from instances import INSTANCES, Instance


def exhaustive(inst: Instance):
    t0 = time.perf_counter()
    best = None
    for x in itertools.product(range(inst.n_max + 1), repeat=inst.ct):
        if not inst.n_min <= sum(x) <= inst.n_max:
            continue
        if sum(c * xi for c, xi in zip(inst.c, x)) > inst.C:
            continue
        G = sum(xi * math.log(1 - r) for xi, r in zip(x, inst.R))
        if best is None or G < best[1]:
            best = (x, G)
    return best[0], best[1], time.perf_counter() - t0


def branch_and_cut(inst: Instance):
    K=range(inst.ct)
    model = ConcreteModel()
    model.X = Var(K,domain=Integers,bounds=(0,inst.n_max))
    model.cons = ConstraintList()
    model.obj= Objective(expr=sum(model.X[k]*math.log(1-inst.R[k]) for k in K), sense=minimize)
    model.cons.add(sum(inst.c[k]*model.X[k] for k in K) <= inst.C)
    model.cons.add(sum(model.X[k] for k in K) >= inst.n_min)
    model.cons.add(sum(model.X[k] for k in K) <= inst.n_max)
    solver = SolverFactory("appsi_highs")
    t0 = time.perf_counter()
    solver.solve(model)
    dt = time.perf_counter() - t0
    xs = tuple(int(round(model.X[k].value)) for k in K)
    return xs, model.obj(), dt


def optimum(inst: Instance):
    """(x*, G*) used by the other scripts to label runs as optimal."""
    x, G, _ = exhaustive(inst)
    return x, G


if __name__ == "__main__":
    print(f"{'#':>2}  {'x* (exhaustive)':<18} {'G*':>8} {'R*':>7} {'t_exh(s)':>9}  {'x* (B&C)':<18} {'t_B&C(s)':>9}")
    for i, inst in INSTANCES.items():
        xe, Ge, te = exhaustive(inst)
        xb, Gb, tb = branch_and_cut(inst)
        flag = "" if xe == xb else "  <-- differ"
        print(f"{i:>2}  {str(xe):<18} {Ge:8.4f} {1 - math.exp(Ge):7.4f} {te:9.4f}  {str(xb):<18} {tb:9.4f}{flag}")
