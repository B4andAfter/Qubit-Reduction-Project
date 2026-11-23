"""QAOA and VQE on the RAP QUBO (gate-based, Qiskit noiseless simulator).

Mirrors the paper's setup (Table 5):
  * Qiskit MinimumEigenOptimizer + QAOA / SamplingVQE
  * Aer sampler, 1000 shots, seed 10598
  * QAOA: reps (layers) p = 1
  * VQE : TwoLocal(ry, rz, cz, linear), 1 layer
  * initial angles uniform in [0, 2*pi]
  * optimizers COBYLA (default) and SPSA (maxiter 100)
  * 30 repetitions per instance

Each run is labelled optimal / feasible / infeasible against the exact optimum.

Examples:
  python run_gate.py --algo qaoa --opt cobyla --instances 1 2 3 --reps 30
  python run_gate.py --algo vqe  --opt spsa   --instances 1 --reps 5 --layers 4
  python run_gate.py --algo qaoa --opt cobyla --encoding compact --instances 1-15
"""
import argparse
import csv
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from classical import optimum
from instances import INSTANCES
from qubo_builder import build_qubo, to_quadratic_program

SEED = 10598
# Aer's V1 Sampler is deprecated but is what qiskit-algorithms 0.3 accepts.
warnings.filterwarnings("ignore", message=".*Sampler has been deprecated.*")


def parse_instances(tokens):
    out = []
    for t in tokens:
        if "-" in t:
            a, b = map(int, t.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(t))
    return out


def make_optimizer(name):
    if name == "cobyla":
        return COBYLA()
    if name == "spsa":
        return SPSA(maxiter=100)
    raise ValueError(name)


def make_sampler(shots):
    if shots > 0:
        return Sampler(run_options={"shots": shots, "seed": SEED})
    # shots = 0 -> exact probabilities (statevector). Every bitstring with
    # non-zero amplitude then counts as "measured", which makes finding the
    # optimum far easier than with real shots. Diagnostic only.
    return Sampler(run_options={"shots": None}, backend_options={"method": "statevector"})


def make_solver(algo, n_qubits, opt_name, layers, rng, shots):
    sampler = make_sampler(shots)
    optimizer = make_optimizer(opt_name)
    if algo == "qaoa":
        n_params = 2 * layers                                  # Eq. 28
        x0 = rng.uniform(0, 2 * np.pi, n_params)
        return QAOA(sampler, optimizer, reps=layers, initial_point=x0)
    if algo == "vqe":
        ansatz = TwoLocal(n_qubits, ["ry", "rz"], "cz", entanglement="linear", reps=layers)
        x0 = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
        return SamplingVQE(sampler, ansatz, optimizer, initial_point=x0)
    raise ValueError(algo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["qaoa", "vqe"], required=True)
    ap.add_argument("--opt", choices=["cobyla", "spsa"], default="cobyla")
    ap.add_argument("--instances", nargs="+", default=["1"])
    ap.add_argument("--reps", type=int, default=30, help="independent runs per instance")
    ap.add_argument("--layers", type=int, default=1, help="QAOA p / VQE ansatz reps")
    ap.add_argument("--encoding", choices=["paper", "compact","unbalanced"], default="paper")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--lambda_1", type=float, default=0.75)
    ap.add_argument("--lambda_2", type=float, default=1.0)
    ap.add_argument("--shots", type=int, default=1000,
                    help="0 = exact probabilities instead of sampling (diagnostic)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    algorithm_globals.random_seed = SEED
    rng = np.random.default_rng(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.algo}_{args.opt}_p{args.layers}_{args.encoding}"
    csv_path = out_dir / f"{tag}.csv"

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "rep", "qubits", "status", "slacks_ok", "x", "energy", "G", "seconds","P_opt"])
        for i in parse_instances(args.instances):
            inst = INSTANCES[i]
            model = build_qubo(inst, args.encoding, args.gamma,
                   lambda_1=args.lambda_1, lambda_2=args.lambda_2)
            qp = to_quadratic_program(model)
            x_star, _ = optimum(inst)
            statuses, energies, times = Counter(), [], []
            p_opt = 0
            for rep in range(args.reps):
                solver = make_solver(args.algo, model.num_qubits, args.opt, args.layers, rng, args.shots)
                t0 = time.perf_counter()
                res = MinimumEigenOptimizer(solver).solve(qp)
                for s in res.samples:
                    sample = {v.name: int(round(b)) for v, b in zip(qp.variables, s.x)}
                    d = model.decode(sample)
                    if d["feasible"] and d["x"] == x_star:
                        p_opt += s.probability


                dt = time.perf_counter() - t0
                sample = {v.name: int(round(val)) for v, val in zip(qp.variables, res.x)}
                d = model.decode(sample)
                status = ("optimal" if d["feasible"] and d["x"] == x_star
                          else "feasible" if d["feasible"] else "infeasible")
                statuses[status] += 1
                if args.encoding != "unbalanced" and d["feasible"] and not d["slacks_ok"]:
                    statuses["slack_err"] += 1
                energies.append(d["energy"])
                times.append(dt)
                slack_col = "n/a" if args.encoding == "unbalanced" else d["slacks_ok"]
                w.writerow([i, rep, model.num_qubits, status, slack_col, d["x"],
                            f"{d['energy']:.4f}", f"{d['G']:.4f}", f"{dt:.2f}",f"{p_opt:.2f}"])
                fh.flush()
            print(f"#{i:>2} q={model.num_qubits:>2}  optimal {statuses['optimal']:>2}  "
                  f"feasible {statuses['feasible']:>2}  infeasible {statuses['infeasible']:>2}  "
                  f"(slack errors {statuses['slack_err']:>2})  "
                  f"energy {np.mean(energies):8.4f} ± {np.std(energies, ddof=1) if len(energies) > 1 else 0:7.4f}  "
                  f"mean time {np.mean(times):7.2f}s"
                  f"P_opt {p_opt:7.2f}")
    print(f"per-run results -> {csv_path}")


if __name__ == "__main__":
    main()
