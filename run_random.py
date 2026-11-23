"""Random-sampling baseline with the same selection rule as run_gate.py.

Per run: draw `shots` uniformly random bitstrings, keep the one with the
lowest QUBO energy, decode it, and label it optimal / feasible / infeasible.
No quantum circuit is involved, so this shows how much of QAOA's success
comes from sampling alone.

Example:
  python run_random.py --encoding compact --instances 1-15 --reps 30
  python run_random.py --encoding unbalanced --lambda_1 0.75 --lambda_2 0.75 --instances 1-15
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from classical import optimum
from instances import INSTANCES
from qubo_builder import build_qubo
from run_gate import parse_instances

SEED = 10598


def best_of_random(model, rng, shots):
    """Lowest-energy bitstring among `shots` uniform random ones."""
    labels = list(model.bqm.variables)
    X = rng.integers(0, 2, size=(shots, len(labels)))
    energies = model.bqm.energies((X, labels))      # all rows at once
    best = int(np.argmin(energies))
    return {v: int(b) for v, b in zip(labels, X[best])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", choices=["paper", "compact", "unbalanced"], default="paper")
    ap.add_argument("--instances", nargs="+", default=["1-15"])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--lambda_1", type=float, default=0.75)
    ap.add_argument("--lambda_2", type=float, default=0.75)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"random_{args.encoding}.csv"

    total = Counter()
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "rep", "qubits", "status", "x", "energy", "G"])
        for i in parse_instances(args.instances):
            inst = INSTANCES[i]
            model = build_qubo(inst, args.encoding, args.gamma,
                               lambda_1=args.lambda_1, lambda_2=args.lambda_2)
            x_star, _ = optimum(inst)
            counts = Counter()
            for rep in range(args.reps):
                d = model.decode(best_of_random(model, rng, args.shots))
                status = ("optimal" if d["feasible"] and d["x"] == x_star
                          else "feasible" if d["feasible"] else "infeasible")
                counts[status] += 1
                w.writerow([i, rep, model.num_qubits, status, d["x"],
                            f"{d['energy']:.4f}", f"{d['G']:.4f}"])
            total.update(counts)
            print(f"#{i:>2} q={model.num_qubits:>2}  optimal {counts['optimal']:>2}  "
                  f"feasible {counts['feasible']:>2}  infeasible {counts['infeasible']:>2}")
    n = sum(total.values())
    print(f"total optimal {total['optimal']} / {n} ({100 * total['optimal'] / n:.1f}%)")
    print(f"results -> {csv_path}")


if __name__ == "__main__":
    main()
