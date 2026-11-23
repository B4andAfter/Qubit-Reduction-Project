"""Quantum annealing on the RAP QUBO (D-Wave Ocean SDK).

  --sampler qpu : real D-Wave hardware (needs `pip install dwave-system`
                  and a Leap API token: `dwave config create` or DWAVE_API_TOKEN).
                  Paper setup: EmbeddingComposite (minorminer), default chain
                  strength, linear schedule, majority-vote unembedding, 1000 reads.
  --sampler sa  : classical simulated annealing on the SAME QUBO (dwave-neal).
                  No account needed. Use it to develop, and as the classical
                  baseline the paper does not have.

Output per instance: how many of the reads hit the optimum (cf. Table 9),
logical qubits, physical qubits after embedding (QPU only), time.

Examples:
  python run_annealing.py --sampler sa  --instances 1-15
  python run_annealing.py --sampler qpu --instances 1-15 --reads 1000
  python run_annealing.py --sampler sa  --encoding compact --instances 1-15
"""
import argparse
import csv
import time
from pathlib import Path

from classical import optimum
from instances import INSTANCES
from qubo_builder import build_qubo
from run_gate import parse_instances


def get_sampler(kind):
    if kind == "sa":
        import neal
        return neal.SimulatedAnnealingSampler()
    if kind == "qpu":
        from dwave.system import DWaveSampler, EmbeddingComposite
        return EmbeddingComposite(DWaveSampler())
    raise ValueError(kind)


def physical_qubits(sampleset):
    """Count physical qubits used by the embedding (QPU runs only)."""
    ctx = sampleset.info.get("embedding_context", {})
    emb = ctx.get("embedding")
    return sum(len(chain) for chain in emb.values()) if emb else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["sa", "qpu"], default="sa")
    ap.add_argument("--instances", nargs="+", default=["1-15"])
    ap.add_argument("--reads", type=int, default=1000)
    ap.add_argument("--encoding", choices=["paper", "compact","unbalanced"], default="paper")
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--lambda_1", type=float, default=0.75)
    ap.add_argument("--lambda_2", type=float, default=1.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    sampler = get_sampler(args.sampler)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"anneal_{args.sampler}_{args.encoding}.csv"

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "logical_qubits", "physical_qubits", "reads",
                    "optimal_reads", "feasible_reads", "infeasible_reads",
                    "best_x", "best_status", "seconds", "qpu_access_us"])
        for i in parse_instances(args.instances):
            inst = INSTANCES[i]
            model = build_qubo(inst, args.encoding, args.gamma,
                   lambda_1=args.lambda_1, lambda_2=args.lambda_2)
            x_star, _ = optimum(inst)

            kwargs = dict(num_reads=args.reads)
            if args.sampler == "qpu":
                kwargs["return_embedding"] = True
            t0 = time.perf_counter()
            ss = sampler.sample(model.bqm, **kwargs)
            dt = time.perf_counter() - t0

            opt = feas = inf = 0
            for rec in ss.data(["sample", "num_occurrences"]):
                d = model.decode(rec.sample)
                n = rec.num_occurrences
                if d["feasible"] and d["x"] == x_star:
                    opt += n
                elif d["feasible"]:
                    feas += n
                else:
                    inf += n
            best = model.decode(ss.first.sample)
            best_status = ("optimal" if best["feasible"] and best["x"] == x_star
                           else "feasible" if best["feasible"] else "infeasible")
            phys = physical_qubits(ss) if args.sampler == "qpu" else None
            qpu_us = ss.info.get("timing", {}).get("qpu_access_time")

            w.writerow([i, model.num_qubits, phys, args.reads, opt, feas, inf,
                        best["x"], best_status, f"{dt:.4f}", qpu_us])
            fh.flush()
            print(f"#{i:>2} logical q={model.num_qubits:>2}"
                  f"{'' if phys is None else f'  physical q={phys:>2}'}"
                  f"  optimal {opt:>4}/{args.reads}  feasible {feas:>4}  infeasible {inf:>4}"
                  f"  best={best_status:<10} wall {dt:6.3f}s"
                  f"{'' if qpu_us is None else f'  QPU access {qpu_us / 1e6:.3f}s'}")
    print(f"results -> {csv_path}")


if __name__ == "__main__":
    main()
