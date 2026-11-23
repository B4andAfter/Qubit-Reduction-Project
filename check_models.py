"""Sanity check before any quantum run.

For every instance and both encodings:
  * count qubits
  * brute-force the QUBO and confirm its lowest-energy state is the
    true optimum (so the penalty weight gamma is large enough)

Run:  python check_models.py            (gamma = 1.5, the paper's value)
      python check_models.py --gamma 0.5
"""
import argparse
import csv

from classical import optimum
from instances import INSTANCES
from qubo_builder import brute_force_ground_state, build_qubo, count_bits, embedding_cost

ap = argparse.ArgumentParser()
ap.add_argument("--gamma", type=float, default=1.5)
args = ap.parse_args()

print(f"gamma = {args.gamma}")
with open('results/check_model_results.csv', 'w',newline="") as f:
    writer = csv.DictWriter(f, fieldnames = ['instance','encoding', 'total_qubits', 'decision_qubits', 'aux_qubits', 'aux_pct', 'ground_state_ok', 'mean_physical_qubits', 'min_physical_qubits', 'max_chain_length'])
    writer.writeheader()
    print(f"{'#':>2} {'encoding':>10} {'total':>5} {'decision':>8} {'aux':>3} {'aux%':>4} {'GS ok':>5} {'mean pq':>7} {'min pq':>6} {'max chain':>9}")

    for i, inst in INSTANCES.items():
        x_star, _ = optimum(inst)
        for enc in ("paper", "compact", "unbalanced"):
            m = build_qubo(inst, enc, args.gamma)
            n_aux = count_bits(m.aux_vars)
            n_x = count_bits(m.x_vars)
            mean_pq, min_pq, max_chain, successful = embedding_cost(m.bqm)
            sample, _ = brute_force_ground_state(m)
            d = m.decode(sample)
            row = m.num_qubits
            ok = ("yes" if d["feasible"] and d["x"] == x_star else f"NO {d['x']}")
            print(
                f"{i:>2} {enc:>8} {row:>5} {n_x:>8} {n_aux:>3} {round(n_aux * 100 / row):>4} {ok:>5} {mean_pq:>7} {min_pq:>6} {max_chain:>9}")
            writer.writerow({
                'instance': i,
                'encoding': enc,
                'total_qubits': row,
                'decision_qubits': n_x,
                'aux_qubits': n_aux,
                'aux_pct': round(n_aux * 100 / row),
                'ground_state_ok': ok,
                'mean_physical_qubits': mean_pq,
                'min_physical_qubits': min_pq,
                'max_chain_length': max_chain})

