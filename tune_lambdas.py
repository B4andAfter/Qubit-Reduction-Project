
import numpy as np
import csv

from instance_generator import generate_instance
from instances import Instance
from classical import exhaustive
from qubo_builder import brute_force_ground_state,build_qubo

def is_correct(inst,l1,l2):
    m = build_qubo(inst,"unbalanced",lambda_1=l1,lambda_2=l2)
    sample, _ = brute_force_ground_state(m)
    d = m.decode(sample)
    G_star = exhaustive(inst)[1]
    return d["feasible"] and abs(d["G"] - G_star) < 1e-9

rng = np.random.default_rng(2026)
train = [generate_instance(rng, 1000 + j) for j in range(50)]
L1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1, 1.25, 1.5, 2]
L2 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2]
scores = {}

for l1 in L1:
    for l2 in L2:
        count = 0
        for inst in train:
            if is_correct(inst,l1,l2):
                count += 1
        scores[(l1,l2)] = count
    print(f"done l1= {l1}")



with open("results/lambda_grid_train.csv", "w", newline="") as f:
    writer = csv.DictWriter(f,["lambda_1","lambda_2", "correct_of_50"])
    writer.writeheader()
    for (l1,l2), count in scores.items():
        writer.writerow({
            "lambda_1": l1,
            "lambda_2": l2,
            "correct_of_50": count
        })