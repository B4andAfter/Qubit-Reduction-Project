# Reducing Qubit Requirements for Quantum Optimization of the Redundancy Allocation Problem

Code and data for the course project *"Reducing Qubit Requirements for Quantum
Optimization of the Redundancy Allocation Problem"*. It reimplements the QUBO
formulation of Lins et al. (2025) and compares it with two qubit-reduced
encodings, using QAOA, simulated annealing and a random-sampling baseline.

## Main results

| | Paper encoding | Compact encoding | Unbalanced encoding |
| --- | --- | --- | --- |
| Logical qubits | 15–19 | 6–12 | 4–9 |
| Estimated physical qubits (Pegasus P16) | 28–49 | 7–24 | 4–14 |
| Ground state = optimum | 15 / 15 | 15 / 15 | 13 / 15 (tuned λ) |
| QAOA, optimal runs (of 450) | 70 (16%) | 333 (74%) | 364 (81%) |
| Random sampling, optimal runs (of 450) | 56 (12%) | 360 (80%) | 380 (84%) |

- The **compact** encoding is exact: tighter variable bounds, bounded-binary
  coefficients and one merged cardinality constraint.
- The **unbalanced** encoding replaces the budget slack with
  −λ₁h + λ₂h² (Montañez-Barrera et al., 2024). λ₁ = λ₂ = 0.75 was tuned on
  50 generated training instances, not on the 15 test instances.
- Higher QAOA success comes from the smaller search space: random sampling
  does just as well, and depth-1 QAOA's final state assigns the optimum *less*
  probability than uniform guessing. Simulated annealing samples it 2–237×
  more often than random.

## Requirements

- **Python 3.12.** Python 3.14 has no prebuilt wheels for `qiskit-aer`.
- Qiskit 1.x, not 2.x (`qiskit-algorithms` 0.3 needs 1.x).
- No quantum hardware or D-Wave account is needed. Everything runs on a
  laptop.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the exact versions used for the results.

## Files

| File | Purpose |
| --- | --- |
| `instances.py` | The 15 benchmark instances (Table 4 of Lins et al., 2025) |
| `instance_generator.py` | Random instances with the paper's rules, for λ tuning |
| `classical.py` | Exact solutions: exhaustive search and Pyomo + HiGHS (branch-and-cut) |
| `qubo_builder.py` | QUBO construction (`paper`, `compact`, `unbalanced`), decoding, qubit counts, embedding, brute-force check |
| `check_models.py` | Qubit counts, physical qubits, ground-state check (Table: qubit requirements) |
| `tune_lambdas.py` | Grid search for λ₁, λ₂ on training instances, then test on the 15 |
| `run_gate.py` | QAOA (and VQE) on the Qiskit Aer simulator, incl. P(opt) |
| `run_annealing.py` | Simulated annealing (dwave-neal); optional real D-Wave QPU |
| `run_random.py` | Random-sampling baseline with the same best-of-N rule as QAOA |
| `results/` | All CSV result files used in the report |

## Reproducing the results

Every command writes a CSV to `results/` (or to the folder given with `--out`).

```bash
# Exact optima of all 15 instances
python classical.py

# Qubit requirements and ground-state check
python check_models.py
python check_models.py --lambda_1 0.75 --lambda_2 0.75     # unbalanced, tuned

# Penalty tuning (50 training instances, seed 2026)
python tune_lambdas.py

# QAOA, 30 runs per instance
python run_gate.py --algo qaoa --opt cobyla --encoding paper   --instances 1-15 --reps 30
python run_gate.py --algo qaoa --opt cobyla --encoding compact --instances 1-15 --reps 30
python run_gate.py --algo qaoa --opt cobyla --encoding unbalanced --lambda_1 0.75 --lambda_2 0.75 --instances 1-15 --reps 30 --out results/lam_0.75

# Simulated annealing, 1000 reads per instance
python run_annealing.py --sampler sa --encoding paper   --instances 1-15
python run_annealing.py --sampler sa --encoding compact --instances 1-15
python run_annealing.py --sampler sa --encoding unbalanced --lambda_1 0.75 --lambda_2 0.75 --instances 1-15 --out results/lam_0.75

# Random-sampling baseline
python run_random.py --encoding paper   --instances 1-15 --reps 30 --out results/random
python run_random.py --encoding compact --instances 1-15 --reps 30 --out results/random
python run_random.py --encoding unbalanced --lambda_1 0.75 --lambda_2 0.75 --instances 1-15 --reps 30 --out results/random
```

Repeat the unbalanced runs with `--lambda_1 1 --lambda_2 1`, `1.5` and `2`
(output folders `results/lam_1`, `lam_1.5`, `lam_2`) for the λ-sensitivity tables.

## Settings and seeds

| Setting | Value |
| --- | --- |
| QAOA | depth p = 1, COBYLA, initial angles uniform in [0, 2π] |
| Sampling | Aer sampler, 1000 shots |
| Penalty weight γ | 1.5 (all encodings) |
| Unbalanced λ₁ = λ₂ | 0.75 (also 1, 1.5, 2 for sensitivity) |
| Simulated annealing | dwave-neal, 1000 reads |
| Embedding | Pegasus P16 (`dwave_networkx.pegasus_graph(16)`), minorminer, seeds 1–10 |
| Seeds | runs 10598 · training instances 2026 · embeddings 1–10 |

## Results folder

| Path | Content |
| --- | --- |
| `results/check_model_results.csv` | Qubits, aux share, physical qubits, ground-state check |
| `results/lambda_grid_train.csv` | λ grid: correct ground states out of 50 training instances |
| `results/qaoa_cobyla_p1_{paper,compact}.csv` | QAOA runs |
| `results/lam_*/` | Unbalanced QAOA and SA runs for each λ pair |
| `results/anneal_sa_{paper,compact}.csv` | Simulated annealing |
| `results/random/` | Random-sampling baseline |
| `results/popt/` | QAOA runs with P(opt) |

## Reference

Lins, I.D., Araújo, L.M.M., Souto Maior, C.B., Teixeira, E.S., Bezerra, P.T.L.,
Moura, M.J.C., Droguett, E.L. (2025). Quantum-based optimization methods for the
linear redundancy allocation problem: A comparative analysis. *Reliability
Engineering & System Safety*, 262, 111153.
https://doi.org/10.1016/j.ress.2025.111153

## License

MIT. See `LICENSE`.
