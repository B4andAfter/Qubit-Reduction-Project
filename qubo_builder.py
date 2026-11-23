"""Build the RAP QUBO (Lins et al. 2025, Section 3) as a dimod BQM.

Two encodings:

  "paper"   -- exactly as in the paper (Eq. 23):
               * x_k written with nb = bits(n_max) powers of two
               * budget slack sl1  in [0, C]
               * excess ex          in [0, ct*(2^nb - 1) - n_min]
               * slack sl2          in [0, n_max]
               * three squared penalties
               -> reproduces Table 10 qubit counts (15-19 qubits)

  "compact" -- a qubit-reduced version of the same model:
               * x_k upper bound u_k = min(n_max, floor(C / c_k)); types that can
                 never fit the budget are dropped
               * "bounded binary" coefficients (1, 2, 4, ..., U - (2^(m-1) - 1)) so
                 every variable reaches EXACTLY its bound, no overshoot states
               * the two count constraints n_min <= sum x <= n_max merged into
                 one equality  sum x - t = n_min,  t in [0, n_max - n_min]
               * budget slack in [0, C - n_min * min(c)]
               -> same optimum, fewer qubits (see check_models.py)

Penalty form: gamma * (lhs - rhs)^2, added with dimod's
add_linear_equality_constraint (it expands the square for you; b^2 = b for bits).
"""
from dataclasses import dataclass, field
import math

import dimod
import numpy as np
import minorminer
import dwave_networkx as dnx

from instances import Instance


# ---------------------------------------------------------------- encodings
def n_bits(v: int) -> int:
    """Number of binary digits of v (the paper's Bin procedure). Bin(0) -> 0."""
    return int(v).bit_length()


def power_coeffs(upper: int) -> list:
    """Paper encoding: plain powers of two, enough bits to write `upper`."""
    return [2 ** i for i in range(n_bits(upper))]


def bounded_coeffs(upper: int) -> list:
    """Bounded binary encoding: max representable value == upper exactly."""
    if upper <= 0:
        return []
    m = n_bits(upper)
    return [2 ** i for i in range(m - 1)] + [upper - (2 ** (m - 1) - 1)]


# ---------------------------------------------------------------- model
@dataclass
class RAPQubo:
    inst: Instance
    encoding: str
    gamma: float
    bqm: dimod.BinaryQuadraticModel
    x_vars: dict = field(default_factory=dict)      # k -> [(var, coeff), ...]
    aux_vars: dict = field(default_factory=dict)    # "sl1"/"ex"/... -> [(var, coeff)]


    @property
    def num_qubits(self) -> int:
        return self.bqm.num_variables

    def decode(self, sample: dict) -> dict:
        """Turn a bit assignment into a design and check it."""
        inst = self.inst
        x = [0] * inst.ct
        for k, terms in self.x_vars.items():
            x[k] = int(sum(a * int(sample[v]) for v, a in terms))
        cost = sum(ck * xk for ck, xk in zip(inst.c, x))
        count = sum(x)
        feasible = cost <= inst.C and inst.n_min <= count <= inst.n_max
        G = sum(xk * math.log(1 - rk) for xk, rk in zip(x, inst.R))   # ln F
        energy = self.bqm.energy(sample)
        # penalty > 0 means some equality is broken. If the design x is feasible,
        # the culprit is a wrong slack/excess bit (fixable classically afterwards).
        penalty = energy - G
        return dict(x=tuple(x), cost=cost, count=count, feasible=feasible,
                    G=G, R_sys=1 - math.exp(G), energy=energy,
                    penalty=penalty, slacks_ok=abs(penalty) < 1e-6)


def build_qubo(inst: Instance, encoding: str = "paper", gamma: float = 1.5, lambda_1=1.0, lambda_2=0.1) -> RAPQubo:
    bqm = dimod.BinaryQuadraticModel("BINARY")
    model = RAPQubo(inst, encoding, gamma, bqm)

    def new_vars(prefix, coeffs):
        names = [f"{prefix}_{i}" for i in range(len(coeffs))]
        for n in names:
            bqm.add_variable(n, 0.0)
        return list(zip(names, coeffs))

    # --- decision variables x_k and the linear objective G (Eq. 10)
    for k in range(inst.ct):
        if encoding == "paper":
            coeffs = power_coeffs(inst.n_max)
        elif encoding in ("compact","unbalanced"):
            coeffs = bounded_coeffs(min(inst.n_max, inst.C // inst.c[k]))
        else:
            raise ValueError(encoding)
        if not coeffs:                      # type can never be installed
            continue
        terms = new_vars(f"x{k}", coeffs)
        model.x_vars[k] = terms
        for v, a in terms:
            bqm.add_linear(v, a * math.log(1 - inst.R[k]))

    x_terms = [(v, a) for terms in model.x_vars.values() for v, a in terms]
    cost_terms = [(v, a * inst.c[k]) for k, terms in model.x_vars.items() for v, a in terms]
    enc = power_coeffs if encoding == "paper" else bounded_coeffs

    def penalty(terms, rhs):
        if terms:
            bqm.add_linear_equality_constraint(terms, lagrange_multiplier=gamma, constant=-rhs)

    if encoding == "paper":
        # Eq. 17: sum c_k x_k + sl1 = C
        sl1 = new_vars("sl1", enc(inst.C))
        # Eq. 18: sum x_k - ex = n_min      (ex_max uses ALL types, see notes)
        ex_max = inst.ct * (2 ** n_bits(inst.n_max) - 1) - inst.n_min
        ex = new_vars("ex", enc(ex_max))
        # Eq. 19: sum x_k + sl2 = n_max
        sl2 = new_vars("sl2", enc(inst.n_max))
        model.aux_vars = {"sl1": sl1, "ex": ex, "sl2": sl2}
        penalty(cost_terms + sl1, inst.C)
        penalty(x_terms + [(v, -a) for v, a in ex], inst.n_min)
        penalty(x_terms + sl2, inst.n_max)
    elif encoding == "unbalanced":
        t = new_vars("t", enc(inst.n_max - inst.n_min))
        bqm.add_linear_equality_constraint(cost_terms, lagrange_multiplier=lambda_2,constant= -1*inst.C) #lambda_2*h^2
        for v,a in cost_terms:
            bqm.add_linear(v, lambda_1*a)
        bqm.offset -= lambda_1*inst.C
        penalty(x_terms + [(v, -a) for v, a in t], inst.n_min)  # n_min <= sum x <= n_max
        model.aux_vars = {"t": t}
    elif encoding == "compact":
        sl1 = new_vars("sl1", enc(max(0, inst.C - inst.n_min * min(inst.c))))
        t = new_vars("t", enc(inst.n_max - inst.n_min))
        model.aux_vars = {"sl1": sl1, "t": t}
        penalty(cost_terms + sl1, inst.C)                       # budget
        penalty(x_terms + [(v, -a) for v, a in t], inst.n_min)  # n_min <= sum x <= n_max
    return model


# ---------------------------------------------------------------- helpers
def brute_force_ground_state(model: RAPQubo, max_vars: int = 24):
    """Exact minimum of the QUBO by enumeration (fine up to ~24 bits)."""
    bqm = model.bqm
    n = bqm.num_variables
    if n > max_vars:
        raise ValueError(f"{n} variables is too many to enumerate")
    order = list(bqm.variables)
    lin = np.array([bqm.get_linear(v) for v in order])
    Q = np.zeros((n, n))
    idx = {v: i for i, v in enumerate(order)}
    for (u, v), b in bqm.quadratic.items():
        Q[idx[u], idx[v]] = b
    best_e, best_bits = math.inf, None
    chunk = 1 << min(n, 18)
    for start in range(0, 1 << n, chunk):
        ints = np.arange(start, start + chunk, dtype=np.int64)
        X = ((ints[:, None] >> np.arange(n)) & 1).astype(float)
        E = X @ lin + np.einsum("ij,jk,ik->i", X, Q, X) + bqm.offset
        i = int(np.argmin(E))
        if E[i] < best_e:
            best_e, best_bits = E[i], X[i]
    sample = {v: int(b) for v, b in zip(order, best_bits)}
    return sample, best_e


def to_quadratic_program(model: RAPQubo):
    """Same QUBO as a Qiskit QuadraticProgram (all-binary, unconstrained)."""
    from qiskit_optimization import QuadraticProgram

    bqm = model.bqm
    qp = QuadraticProgram(f"rap_{model.inst.idx}_{model.encoding}")
    for v in bqm.variables:
        qp.binary_var(v)
    qp.minimize(constant=bqm.offset,
                linear=dict(bqm.linear),
                quadratic={(u, v): b for (u, v), b in bqm.quadratic.items()})
    return qp

def count_bits(vars):
    """Count the number of bits in a variable."""
    n = 0
    for key in vars.keys():
        n+=len(vars[key])
    return n

def embedding_cost(bqm, seeds=range(1,11)):
    chip = dnx.pegasus_graph(16)
    i = 0
    max_chain = 1
    min_pq = float("inf")
    sum_pq = 0
    for s in seeds:
        l_qubits =  minorminer.find_embedding(list(bqm.quadratic), chip, random_seed=s)
        if len(l_qubits) > 0:
            i+=1
            n = 0
            for val in l_qubits.values():
                chain = len(val)
                if chain>max_chain:
                    max_chain = chain
                n += chain
            if n < min_pq:
                min_pq = n
            sum_pq += n
            n=0
    mean_pq = sum_pq / i
    return mean_pq, min_pq, max_chain, i


