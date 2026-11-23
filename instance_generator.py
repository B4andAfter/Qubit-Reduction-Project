from instances import Instance
import numpy as np

def generate_instance(rng, idx):
    ct=int(rng.choice([3,4,5]))
    n_max=int(rng.choice([2,3]))
    R = rng.uniform(0.55,0.99,ct)
    R = np.round(R,4)
    c=rng.integers(1,7,ct)
    R = np.sort(R)
    c = np.sort(c)
    p = rng.permutation(ct)
    R = R[p]
    c = c[p]
    C = int(round(rng.uniform(0.4,0.7)*sum(c)))
    return Instance(idx=idx,
                    n_max=n_max,
                    C=C,
                    R=tuple(float(r) for r in R),
                    c = tuple(int(x) for x in c)
    )

