"""The 15 test instances from Lins et al. (2025), Table 4, plus the
optimal G values from Table 6 (for reference only; we recompute them).

Each instance is a single active parallel system:
    R[k]  reliability of component type k
    c[k]  cost of component type k
    C     budget
    n_min, n_max  min / max total number of components
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    idx: int
    n_max: int
    C: int
    R: tuple
    c: tuple
    n_min: int = 1

    @property
    def ct(self) -> int:
        return len(self.R)


_TABLE4 = [
    # (n_max, C, R, c)
    (2, 9,  (0.8486, 0.7645, 0.9860), (5, 4, 6)),
    (2, 6,  (0.6086, 0.9860, 0.6999), (3, 6, 5)),
    (2, 5,  (0.9287, 0.5632, 0.7523), (6, 2, 3)),
    (2, 4,  (0.8248, 0.7704, 0.8028), (4, 1, 3)),
    (2, 7,  (0.9492, 0.6906, 0.6504, 0.8225), (6, 3, 1, 4)),
    (2, 9,  (0.8283, 0.9280, 0.6983, 0.7605), (5, 6, 2, 3)),
    (2, 6,  (0.6827, 0.8870, 0.9025, 0.7970), (1, 4, 5, 3)),
    (2, 6,  (0.8871, 0.8101, 0.5876, 0.8661), (3, 2, 2, 3)),
    (2, 9,  (0.8807, 0.7835, 0.6523, 0.8950), (5, 4, 2, 5)),
    (2, 7,  (0.8691, 0.7165, 0.6363, 0.9265), (5, 4, 3, 6)),
    (2, 11, (0.7771, 0.9424, 0.6021, 0.8328), (3, 6, 2, 6)),
    (2, 9,  (0.8179, 0.6679, 0.6394, 0.9884), (5, 4, 3, 6)),
    (3, 4,  (0.8761, 0.9117, 0.7471), (3, 4, 2)),
    (2, 5,  (0.8394, 0.6332, 0.7882, 0.9410, 0.9865), (2, 1, 1, 4, 4)),
    (2, 7,  (0.9844, 0.7781, 0.6301, 0.6782, 0.6734), (6, 4, 2, 3, 2)),
]

INSTANCES = {i + 1: Instance(i + 1, *row) for i, row in enumerate(_TABLE4)}
