"""Problem instances: a list of container sizes and one machine capacity.

Sizes are in "CPU units" -- think 1000 = 1 vCPU, machine capacity 4000 = 4 vCPU.
"""
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    capacity: int
    items: tuple            # container sizes
    name: str = ""

    @property
    def total(self):
        return sum(self.items)

    def lower_bound(self):
        """No packing can use fewer bins than this (L1 bound).

        You cannot fit more than `capacity` units of work on one machine, so
        you need at least total/capacity machines, rounded up. It costs O(n)
        and gives something to measure heuristics against when the true
        optimum is unknown.
        """
        return -(-self.total // self.capacity)   # ceiling division


def uniform(n, capacity=4000, low=200, high=2000, seed=None):
    """Container sizes spread evenly -- the textbook case."""
    rng = random.Random(seed)
    return Instance(capacity, tuple(rng.randint(low, high) for _ in range(n)), "uniform")


def bimodal(n, capacity=4000, seed=None):
    """Mostly small services, a few large ones -- closer to a real fleet."""
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        if rng.random() < 0.8:
            items.append(rng.randint(100, 600))     # small services
        else:
            items.append(rng.randint(1800, 3200))   # databases, ML workers
    return Instance(capacity, tuple(items), "bimodal")


def awkward(n, capacity=4000, seed=None):
    """Sizes just over capacity/3, so only two fit per machine and 1/3 is wasted.

    This is the classic adversarial shape for first-fit: it shows that a
    heuristic's average case tells you nothing about its worst case.
    """
    rng = random.Random(seed)
    base = capacity // 3 + 1
    return Instance(capacity, tuple(base + rng.randint(0, 80) for _ in range(n)), "awkward")


GENERATORS = {"uniform": uniform, "bimodal": bimodal, "awkward": awkward}
