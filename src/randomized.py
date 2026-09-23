"""Randomized search: shuffle, pack, keep the best. A Monte Carlo method --
fixed budget, no guarantee of the optimum, but it usually beats plain first fit.
"""
import random

from src.heuristics import first_fit, first_fit_decreasing


def random_restarts(instance, tries=50, seed=0, start_from_ffd=True):
    """Try `tries` random orders through first fit and keep the best result.

    With start_from_ffd=True the answer can never be worse than FFD -- a cheap
    way to keep FFD's guarantee while looking for something better. Setting it
    to False shows what random search achieves on its own.
    """
    rng = random.Random(seed)
    items = list(instance.items)
    best = (first_fit_decreasing(items, instance.capacity) if start_from_ffd
            else first_fit(items, instance.capacity))
    for _ in range(tries):
        rng.shuffle(items)
        candidate = first_fit(items, instance.capacity)
        if len(candidate) < len(best):
            best = candidate
    return best
