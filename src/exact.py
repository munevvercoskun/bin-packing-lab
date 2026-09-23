"""Exact optimum by branch and bound -- and a demonstration of why you can't
use it at scale.

The search places items one at a time, trying every bin the item fits in plus
one fresh bin. Unpruned that is exponential. Four cuts keep it alive on small
inputs:

  1. start from the first-fit-decreasing answer, so there's a good bound immediately,
  2. abandon a branch once it already uses as many bins as the best answer so far,
  3. try only ONE empty bin per item (empty bins are interchangeable) and skip a bin
     whose load equals one already tried at this step -- both are symmetry breaking,
  4. stop the instant the L1 lower bound is reached, since nothing can beat it.

`node_limit` makes the blow-up visible instead of hanging: when the search runs
out of budget it returns None, meaning "no proven optimum in this budget".
"""
from src.heuristics import first_fit_decreasing


def solve(instance, node_limit=300_000):
    items = sorted(instance.items, reverse=True)   # place big items first
    capacity = instance.capacity
    target = instance.lower_bound()

    best_count = [len(first_fit_decreasing(items, capacity))]
    best_packing = [None]
    nodes = [0]

    def search(i, loads, packing):
        """Returns False if the node budget ran out, True otherwise."""
        if nodes[0] >= node_limit:
            return False
        nodes[0] += 1

        if i == len(items):
            if len(loads) < best_count[0]:
                best_count[0] = len(loads)
                best_packing[0] = [list(b) for b in packing]
            return True
        if len(loads) >= best_count[0]:
            return True

        size = items[i]
        tried_loads = set()
        for b in range(len(loads)):
            if loads[b] + size > capacity or loads[b] in tried_loads:
                continue
            tried_loads.add(loads[b])
            loads[b] += size
            packing[b].append(size)
            ok = search(i + 1, loads, packing)
            loads[b] -= size
            packing[b].pop()
            if not ok:
                return False
            if best_count[0] == target:
                return True

        if len(loads) + 1 < best_count[0]:      # open one fresh bin
            loads.append(size)
            packing.append([size])
            ok = search(i + 1, loads, packing)
            loads.pop()
            packing.pop()
            if not ok:
                return False
        return True

    finished = search(0, [], [])
    if not finished:
        return None, nodes[0]          # budget exhausted: optimum unproven
    return best_count[0], nodes[0]
