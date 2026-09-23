"""Greedy heuristics. Each returns a list of bins; each bin is a list of item sizes.

All of them are O(n log n) or better, and none of them is guaranteed optimal --
that's the point. Bin packing is NP-hard, so these buy speed by giving up the
guarantee, and we measure what that costs.
"""


def next_fit(items, capacity):
    """Keep one open bin; when the item doesn't fit, close it and open a new one.

    O(n), and never revisits a closed bin -- which is why it wastes so much.
    Worst case: 2x the optimal number of bins.
    """
    bins, current, load = [], [], 0
    for size in items:
        if load + size > capacity:
            bins.append(current)
            current, load = [], 0
        current.append(size)
        load += size
    if current:
        bins.append(current)
    return bins


def first_fit(items, capacity):
    """Put each item in the first bin it fits in. Worst case: 1.7 x optimal."""
    bins, loads = [], []
    for size in items:
        for i, load in enumerate(loads):
            if load + size <= capacity:
                bins[i].append(size)
                loads[i] += size
                break
        else:                       # no bin had room
            bins.append([size])
            loads.append(size)
    return bins


def best_fit(items, capacity):
    """Put each item in the bin that will have the least room left afterwards.

    Same 1.7 worst-case ratio as first fit, usually a little better in practice.
    """
    bins, loads = [], []
    for size in items:
        best, best_slack = None, capacity + 1
        for i, load in enumerate(loads):
            slack = capacity - load - size
            if 0 <= slack < best_slack:
                best, best_slack = i, slack
        if best is None:
            bins.append([size])
            loads.append(size)
        else:
            bins[best].append(size)
            loads[best] += size
    return bins


def first_fit_decreasing(items, capacity):
    """Sort largest-first, then first fit. Guaranteed within (11/9)·OPT + 6/9.

    Sorting first is the whole trick: big items are placed while bins are still
    empty, and the small ones fill the gaps afterwards.
    """
    return first_fit(sorted(items, reverse=True), capacity)


def best_fit_decreasing(items, capacity):
    return best_fit(sorted(items, reverse=True), capacity)


HEURISTICS = {
    "Next fit": next_fit,
    "First fit": first_fit,
    "Best fit": best_fit,
    "First fit decreasing": first_fit_decreasing,
    "Best fit decreasing": best_fit_decreasing,
}


def is_valid(bins, instance):
    """Every item placed exactly once, and no machine over capacity."""
    packed = sorted(s for b in bins for s in b)
    return packed == sorted(instance.items) and all(sum(b) <= instance.capacity for b in bins)
