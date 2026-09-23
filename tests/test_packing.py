import pytest

from src import instances
from src.exact import solve
from src.heuristics import (HEURISTICS, best_fit, first_fit, first_fit_decreasing,
                            is_valid, next_fit)
from src.randomized import random_restarts

KINDS = ["uniform", "bimodal", "awkward"]


def brute_force(instance):
    """Complete search with no pruning. Only usable for tiny inputs -- it is
    here as an independent check that the pruned solver isn't cutting too much."""
    items = list(instance.items)
    best = [len(items)]

    def place(i, loads):
        if len(loads) >= best[0]:
            return
        if i == len(items):
            best[0] = len(loads)
            return
        for b in range(len(loads)):
            if loads[b] + items[i] <= instance.capacity:
                loads[b] += items[i]
                place(i + 1, loads)
                loads[b] -= items[i]
        loads.append(items[i])
        place(i + 1, loads)
        loads.pop()

    place(0, [])
    return best[0]


@pytest.mark.parametrize("kind", KINDS)
def test_every_item_fits_in_one_machine(kind):
    inst = instances.GENERATORS[kind](200, seed=3)
    assert max(inst.items) <= inst.capacity    # otherwise the problem is unsolvable


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", list(HEURISTICS))
def test_packings_are_valid(kind, name):
    inst = instances.GENERATORS[kind](300, seed=5)
    packing = HEURISTICS[name](inst.items, inst.capacity)
    assert is_valid(packing, inst)


@pytest.mark.parametrize("kind", KINDS)
def test_lower_bound_never_exceeds_any_real_packing(kind):
    inst = instances.GENERATORS[kind](150, seed=11)
    assert inst.lower_bound() <= len(first_fit_decreasing(inst.items, inst.capacity))


def test_exact_solver_agrees_with_brute_force():
    for seed in range(8):
        inst = instances.uniform(9, seed=seed)
        best, _ = solve(inst)
        assert best == brute_force(inst)


def test_exact_solver_reports_failure_instead_of_hanging():
    inst = instances.awkward(30, seed=1)
    best, nodes = solve(inst, node_limit=5_000)
    assert best is None            # no proven optimum inside the budget
    assert nodes >= 5_000


def test_ffd_respects_its_approximation_guarantee():
    """FFD uses at most (11/9)·OPT + 6/9 machines. Checked against the real optimum."""
    for seed in range(12):
        inst = instances.bimodal(14, seed=seed)
        opt, _ = solve(inst)
        used = len(first_fit_decreasing(inst.items, inst.capacity))
        assert used <= (11 / 9) * opt + 6 / 9


def test_next_fit_stays_within_twice_optimal():
    for seed in range(8):
        inst = instances.uniform(12, seed=seed)
        opt, _ = solve(inst)
        assert len(next_fit(inst.items, inst.capacity)) <= 2 * opt


def test_random_restarts_never_worse_than_ffd():
    inst = instances.bimodal(400, seed=2)
    ffd = len(first_fit_decreasing(inst.items, inst.capacity))
    assert len(random_restarts(inst, tries=20)) <= ffd


def test_sorting_is_what_makes_ffd_better():
    """On a case built for it, plain first fit needs more machines than FFD."""
    inst = instances.Instance(capacity=10, items=(4, 5, 4, 5, 4, 5, 6, 6, 6), name="demo")
    assert len(first_fit(inst.items, inst.capacity)) >= len(
        first_fit_decreasing(inst.items, inst.capacity))


def test_best_fit_leaves_less_slack_than_next_fit():
    inst = instances.bimodal(500, seed=4)
    assert len(best_fit(inst.items, inst.capacity)) <= len(next_fit(inst.items, inst.capacity))
