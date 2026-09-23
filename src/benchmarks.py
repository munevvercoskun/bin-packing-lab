"""Run the experiments, write charts to results/ and a summary to results/results.md.

    python -m src.benchmarks           # full run (a couple of minutes)
    python -m src.benchmarks --quick   # smaller inputs
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import instances
from src.exact import solve
from src.heuristics import HEURISTICS, first_fit, first_fit_decreasing, is_valid
from src.randomized import random_restarts

RESULTS = Path(__file__).resolve().parent.parent / "results"
MACHINE_COST_PER_MONTH = 70.00      # roughly a 4-vCPU cloud VM, list price


def save_chart(name, title, xlabel, ylabel, series, log=False, hline=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, xs, ys in series:
        ax.plot(xs, ys, marker="o", label=label)
    if hline is not None:
        ax.axhline(hline[0], color="grey", linestyle="--", linewidth=1, label=hline[1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- experiment 1
def exp_quality(sizes, kind):
    """How many machines does each heuristic use, relative to the lower bound?"""
    rows = {name: [] for name in HEURISTICS}
    rows["Random restarts (50)"] = []
    for n in sizes:
        inst = instances.GENERATORS[kind](n, seed=n)
        lb = inst.lower_bound()
        for name, fn in HEURISTICS.items():
            packing = fn(inst.items, inst.capacity)
            assert is_valid(packing, inst)
            rows[name].append(len(packing) / lb)
        rows["Random restarts (50)"].append(len(random_restarts(inst, tries=50)) / lb)
    save_chart(f"1_quality_{kind}", f"Machines used vs. lower bound ({kind} sizes)",
               "containers", "bins / lower bound",
               [(k, sizes, v) for k, v in rows.items()], hline=(1.0, "lower bound"))
    return rows


# ---------------------------------------------------------------- experiment 2
def exp_exact_blowup(sizes, node_limit):
    """Time and search nodes for the exact solver as n grows -- NP-hardness, measured."""
    times, nodes, solved = [], [], []
    for n in sizes:
        inst = instances.awkward(n, seed=n)
        start = time.perf_counter()
        best, visited = solve(inst, node_limit=node_limit)
        times.append(time.perf_counter() - start)
        nodes.append(visited)
        solved.append(best is not None)
    save_chart("2_exact_blowup", "Exact branch and bound: cost of proving the optimum",
               "containers", "search nodes (log scale)",
               [("nodes explored", sizes, nodes)], log=True,
               hline=(node_limit, "node limit — search gave up"))
    return times, nodes, solved


# ---------------------------------------------------------------- experiment 3
def exp_gap_to_optimum(n, trials, node_limit):
    """On instances small enough to solve exactly, how far off are the heuristics?"""
    counts = {name: [] for name in ("First fit", "First fit decreasing")}
    optima, lbs = [], []
    for t in range(trials):
        inst = instances.bimodal(n, seed=1000 + t)
        best, _ = solve(inst, node_limit=node_limit)
        if best is None:
            continue                      # skip instances we could not prove
        optima.append(best)
        lbs.append(inst.lower_bound())
        counts["First fit"].append(len(first_fit(inst.items, inst.capacity)))
        counts["First fit decreasing"].append(len(first_fit_decreasing(inst.items, inst.capacity)))
    out = {}
    for name, values in counts.items():
        out[name] = {
            "mean ratio": sum(v / o for v, o in zip(values, optima)) / len(values),
            "optimal on": sum(v == o for v, o in zip(values, optima)),
            "worst ratio": max(v / o for v, o in zip(values, optima)),
        }
    out["instances solved"] = len(optima)
    out["lower bound = optimum on"] = sum(l == o for l, o in zip(lbs, optima))
    return out


# ---------------------------------------------------------------- experiment 4
def exp_restarts(n, budgets):
    """Does more random searching help? (Monte Carlo, diminishing returns.)"""
    inst = instances.bimodal(n, seed=7)
    lb = inst.lower_bound()
    ratios = [len(random_restarts(inst, tries=b)) / lb for b in budgets]
    pure = [len(random_restarts(inst, tries=b, start_from_ffd=False)) / lb for b in budgets]
    ffd = len(first_fit_decreasing(inst.items, inst.capacity)) / lb
    save_chart("3_random_restarts", f"Random restarts on {n} containers",
               "random orders tried", "bins / lower bound",
               [("best of k random orders", budgets, pure),
                ("same, but seeded with FFD", budgets, ratios),
                ("first fit decreasing alone", budgets, [ffd] * len(budgets))])
    return budgets, ratios, pure, ffd


# ---------------------------------------------------------------- experiment 5
def exp_cost(n):
    """Translate bins into money, which is what the packing is really about."""
    inst = instances.bimodal(n, seed=99)
    rows = {name: len(fn(inst.items, inst.capacity)) for name, fn in HEURISTICS.items()}
    rows["Random restarts (200)"] = len(random_restarts(inst, tries=200))
    rows["Lower bound (unreachable in general)"] = inst.lower_bound()
    return inst, rows


# ---------------------------------------------------------------- report
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.quick:
        sizes, exact_sizes, node_limit = [100, 250, 500], list(range(8, 19)), 50_000
        trials, fleet, budgets = 15, 300, [1, 5, 10, 25, 50]
    else:
        sizes, exact_sizes, node_limit = [100, 250, 500, 1000, 2000], list(range(8, 25)), 300_000
        trials, fleet, budgets = 40, 1000, [1, 5, 10, 25, 50, 100, 200]

    print("1/5 heuristic quality ...")
    q = {kind: exp_quality(sizes, kind) for kind in ("uniform", "bimodal", "awkward")}
    print("2/5 exact solver blow-up ...")
    times, nodes, solved = exp_exact_blowup(exact_sizes, node_limit)
    print("3/5 gap to true optimum ...")
    gap = exp_gap_to_optimum(16, trials, node_limit)
    print("4/5 random restarts ...")
    budgets, ratios, pure, ffd = exp_restarts(fleet, budgets)
    print("5/5 cost ...")
    inst, cost_rows = exp_cost(fleet)

    out = ["# Results", "",
           "Generated by `python -m src.benchmarks"
           f"{' --quick' if args.quick else ''}`. Machine capacity 4000 CPU units "
           f"(4 vCPU); machine cost ${MACHINE_COST_PER_MONTH:.0f}/month.", ""]

    for kind, rows in q.items():
        out += [f"### 1. Machines used / lower bound — {kind} sizes", "",
                "| containers | " + " | ".join(rows) + " |",
                "|---" * (len(rows) + 1) + "|"]
        for i, n in enumerate(sizes):
            out.append(f"| {n:,} | " + " | ".join(f"{v[i]:.3f}" for v in rows.values()) + " |")
        out.append("")

    out += ["### 2. Exact branch and bound as n grows", "",
            "| containers | nodes explored | seconds | optimum proven? |", "|---|---|---|---|"]
    for n, node, t, ok in zip(exact_sizes, nodes, times, solved):
        out.append(f"| {n} | {node:,} | {t:.3f} | {'yes' if ok else 'NO — hit the node limit'} |")
    out.append("")

    out += ["### 3. How far from the true optimum? (16 containers, bimodal)", "",
            f"Instances proven optimal: {gap['instances solved']} of {trials}. "
            f"The L1 lower bound equalled the optimum on {gap['lower bound = optimum on']} of them.", "",
            "| heuristic | mean bins / optimum | hit the optimum | worst case |", "|---|---|---|---|"]
    for name in ("First fit", "First fit decreasing"):
        g = gap[name]
        out.append(f"| {name} | {g['mean ratio']:.3f} | "
                   f"{g['optimal on']}/{gap['instances solved']} | {g['worst ratio']:.3f} |")
    out.append("")

    out += [f"### 4. Random restarts ({fleet} containers)", "",
            "| random orders tried | random only | seeded with FFD |", "|---|---|---|"]
    for b, r, pr in zip(budgets, ratios, pure):
        out.append(f"| {b} | {pr:.4f} | {r:.4f} |")
    out += ["", f"First fit decreasing alone: {ffd:.4f}.", ""]

    base = cost_rows["Next fit"]
    out += [f"### 5. What it costs — fleet of {fleet} containers", "",
            "| strategy | machines | $/month | saved vs. next fit |", "|---|---|---|---|"]
    for name, count in cost_rows.items():
        out.append(f"| {name} | {count} | {count * MACHINE_COST_PER_MONTH:,.0f} | "
                   f"{(base - count) * MACHINE_COST_PER_MONTH:,.0f} |")
    out.append("")

    (RESULTS / "results.md").write_text("\n".join(out))
    print(f"Done. Charts and results.md written to {RESULTS}")


if __name__ == "__main__":
    main()
