"""Exact class-specific threshold selection under one shared false-box budget."""
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

import numpy as np

from steel.metrics import concatenate
from .head_calibration import NO_DETECTIONS


@dataclass(frozen=True)
class Option:
    tp: int
    fp: int
    cutoff: float


def class_frontier(scores, hits, budget):
    """Keep nondominated prefixes of whole score ties, including an empty prefix."""
    options = [Option(0, 0, NO_DETECTIONS)]
    if not len(scores):
        return options
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    counts = np.cumsum(hits, dtype=np.int64)[ends]
    for end, count in zip(ends, counts, strict=True):
        fp = int(end + 1 - count)
        if fp > budget:
            break
        if count <= options[-1].tp:
            continue
        point = Option(int(count), fp, float(scores[end]))
        if fp == options[-1].fp:
            options[-1] = point
        else:
            options.append(point)
    return options


def fit_vectors(items, budget):
    scores, labels, flags, counts = concatenate(items)
    gt = [int(n) for n in counts]
    if min(gt) <= 0:
        raise ValueError("Threshold fitting needs annotations in every class.")
    frontiers = [class_frontier(scores[labels == c], flags[labels == c, 0], budget)
                 for c in range(len(gt))]
    # Integer weights give exact orderings of macro recalls, including ties.
    common = lcm(*gt)
    weights = [common // n for n in gt]
    levels = sorted({Fraction(o.tp, n) for n, options in zip(gt, frontiers) for o in options})

    def feasible(minimum):
        cost = sum(next((o.fp for o in options if o.tp * minimum.denominator >= n * minimum.numerator),
                        budget + 1) for n, options in zip(gt, frontiers))
        return cost <= budget

    # A minimum class recall is feasible exactly when its cheapest class options fit together.
    worst_optimum = next(level for level in reversed(levels) if feasible(level))
    results = {}
    for policy, minimum in (("class_macro", Fraction(0)), ("class_minimum", worst_optimum)):
        states = {0: (0, ())}
        for n, weight, options in zip(gt, weights, frontiers, strict=True):
            updated = {}
            for option in options:
                if option.tp * minimum.denominator < n * minimum.numerator:
                    continue
                for spent, (value, thresholds) in states.items():
                    total = spent + option.fp
                    if total <= budget:
                        choice = (value + weight * option.tp, thresholds + (option.cutoff,))
                        if total not in updated or choice > updated[total]:
                            updated[total] = choice
            states = updated
        cost = max(states, key=lambda b: (states[b][0], -b, states[b][1]))
        results[policy] = list(states[cost][1])
    return results
