"""Independently certify both class-threshold objectives and replay the released policy tables."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from steel.common import CLASSES, put, read, sha
from steel.metrics import concatenate
from .head_calibration import load_annotations
from .operating_policy import read_path, run


def solver_optima(items, budget):
    """Multiple-choice integer program, independent of the published threshold DP."""
    score, labels, matched, targets = concatenate(items)
    options = []
    for label, total in enumerate(targets):
        mask = labels == label
        scores, true = score[mask], matched[mask, 0]
        # Generate the available TP counts at each exact FP cost without using class_frontier.
        available = {0: 0}
        hits = np.r_[0, np.cumsum(true, dtype=int)]
        for n in range(1, len(scores) + 1):
            if n < len(scores) and scores[n] == scores[n-1]:
                continue
            fp = int(n - hits[n])
            if fp > budget:
                break
            available[fp] = int(hits[n])
        options.extend((label, fp, tp / int(total)) for fp, tp in available.items())
    n_classes = len(CLASSES)
    # One binary per attainable class option and a final continuous worst-recall bound.
    matrix = np.zeros((2*n_classes + 1, len(options) + 1))
    for i, (label, fp, recall) in enumerate(options):
        matrix[label, i] = 1
        matrix[n_classes, i] = fp
        matrix[n_classes + 1 + label, i] = recall
    matrix[n_classes + 1:, -1] = -1
    lower = [1]*n_classes + [0] + [0]*n_classes
    upper = [1]*n_classes + [budget] + [np.inf]*n_classes
    def solve(objective, floor):
        bounds = Bounds([0]*len(options) + [floor], np.ones(len(options) + 1))
        answer = milp(objective, integrality=[1]*len(options) + [0], bounds=bounds,
                      constraints=LinearConstraint(matrix, lower, upper), options={"mip_rel_gap": 0})
        if not answer.success:
            raise ValueError(f"Independent optimization failed: {answer.message}")
        return matrix[n_classes + 1:, :-1] @ (answer.x[:-1] > .5)
    macro_objective = [-o[2] for o in options] + [0]
    optimal_macro = solve(macro_objective, 0).mean()
    optimal_minimum = solve([0]*len(options) + [-1], 0).min()
    secondary_macro = solve(macro_objective, max(0., optimal_minimum - 1e-10)).mean()
    return np.array([optimal_macro, optimal_minimum, secondary_macro])


def verify(data, artifacts):
    import csv
    directory = artifacts / "cpu-extension/policies"
    with TemporaryDirectory(prefix="steel-policy-replay-") as temporary:
        target = Path(temporary)
        run(data, artifacts, target)
        names = ("freeze.json", "operating-results.csv", "path-eligibility.csv", "recall-tolerance.csv",
                 "source-digests.csv")
        for name in names:
            if sha(target / name) != sha(directory / name):
                raise ValueError(f"Clean policy replay differs for {name}.")
        if read(target / "verification.json") != read(directory / "verification.json"):
            raise ValueError("Policy replay receipt differs.")
    freeze = read(directory / "freeze.json")
    with (directory / "operating-results.csv").open(newline="", encoding="utf-8") as source:
        measured = {(r["path"], r["policy"]): r for r in csv.DictReader(source) if r["split"] == "val"}
    rows = load_annotations(data, "val")
    val_sha = sha(data / "val.json")
    study, out = artifacts / "study", artifacts / "cpu-extension"
    original = read(study / "freeze.json")
    discrepancies = []
    for name, entry in freeze["fits"].items():
        if entry["side"] != 480:
            continue
        _, items, _, _ = read_path(study, out, original, rows, "val", val_sha, entry)
        expected = solver_optima(items, len(rows) // 10)
        macro, minimum = measured[name,"class_macro"], measured[name,"class_minimum"]
        observed = np.array([float(macro["macro"]), float(minimum["minimum"]), float(minimum["macro"])])
        np.testing.assert_allclose(expected, observed, rtol=0, atol=1e-10, err_msg=name)
        discrepancies.append(float(np.max(np.abs(expected - observed))))
    if len(discrepancies) != 36:
        raise ValueError("Expected all 36 OpenVINO paths at 480.")
    return dict(freeze_sha=sha(directory / "freeze.json"), graphs_at_480=len(discrepancies),
                checks=len(discrepancies)*3, largest_absolute_difference=max(discrepancies),
                method="Independent multiple-choice integer programs for mean recall, minimum recall and its mean-recall tie break.",
                policy_replay_rows=read(directory / "verification.json")["policy_rows"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.data, args.artifacts)
    if args.receipt:
        put(args.receipt, result)
    print(result)
