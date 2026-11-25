"""Compare dual detector heads at independent validation operating points.

Run with `python -m cpu_study.heads freeze` or `python -m cpu_study.heads evaluate`.
The latter rechecks the validation-only freeze, then reads both splits.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from filelock import FileLock

from scripts.verify_research_metrics import rematch
from steel.common import CLASSES, put, read, replace_file, sha, table
from steel.metrics import events
from .head_calibration import (
    DEFAULT_OUTPUT, PRIMARY_BUDGET, freeze_validation, iter_head_cutoffs,
    load_annotations, load_cell, verify_floor,
)


@dataclass
class Counts:
    """One evaluation at a single frozen cutoff, with matched annotation identities."""
    gt: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    matched: dict

    def totals(self):
        gt, tp, fp = [array.sum(axis=0) for array in (self.gt, self.tp, self.fp)]
        recalls = tp / gt
        values = dict(gt=int(gt.sum()), tp=int(tp.sum()), fn=int((gt - tp).sum()), fp=int(fp.sum()),
                      fp_per_image=float(fp.sum()) / len(self.gt), macro_recall=float(recalls.mean()))
        for i, name in enumerate(CLASSES):
            values.update({f"{name}_gt": int(gt[i]), f"{name}_tp": int(tp[i]),
                           f"{name}_fn": int(gt[i] - tp[i]), f"{name}_fp": int(fp[i]),
                           f"{name}_recall": float(recalls[i])})
        return values


def count_at_cutoff(rows, predictions, metric_events, cutoff):
    n_classes = len(CLASSES)
    gt = np.zeros((len(rows), n_classes), dtype=np.int64)
    tp = np.zeros_like(gt)
    fp = np.zeros_like(gt)
    identities = {name: set() for name in CLASSES}
    for i, (row, boxes, event) in enumerate(zip(rows, predictions, metric_events, strict=True)):
        accepted = boxes[boxes[:, 4] >= cutoff]
        score, labels, matches, target_counts = event
        gt[i] = target_counts
        keep = score >= cutoff
        tp[i] = np.bincount(labels[keep & matches[:, 0]], minlength=n_classes)
        fp[i] = np.bincount(labels[keep & ~matches[:, 0]], minlength=n_classes)
        # A second scalar-IoU implementation identifies each matched ground-truth object.
        assignments, false = [set() for _ in CLASSES], Counter()
        rematch(row, accepted.tolist(), assignments, false)
        for c, name in enumerate(CLASSES):
            if (len(assignments[c]), false[c]) != (tp[i, c], fp[i, c]):
                raise ValueError(f"Independent counts disagree at {row['id']}/{name}/{cutoff}")
            identities[name].update(assignments[c])
    return Counts(gt, tp, fp, identities)


def paired_objects(dense, one):
    for i, name in enumerate(CLASSES):
        left, right = dense.matched[name], one.matched[name]
        n = int(one.gt[:, i].sum())
        yield dict(defect=name, gt=n, both=len(left & right), dense_only=len(left - right),
                   one_only=len(right - left), neither=n - len(left | right),
                   dense_tp=len(left), one_tp=len(right),
                   dense_fp=int(dense.fp[:, i].sum()), one_fp=int(one.fp[:, i].sum()),
                   one_minus_dense_recall_pp=100 * (len(right) - len(left)) / n)


def grouped_draws(rows, spec):
    groups, inverse = np.unique([row["group"] for row in rows], return_inverse=True)
    rng = np.random.default_rng(spec["seed"])
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                              size=spec["draws"]).astype(np.int64)
    return groups, weights[:, inverse]


def boot_metrics(counts, weights):
    denominator = weights @ counts.gt
    if np.any(denominator == 0):
        raise ValueError("A whole-group bootstrap draw lost every object of a class.")
    recalls = (weights @ counts.tp) / denominator
    per_image_fp = weights @ counts.fp.sum(axis=1) / weights.sum(axis=1)
    return np.column_stack([recalls.mean(axis=1), recalls, per_image_fp])


def paired_results(split, grouped, n_groups, n_draws):
    """Average the paired seed effects within each common group resample."""
    metrics = ["macro_recall"] + [f"{name}_recall" for name in CLASSES] + ["fp_per_image"]
    for (model, runtime), seeds in grouped.items():
        if set(seeds) != {0, 1, 2}:
            raise ValueError(f"Incomplete seed pairing for {model}/{runtime}")
        for group in ([0], [1], [2], [0, 1, 2]):
            values = {}
            for head in ("dense", "one"):
                draws = np.mean([seeds[s][head]["draws"] for s in group], axis=0)
                observed = np.mean([[seeds[s][head]["summary"][m] for m in metrics] for s in group], axis=0)
                values[head] = (draws, observed)
            differences = values["one"][0] - values["dense"][0]
            point = values["one"][1] - values["dense"][1]
            for i, metric in enumerate(metrics):
                multiplier = 1 if metric == "fp_per_image" else 100
                low, high = np.percentile(multiplier * differences[:, i], [2.5, 97.5])
                yield dict(split=split, model=model, runtime=runtime, side=480,
                           seed=str(group[0]) if len(group) == 1 else "paired mean", metric=metric,
                           dense=float(multiplier * values["dense"][1][i]),
                           one=float(multiplier * values["one"][1][i]),
                           one_minus_dense=float(multiplier * point[i]),
                           low=float(low), high=float(high),
                           difference_unit="FP/image" if multiplier == 1 else "percentage points",
                           groups=n_groups, bootstrap_draws=n_draws)


def evaluate(data=Path("data"), artifacts=Path("artifacts"), output=DEFAULT_OUTPUT):
    data, artifacts, output = map(Path, (data, artifacts, output))
    # Finish all 42 primary cutoffs and all sensitivity cutoffs before opening test predictions.
    freeze_validation(data, artifacts, output)
    freeze_sha = sha(output / "freeze.json")
    entries = list(iter_head_cutoffs(output))
    study = artifacts / "study"
    root = Path(__file__).resolve().parents[1]
    original = read(study / "freeze.json")
    inventory = read(study / "export-inventory.json")
    protocol = read(root / "protocol.json")
    sensitivity, pairs, intervals, files, main = [], [], [], {}, []
    for split in ("val", "test"):
        rows = load_annotations(data, split)
        groups, weights = grouped_draws(rows, protocol["uncertainty"])
        by_pair = {}
        for entry in entries:
            key = (entry["model"], entry["seed"], entry["runtime"], entry["head"])
            cell = f"{entry['run']}-{entry['runtime']}-{entry['head']}-480"
            meta, quality, pred = load_cell(study, rows, split, key, original, inventory)
            if (entry["artifact_sha"], entry["checkpoint_sha"]) != (meta["artifact_sha"], meta["checkpoint_sha"]):
                raise ValueError(f"Split artifacts differ at {split}/{cell}")
            if split == "val" and entry["predictions_sha"] != quality["predictions_sha"]:
                raise ValueError(f"Validation scores differ from the freeze at {cell}")
            tick = events(rows, pred)
            context = dict(split=split, run=entry["run"], model=entry["model"], seed=entry["seed"],
                           runtime=entry["runtime"], head=entry["head"], side=480)
            files[f"{split}/{cell}"] = dict(predictions_sha=quality["predictions_sha"], floor=meta["floor"],
                                            images=len(rows), max_retained_boxes=max(map(len, pred)))
            for op in entry["operating_points"]:
                cutoff = op["cutoff"]
                verify_floor(rows, pred, cutoff)
                counts = count_at_cutoff(rows, pred, tick, cutoff)
                summary = counts.totals()
                if split == "val" and (summary["fp"] != op["validation_fp"]
                        or summary["tp"] != op["validation_tp"]
                        or summary["fp"] > op["max_validation_fp"]
                        or not math.isclose(summary["macro_recall"], op["validation_macro_recall"], abs_tol=1e-14)
                        or any(summary[f"{name}_tp"] != op["validation_class_tp"][name] for name in CLASSES)):
                    raise ValueError(f"The frozen operating point does not reproduce on {cell}")
                fields = dict(**context, val_budget_fp_per_image=op["fp_budget_per_image"],
                              val_fp_cap=op["max_validation_fp"], cutoff=cutoff, images=len(rows), groups=len(groups),
                              **summary)
                sensitivity.append(fields)
                if op["fp_budget_per_image"] != PRIMARY_BUDGET / 100:
                    continue
                main.append(dict(**fields, shared_cutoff=entry["shared_cutoff"], map5095=quality["metrics"]["map5095"],
                                 map50=quality["metrics"]["map50"], checkpoint_sha=entry["checkpoint_sha"],
                                 artifact_sha=entry["artifact_sha"], predictions_sha=quality["predictions_sha"],
                                 freeze_sha=freeze_sha))
                sub = by_pair.setdefault((entry["model"], entry["runtime"]), {}).setdefault(entry["seed"], {})
                sub[entry["head"]] = dict(counts=counts, summary=summary, draws=boot_metrics(counts, weights))
        for (model, runtime), by_seed in by_pair.items():
            for seed, heads in by_seed.items():
                for record in paired_objects(heads["dense"]["counts"], heads["one"]["counts"]):
                    pairs.append(dict(split=split, model=model, seed=seed, runtime=runtime, side=480, **record))
        intervals.extend(paired_results(split, by_pair, len(groups), len(weights)))
    tables = dict(operating_points=main, budget_sensitivity=sensitivity, paired_objects=pairs, paired_effects=intervals)
    sources = ("cpu_study/head_calibration.py", "cpu_study/heads.py", "cpu_study/head_figure.py",
               "scripts/verify_research_metrics.py", "scripts/survey_evidence.py", "steel/metrics.py", "steel/common.py")
    result = dict(freeze_sha=freeze_sha, protocol_sha=sha(root / "protocol.json"),
                  analysis_sources_sha256={name: sha(root / name) for name in sources},
                  original_study_freeze_sha=sha(study / "freeze.json"),
                  split_sha=original["split_sha"], input_sources=files,
                  split_files={split: sha(data / f"{split}.json") for split in ("val", "test")},
                  calibration="Separate cutoff for each checkpoint, runtime and deployed head. Validation only.",
                  matching="Scalar and vectorized same-class greedy matches agree for every image and operating point.",
                  coverage="All retained boxes are sorted by descending score. The archived 0.001 floor, "
                           "300-box cap and ordered records cover every chosen cutoff. Filtering low-score records "
                           "reproduced the directly executed shared operating predictions image-by-image.",
                  uncertainty="1,000 paired whole-group resamples conditional on the fitted cutoffs and three trained "
                              "seeds. Per-class recall denominators are recomputed per draw and the six recalls have equal "
                              "weight. Difference = one-to-one minus dense. FP/image counts alarm boxes on annotated images.",
                  timing="The separate CPU timing matrix must use each frozen primary cutoff.")
    from .head_figure import plot_head_sensitivity
    with FileLock(str(output / "results.lock")):
        for name, records in tables.items():
            table(output / f"{name}.csv", records)
        result["tables"] = {name: dict(rows=len(rows), sha256=sha(output / f"{name}.csv"))
                            for name, rows in tables.items()}
        figure = output / "budget_sensitivity.pdf"
        temporary = figure.with_suffix(".pdf.tmp")
        plot_head_sensitivity(sensitivity, temporary)
        replace_file(temporary, figure)
        result["figure"] = dict(file=figure.name, sha256=sha(figure), source="budget_sensitivity.csv",
                               population="OpenVINO FP32, 480 pixels, test split, stock A/E, mean of seeds 0/1/2.",
                               allowances=[0.05, 0.10, 0.20, 0.30])
        put(output / "verification.json", result)
    print(f"{len(entries)} calibrated settings, {len(main)} primary split rows, {len(pairs)} paired object rows.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.stage == "freeze":
        r = freeze_validation(args.data, args.artifacts, args.output)
        print(f"{len(r['entries'])} cutoffs frozen at {r['frozen_utc']}.")
    else:
        evaluate(args.data, args.artifacts, args.output)


if __name__ == "__main__":
    main()
