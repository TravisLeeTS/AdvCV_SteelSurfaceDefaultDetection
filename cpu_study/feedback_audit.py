"""Rebuild the defect-frame, class, tolerance and session contrasts from archived evidence.

Run `python -m cpu_study.feedback_audit` from the project root to write the tables.
Add `--verify` to compare a temporary rebuild with the published tables and receipt.
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np
from filelock import FileLock

from scripts.verify_research_metrics import rematch
from steel.common import CLASSES, put, read, sha, table
from steel.metrics import summarize
from .feedback_timing import timing_tables
from .head_calibration import load_annotations, verify_floor
from .heads import grouped_draws
from .operating_policy import read_path


SEEDS = (0, 1, 2)
MEAN = "paired mean"
CONDITIONS = {
    "stock_one_global": ("A", "one", "global"),
    "stock_dense_global": ("A", "dense", "global"),
    "pconv_one_global": ("B", "one", "global"),
    "pconv_dense_global": ("B", "dense", "global"),
    "pconv_dense_macro": ("B", "dense", "class_macro"),
    "pconv_dense_minimum": ("B", "dense", "class_minimum"),
}
CONTRASTS = {
    "combined": ("pconv_dense_global", "stock_one_global"),
    "backbone_dense": ("pconv_dense_global", "stock_dense_global"),
    "backbone_one": ("pconv_one_global", "stock_one_global"),
    "head_pconv": ("pconv_dense_global", "pconv_one_global"),
    "head_stock": ("stock_dense_global", "stock_one_global"),
    "macro_vs_global": ("pconv_dense_macro", "pconv_dense_global"),
    "minimum_vs_global": ("pconv_dense_minimum", "pconv_dense_global"),
    "minimum_vs_macro": ("pconv_dense_minimum", "pconv_dense_macro"),
    "macro_vs_stock": ("pconv_dense_macro", "stock_one_global"),
    "minimum_vs_stock": ("pconv_dense_minimum", "stock_one_global"),
}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def check_digest(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Recorded checksum differs at {path}")


def checked_index(records, keys):
    lookup = {}
    for row in records:
        key = tuple(row[k] for k in keys)
        if key in lookup:
            raise ValueError(f"Duplicate observation for {key}")
        lookup[key] = row
    return lookup


def labeled_counts(rows, boxes, items, thresholds):
    """Keep image and class identities for independent object matching and paired sampling."""
    cuts = np.asarray(thresholds, dtype=np.float64)
    if cuts.shape != (len(CLASSES),) or not np.isfinite(cuts).all():
        raise ValueError("Invalid frozen threshold vector")
    verify_floor(rows, boxes, cuts.min())
    gt, hits, alarms = (np.empty((len(rows), len(CLASSES)), dtype=np.int64) for _ in range(3))
    for i, (row, b, event) in enumerate(zip(rows, boxes, items, strict=True)):
        labels = b[:, 5].astype(int)
        accepted = b[b[:, 4] >= cuts[labels]]
        objects, false = [set() for _ in CLASSES], Counter()
        rematch(row, accepted.tolist(), objects, false)
        gt[i] = np.bincount(row["classes"], minlength=len(CLASSES))
        hits[i] = [len(x) for x in objects]
        alarms[i] = [false[c] for c in range(len(CLASSES))]
        scores, label, matched, expected_gt = event
        keep = scores >= cuts[label]
        true = np.bincount(label[keep & matched[:, 0]], minlength=len(CLASSES))
        extras = np.bincount(label[keep & ~matched[:, 0]], minlength=len(CLASSES))
        if not (np.array_equal(hits[i], true) and np.array_equal(alarms[i], extras)
                and np.array_equal(gt[i], expected_gt) and len(accepted) == hits[i].sum() + alarms[i].sum()):
            raise ValueError(f"Independent object or frame matching differs for {row['id']}")
    return gt, hits, alarms


def weighted_rates(gt, hits, false, weights):
    gt_w, hits_w = weights @ gt, weights @ hits
    if (gt_w == 0).any():
        raise ValueError("A group draw contains no annotations of a class")
    recall = 100 * hits_w / gt_w
    defective = gt.sum(axis=1) > 0
    located = hits.any(axis=1) & defective
    flagged = (located | false.any(axis=1)) & defective
    denominator = weights @ defective
    measures = {"macro_recall": recall.mean(axis=1), "minimum_recall": recall.min(axis=1),
                "false_boxes_per_defective_image": (weights @ (false.sum(axis=1) * defective)) / denominator}
    frames = {"any_alarm": flagged, "correct_match": located, "unmatched_only": flagged & ~located,
              "no_alarm": defective & ~flagged}
    for name, mask in frames.items():
        measures[name] = 100 * (weights @ mask) / denominator
    for c, name in enumerate(CLASSES):
        measures[f"{name}_recall"] = recall[:, c]
        measures[f"{name}_false_boxes_per_defective_image"] = (weights @ (false[:, c] * defective)) / denominator
    return measures, {k: int(v.sum()) for k, v in frames.items()}


def check_resampled_images(gt, hits, false, weights, measured):
    """Check resampling denominators by physically repeating image indices on three fixed draws."""
    for draw in (1, len(weights) // 2, len(weights) - 1):
        ids = np.repeat(np.arange(len(gt)), weights[draw])
        g, t, f = gt[ids], hits[ids], false[ids]
        defective = (g.sum(axis=1) > 0)
        yes = (t[defective].sum(axis=1) > 0)
        spurious = (f[defective].sum(axis=1) > 0)
        recall = 100 * t.sum(axis=0) / g.sum(axis=0)
        expected = {"macro_recall": recall.mean(), "minimum_recall": recall.min(),
                    "false_boxes_per_defective_image": f[defective].sum() / defective.sum(),
                    "any_alarm": 100 * (yes | spurious).mean(), "correct_match": 100 * yes.mean(),
                    "unmatched_only": 100 * (~yes & spurious).mean(), "no_alarm": 100 * (~yes & ~spurious).mean()}
        for i, defect in enumerate(CLASSES):
            expected[f"{defect}_recall"] = recall[i]
            expected[f"{defect}_false_boxes_per_defective_image"] = f[defective, i].mean()
        np.testing.assert_allclose([measured[k][draw] for k in expected], list(expected.values()),
                                   rtol=0, atol=1e-12)


def condition_rows(split, condition, seed, rows, boxes, items, entry, ap, released, weights):
    model, head, policy = CONDITIONS[condition]
    cuts = entry["thresholds"][policy]
    gt, hits, false = labeled_counts(rows, boxes, items, cuts)
    measured, frame_counts = weighted_rates(gt, hits, false, weights)
    check_resampled_images(gt, hits, false, weights, measured)
    defective = gt.any(axis=1)
    clean_flags = int((~defective & false.any(axis=1)).sum())
    n_clean = int((~defective).sum())
    total = dict(images=len(rows), defective_images=int(defective.sum()), clean_images=n_clean,
                 clean_alarms=clean_flags, gt_objects=int(gt.sum()), matched_objects=int(hits.sum()),
                 missed_objects=int(gt.sum()-hits.sum()), false_boxes=int(false.sum()), **frame_counts)
    if (total["correct_match"] + total["unmatched_only"] != total["any_alarm"] or
            total["any_alarm"] + total["no_alarm"] != total["defective_images"]):
        raise ValueError("Incomplete or overlapping defective-frame categories")
    # Every fit is evaluated on the same defective images. Undefined clean FRR remains blank.
    context = dict(split=split, condition=condition, model=model, head=head, policy=policy, side=480, seed=seed)
    frame = dict(**context, **total, map5095_pct=100*ap, macro_recall_pct=measured["macro_recall"][0],
                 minimum_recall_pct=measured["minimum_recall"][0],
                 false_boxes_per_defective_image=measured["false_boxes_per_defective_image"][0],
                 clean_false_rejection_pct=(100*clean_flags/n_clean if n_clean else None),
                 **{f"{k}_pct": measured[k][0] for k in frame_counts})
    alias = {"defective_images": "positive_images", "clean_alarms": "clean_flagged", "gt_objects": "gt",
             "matched_objects": "tp", "missed_objects": "fn", "false_boxes": "fp", "any_alarm": "flagged",
             "correct_match": "localized", "unmatched_only": "flagged_only_by_fp", "no_alarm": "unflagged"}
    for k, value in total.items():
        if value != int(released[alias.get(k, k)]):
            raise ValueError(f"{split}/{condition}/{seed}: independent count differs for {k}")
    for new, old in (("map5095_pct", "map5095"), ("macro_recall_pct", "macro"), ("minimum_recall_pct", "minimum")):
        np.testing.assert_allclose(frame[new], 100 * float(released[old]), atol=1e-10, rtol=0)
    classes = []
    for i, name in enumerate(CLASSES):
        row = dict(**context, defect=name, cutoff=cuts[i], gt_objects=int(gt[:, i].sum()),
                   matched_objects=int(hits[:, i].sum()), missed_objects=int(gt[:, i].sum() - hits[:, i].sum()),
                   false_boxes=int(false[:, i].sum()), recall_pct=measured[f"{name}_recall"][0])
        for k, old in (("gt_objects", "gt"), ("matched_objects", "tp"), ("false_boxes", "fp")):
            if row[k] != int(released[f"{name}_{old}"]):
                raise ValueError(f"Class counts differ at {split}/{condition}/{seed}/{name}")
        np.testing.assert_allclose(row["recall_pct"], 100 * float(released[f"{name}_recall"]), atol=1e-10, rtol=0)
        classes.append(row)
    frame["weakest_classes"] = "|".join(r["defect"] for r in classes
                                        if abs(r["recall_pct"] - frame["minimum_recall_pct"]) < 1e-12)
    return frame, classes, measured


def with_seed_means(records, keys):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    answer = []
    for group in groups.values():
        group.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in group] != list(SEEDS):
            raise ValueError("Expect one observation for each trained seed")
        out = {}
        for k, v in group[0].items():
            vals = [r[k] for r in group]
            if k == "seed":
                out[k] = MEAN
            elif k == "weakest_classes":
                out[k] = "|".join(f"{s}={x}" for s, x in zip(SEEDS, vals, strict=True))
            elif k == "cutoff":
                out[k] = None
            elif k in keys or k in ("model", "head", "side", "policy"):
                out[k] = v
            elif all(x is None for x in vals):
                out[k] = None
            else:
                out[k] = mean(vals)
        answer.extend([*group, out])
    return answer


def paired_effects(split, results, n_groups):
    for contrast, (candidate, reference) in CONTRASTS.items():
        for group in ((0,), (1,), (2,), SEEDS):
            for metric in results[candidate, 0]:
                a, b = (np.mean([results[name, s][metric] for s in group], axis=0)
                        for name in (candidate, reference))
                # Row zero gives the observed estimate. Remaining rows are matched whole-group draws.
                difference = a - b
                low, high = np.percentile(difference[1:], (2.5, 97.5))
                yield dict(split=split, contrast=contrast, candidate=candidate, reference=reference,
                           seed=MEAN if len(group) == 3 else group[0], metric=metric,
                           candidate_value=a[0], reference_value=b[0], difference=difference[0],
                           low_95=low, high_95=high,
                           difference_unit="boxes/defective image" if "false_boxes" in metric else "percentage points",
                           groups=n_groups, draws=len(difference)-1)


def check_prior_head_intervals(out, contrasts):
    """The original scalar-cutoff analysis reported the same head comparison in reverse."""
    path = out / "heads/paired_effects.csv"
    check_digest(path, read(out / "heads/verification.json")["tables"]["paired_effects"]["sha256"])
    expected = checked_index(csv_rows(path), ("split", "model", "runtime", "seed", "metric"))
    head_models = {"head_stock": "A", "head_pconv": "B"}
    metrics = {"macro_recall", "false_boxes_per_defective_image", *[f"{c}_recall" for c in CLASSES]}
    checked = 0
    for row in contrasts:
        if row["contrast"] not in head_models or row["metric"] not in metrics:
            continue
        m = "fp_per_image" if row["metric"] == "false_boxes_per_defective_image" else row["metric"]
        old = expected[row["split"], head_models[row["contrast"]], "openvino", str(row["seed"]), m]
        np.testing.assert_allclose([row[k] for k in ("difference", "low_95", "high_95")],
            [-float(old[k]) for k in ("one_minus_dense", "high", "low")], rtol=0, atol=1e-10)
        checked += 1
    if checked != 2 * len(head_models) * 4 * len(metrics):
        raise ValueError("Incomplete cross-check against prior matched-head intervals")
    return checked


def tolerance_table(out, released):
    path = out / "tables/val-paths.csv"
    check_digest(path, read(out / "complete.json")["tables"][path.name])
    selected = [r for r in csv_rows(path) if r["runtime"] == "openvino"]
    lookup = checked_index(selected, ("model", "head", "seed"))
    gates = {r["path"]:r["eligible"] == "True" for r in csv_rows(out / "policies/path-eligibility.csv")
             if float(r["tolerance_pp"]) == 2}
    fields = ["map5095", *[f"recall_{c}" for c in CLASSES]]
    for (model, head) in sorted({(r["model"], r["head"]) for r in selected}):
        path_id = f"{model}/{head}"
        records = []
        for m in fields:
            limit = 1 if m == "map5095" else 2
            series = []
            for s in SEEDS:
                row = lookup[model, head, str(s)]
                ref = lookup["A", "one", str(s)]
                if row["split"] != "val" or int(row["fp"]) > 36:
                    raise ValueError(f"Invalid validation gate input for {path_id}")
                sample = released["val", f"{row['run']}-openvino-{head}-480", "global"]
                metric = "map5095" if m == "map5095" else m.removeprefix("recall_") + "_recall"
                np.testing.assert_allclose(float(row[m]), float(sample[metric]), rtol=0, atol=1e-13)
                series.append((100*float(row[m]), 100*float(ref[m])))
            for scope, (a, b) in [*zip(SEEDS, series, strict=True), (MEAN, tuple(map(mean, zip(*series))))]:
                delta = a - b
                records.append(dict(path=path_id, reference="A/one", seed=scope, metric=m, candidate_pct=a,
                                    reference_pct=b, difference_pp=delta, loss_limit_pp=limit,
                                    passes_this_metric=delta >= -limit - 1e-10,
                                    used_in_gate=scope in (0, MEAN)))
        eligible = all(r["passes_this_metric"] for r in records if r["used_in_gate"])
        if eligible != gates[path_id]:
            raise ValueError(f"Recomputed eligibility differs at {path_id}")
        strict = all(r["passes_this_metric"] for r in records if r["seed"] != MEAN)
        for row in records:
            yield dict(**row, eligible_under_registered_gate=eligible, passes_at_every_seed=strict)


def run(data, artifacts, destination):
    out, study = artifacts / "cpu-extension", artifacts / "study"
    policies = out / "policies"
    proof = read(policies / "verification.json")
    check_digest(policies / "freeze.json", proof["freeze_sha"])
    for f, checksum in proof["files"].items():
        check_digest(policies / f, checksum)
    frozen = read(policies / "freeze.json")
    split = read(data / "split.json")
    original = read(study / "freeze.json")
    if (frozen["split_hash"] != original["split_sha"] or frozen["split_hash"] != split["split_hash"]
            or frozen["validation_annotations_sha"] != split["split_files"]["val"]):
        raise ValueError("Policy freeze and split identities differ")
    released = checked_index(csv_rows(policies / "operating-results.csv"), ("split", "path", "policy"))
    sources = {r["path"]:r["sha256"] for r in csv_rows(policies / "source-digests.csv")}
    root = Path(__file__).resolve().parents[1]
    spec = read(root / "protocol.json")["uncertainty"]
    frames, classes, contrasts = [], [], []
    proof_sources, groups_by_split = {}, {}
    tables = {}
    for part in ("val", "test"):
        samples = load_annotations(data, part)
        groups, draws = grouped_draws(samples, spec)
        groups_by_split[part] = len(groups)
        weights = np.vstack((np.ones((1,len(samples)), dtype=np.int64), draws))
        results, cached = {}, {}
        for condition, (model, head, policy) in CONDITIONS.items():
            for seed in SEEDS:
                path = f"{model}{seed}-openvino-{head}-480"
                entry = frozen["fits"][path]
                if (entry["model"], entry["head"], entry["seed"], entry["side"]) != (model, head, seed, 480):
                    raise ValueError(f"Invalid policy graph identity for {path}")
                if path not in cached:
                    boxes, items, ap, checksum = read_path(study, out, original, samples, part,
                                                          split["split_files"][part], entry)
                    if checksum != sources[f"{part}/{path}"] or (part == "val" and checksum != entry["validation_predictions_sha"]):
                        raise ValueError(f"Policy source differs at {part}/{path}")
                    np.testing.assert_allclose(ap, summarize(items)["map5095"], atol=1e-12, rtol=0)
                    cached[path] = boxes, items, ap
                    proof_sources[f"{part}/{path}"] = checksum
                boxes, items, ap = cached[path]
                frame, defects, measured = condition_rows(part, condition, seed, samples, boxes, items,
                    entry, ap, released[part, path, policy], weights)
                frames.append(frame)
                classes.extend(defects)
                results[condition, seed] = measured
        contrasts.extend(paired_effects(part, results, len(groups)))
    tables["frames"] = with_seed_means(frames, ("split", "condition"))
    tables["classes"] = with_seed_means(classes, ("split", "condition", "defect"))
    tables["paired-effects"] = contrasts
    prior_interval_checks = check_prior_head_intervals(out, contrasts)
    tables["validation-tolerance"] = list(tolerance_table(out, released))
    timings, timing_sources = timing_tables(splits={p: load_annotations(data, p) for p in ("val", "test")},
        out=out, conditions=CONDITIONS, contrasts=CONTRASTS, freeze=frozen, freeze_sha=proof["freeze_sha"])
    tables.update(timings)
    destination.mkdir(exist_ok=True, parents=True)
    with FileLock(str(destination / "audit.lock")):
        for name, rows in tables.items():
            table(destination / (name + ".csv"), rows)
        receipt = dict(policy_freeze_sha256=proof["freeze_sha"], split_hash=frozen["split_hash"],
            annotation_sha256={k:split["split_files"][k] for k in ("val","test")}, prediction_sha256=proof_sources,
            split_groups=groups_by_split, scalar_image_policy_checks=sum(f["images"] for f in frames),
            independently_expanded_bootstrap_samples=3*len(frames),
            matched_prior_head_intervals=prior_interval_checks,
            prior_head_intervals_sha256=sha(out/"heads/paired_effects.csv"),
            inputs=proof["files"], timing_input_sha256=timing_sources,
            analysis_sha256={f:sha(root/f) for f in ("cpu_study/feedback_audit.py","cpu_study/feedback_timing.py")},
            tables={n:dict(rows=len(rs),sha256=sha(destination/(n+".csv"))) for n,rs in tables.items()},
            uncertainty=dict(**spec, scope="Conditional on frozen policies, the inspected split and three fitted seeds. "
                "Group resamples are paired across seeds and policies. The minimum is computed per seed before averaging. "
                "Intervals are marginal per contrast and metric."),
            frame_denominator="The same 360 defect-bearing frames per seed per split. Mean counts average repeated "
                "predictions of the same images. Clean-frame false rejection is undefined with zero clean exposures.",
            timing_estimand="Per run, each path's median of three session p95 values is computed in each seed, "
                "then averaged across seeds. Paired session differences are descriptive. Equal split, seed and session "
                "index reproduce image order. Paths and runs were scheduled separately at the recorded slots.",
            relative_gate="The validation gate checks seed 0 and the paired three-seed mean. A 1 pp mAP50-95 "
                "and 2 pp per-class recall loss are allowed. Every validation fit stays within 36 false boxes.")
        put(destination / "verification.json", receipt)
    return receipt


def verify(data, artifacts, published):
    """Compare every generated evidence file with the published version, byte for byte."""
    with TemporaryDirectory(prefix="steel-feedback-replay-") as temporary:
        replay = Path(temporary)
        result = run(data, artifacts, replay)
        names = [f"{name}.csv" for name in result["tables"]]
        for name in [*names, "verification.json"]:
            target = published / name
            if not target.is_file() or target.read_bytes() != (replay / name).read_bytes():
                raise ValueError(f"Published feedback audit differs from its independent rebuild: {target}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, help="Evidence directory to write or verify.")
    parser.add_argument("--verify", action="store_true",
                        help="Rebuild in a temporary directory and verify the published CSVs and receipt.")
    args = parser.parse_args()
    destination = args.output or args.artifacts/"cpu-extension/feedback"
    if args.verify:
        result = verify(args.data, args.artifacts, destination)
        print(f"Verified {len(result['tables'])} published CSVs and verification.json.")
    else:
        result = run(args.data, args.artifacts, destination)
        print(", ".join(f"{key}={v['rows']}" for key,v in result["tables"].items()))


if __name__ == "__main__":
    main()
