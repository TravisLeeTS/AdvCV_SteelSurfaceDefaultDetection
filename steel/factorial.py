"""Paired 2 x 2 YOLO26n contrasts at the registered 480-pixel operating point."""
import numpy as np

from .common import CLASSES


CONTRASTS = {
    "B-A": {"B": 1, "A": -1},
    "C-A": {"C": 1, "A": -1},
    "D-B": {"D": 1, "B": -1},
    "D-C": {"D": 1, "C": -1},
    "D-A": {"D": 1, "A": -1},
    "(D-B)-(C-A)": {"D": 1, "B": -1, "C": -1, "A": 1},
}
Q = "map5095"


def factorial_rows(rows, split, seeds, bootstrap=None, group_count=None):
    # Partial/pilot reports still emit their usual measurements.
    all_rows = {seed: {m: rows.get((split, m, seed, "openvino", "one", 480))
                       for m in "ABCD"} for seed in seeds}
    if not all(all(cells.values()) for cells in all_rows.values()):
        return []
    estimates = []
    metrics = {Q: 100, **{c+"_recall": 100 for c in CLASSES}}
    if bootstrap is None:
        metrics.update(map50=100, pipeline_p95_ms=1, fp_per_image=1)
    for label, coefficients in CONTRASTS.items():
        for metric, scale in metrics.items():
            paired = []
            samples = []
            unit = {"pipeline_p95_ms": "ms", "fp_per_image": "FP/image"}.get(metric, "pp")
            for seed, cells in all_rows.items():
                estimate = scale * sum(weight * cells[m][metric] for m, weight in coefficients.items())
                paired.append(estimate)
                row = dict(split=split, contrast=label, metric=metric, seed=seed, unit=unit, estimate=estimate,
                           runtime="openvino", head="one", side=480, precision="FP32")
                for m in "ABCD":
                    row[m+"_source"] = cells[m]["cell_dir"]
                    row[m+"_checkpoint_sha"] = cells[m]["checkpoint_sha"]
                if bootstrap is not None:
                    sample = scale * sum(weight * bootstrap((split, m, seed, "openvino", "one", 480))[metric]
                                         for m, weight in coefficients.items())
                    samples.append(sample)
                    low, high = np.percentile(sample, [2.5, 97.5])
                    row.update(low=low, high=high, draws=len(sample), groups=group_count)
                estimates.append(row)
            mean = dict(row, seed="paired mean", estimate=float(np.mean(paired)))
            for m in "ABCD":
                mean[m+"_source"] = "|".join(c[m]["cell_dir"] for c in all_rows.values())
                mean[m+"_checkpoint_sha"] = "|".join(c[m]["checkpoint_sha"] for c in all_rows.values())
            if samples:
                low, high = np.percentile(np.mean(samples, axis=0), [2.5, 97.5])
                mean.update(low=low, high=high)
            estimates.append(mean)
    return estimates
