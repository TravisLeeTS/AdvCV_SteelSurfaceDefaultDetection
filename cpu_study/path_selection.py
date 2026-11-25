"""Validation decisions at alternative class-recall tolerances."""
import csv
from collections import defaultdict
from statistics import mean

from steel.common import CLASSES, read, sha


TOLERANCES = (1, 2, 3, 5)


def failures(paired, tolerance):
    failed = []
    for key, changes in paired.items():
        limit = (1 if key == "map5095" else tolerance) / 100
        if min(changes[0], mean(changes)) < -limit - 1e-12:
            failed.append(key)
    return failed


def original_sensitivity(original):
    rows = []
    architecture = original["selection"]["architecture"]
    for tol in TOLERANCES:
        eligible = []
        for candidate in architecture:
            alarms = candidate["fp_per_image"]
            passes = not failures(candidate["paired"], tol) and max(alarms[0], mean(alarms)) <= .1 + 1e-12
            if tol == 2 and passes != candidate["eligible"]:
                raise ValueError("Could not reproduce original eligibility.")
            if passes:
                eligible.append(candidate)
        fastest = min(eligible, key=lambda r: r["p95"])
        tied = [r for r in eligible if r["low"] <= fastest["high"] and fastest["low"] <= r["high"]]
        chosen = next((r for r in tied if r["model"] == "A"), fastest)
        rows.append(dict(protocol="original_shared", tolerance_pp=tol,
                         eligible="|".join(r["model"] for r in eligible),
                         fastest=fastest["model"], selected=chosen["model"],
                         recall_first="", overlapping="|".join(r["model"] for r in tied if r != fastest)))
    return rows


def unified_selection(out):
    path = out / "tables/val-paths.csv"
    if sha(path) != read(out / "complete.json")["tables"][path.name]:
        raise ValueError("The validation timing and quality table differs from the completed benchmark.")
    groups = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["runtime"] == "openvino":
                if row["split"] != "val" or int(row["fp"]) > 36:
                    raise ValueError("Every candidate must meet the validation alarm budget.")
                groups[f"{row['model']}/{row['head']}"].append(row)
    if len(groups) != 12:
        raise ValueError("Expected ten dual-head paths and two historical stock paths.")
    for seeds in groups.values():
        seeds.sort(key=lambda r: int(r["seed"]))
        if [int(r["seed"]) for r in seeds] != [0, 1, 2]:
            raise ValueError("Incomplete paired validation seeds.")
    reference = groups["A/one"]
    sensitivity, details = [], []
    for tol in TOLERANCES:
        options = []
        for name, seeds in groups.items():
            deltas = {k: [float(r[k]) - float(b[k]) for r, b in zip(seeds, reference, strict=True)]
                      for k in ["map5095", *(f"recall_{c}" for c in CLASSES)]}
            failed = failures(deltas, tol)
            latency = [float(r[f"session{s}_ms"]) for r in seeds for s in range(3)]
            row = dict(path=name, tolerance_pp=tol, eligible=not failed, failed="|".join(failed),
                       val_p95_ms=mean(float(r["p95_ms"]) for r in seeds),
                       val_macro=mean(float(r["macro_recall"]) for r in seeds),
                       val_map=mean(float(r["map5095"]) for r in seeds),
                       low_ms=min(latency), high_ms=max(latency))
            details.append(row)
            if not failed:
                options.append(row)
        speed = min(options, key=lambda r: (r["val_p95_ms"], -r["val_macro"], r["path"]))
        recall = min(options, key=lambda r: (-r["val_macro"], r["val_p95_ms"], r["path"]))
        overlaps = [r["path"] for r in options if r != speed and r["low_ms"] <= speed["high_ms"]
                    and speed["low_ms"] <= r["high_ms"]]
        sensitivity.append(dict(protocol="unified_global", tolerance_pp=tol,
                                eligible="|".join(sorted(r["path"] for r in options)),
                                fastest=speed["path"], selected=speed["path"], recall_first=recall["path"],
                                overlapping="|".join(sorted(overlaps))))
    return details, sensitivity
