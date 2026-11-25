"""Compare independently fitted deployment paths and freeze a validation-only recommendation."""
from statistics import mean, median

from steel.bench import read_predictions
from steel.common import CLASSES, put, read, sha, table
from steel.metrics import events, summarize
from .benchmark import matrix, name, predictions
from .head_calibration import load_annotations


LABELS = {"A":"YOLO26n", "E":"YOLOv10n", "yolov8n":"YOLOv8n", "yolov5s":"YOLOv5s"}


def timed(out, split, entry):
    directory = out / "timing" / split / name(entry)
    values = []
    for session in range(3):
        status = read(directory / f"session-{session}.json")
        raw = directory / f"timings-s{session}.csv.gz"
        if (sha(raw) != status["observations_sha"] or status["key"]["cutoff"] != entry["cutoff"]
                or status["key"]["artifact"] != entry["artifact_sha"]):
            raise ValueError(f"Timing identity differs at {raw}")
        values.append(status["summary"]["p95_ms"])
    return values


def detail(data, artifacts, out, split):
    rows = load_annotations(data, split)
    result = []
    for entry in matrix(out):
        if entry["model"].startswith("yolo"):
            record = read(out / "stock" / split / entry["run"] / "quality.json")
            q, cost = record["quality"], record["complexity"]
        else:
            src = predictions(artifacts, out, split, entry)
            pred = read_predictions(src)
            if [i for i, _ in pred] != [r["id"] for r in rows]:
                raise ValueError(f"Prediction order differs at {src}")
            q = summarize(events(rows, [v for _, v in pred]), entry["cutoff"])
            cost = read(src.parent / "complexity.json")
        sessions = timed(out, split, entry)
        classes = q["classes"]
        result.append(dict(split=split, model=entry["model"], run=entry["run"], seed=entry["seed"],
            runtime=entry["runtime"], head=entry["head"], cutoff=entry["cutoff"], map5095=q["map5095"],
            fp_per_image=q["fp_per_image"], tp=sum(c["tp"] for c in classes.values()),
            fp=sum(c["fp"] for c in classes.values()), macro_recall=mean(c["recall"] for c in classes.values()),
            p95_ms=median(sessions), session0_ms=sessions[0], session1_ms=sessions[1], session2_ms=sessions[2],
            parameters=cost["active_parameters"], flops=cost["flops"],
            **{f"recall_{c}":classes[c]["recall"] for c in CLASSES}))
    return result


def stock_rows(detail_rows):
    return [r for r in detail_rows if r["runtime"] == "openvino" and r["model"] in LABELS
            and r["head"] == ("one" if r["model"] in ("A", "E") else "dense")]


def aggregate(rows):
    result = []
    for model, family in LABELS.items():
        group = sorted((r for r in rows if r["model"] == model), key=lambda r:r["seed"])
        if [r["seed"] for r in group] != [0,1,2]:
            raise ValueError(f"Missing trained seed for {family}")
        avg = {k:mean(r[k] for r in group) for k in ("map5095", "macro_recall", "fp_per_image", "p95_ms", "tp", "fp")}
        sessions = [r[f"session{s}_ms"] for r in group for s in range(3)]
        result.append(dict(split=group[0]["split"],family=family,model=model,head=group[0]["head"],
                           parameters=group[0]["parameters"], flops=group[0]["flops"], **avg,
                           p95_min_session=min(sessions), p95_max_session=max(sessions),
                           **{f"seed{r['seed']}_ms":r["p95_ms"] for r in group}))
    return result


def selection(out, rows, summary):
    reference = {r["seed"]:r for r in rows if r["model"]=="A"}
    deltas = {}
    for model in LABELS:
        model_rows = sorted((r for r in rows if r["model"] == model), key=lambda r:r["seed"])
        metrics = ["map5095"] + [f"recall_{c}" for c in CLASSES]
        checks = {}
        for metric in metrics:
            changes = [r[metric]-reference[r["seed"]][metric] for r in model_rows]
            limit = .01 if metric == "map5095" else .02
            checks[metric] = dict(seed0_delta=changes[0], mean_delta=mean(changes), limit=limit,
                                  passed=min(changes[0], mean(changes)) >= -limit - 1e-12)
        deltas[model] = dict(eligible=all(c["passed"] for c in checks.values()), checks=checks)
    ranked = sorted(summary, key=lambda r:r["p95_ms"])
    qualified = [r for r in ranked if deltas[r["model"]]["eligible"]]
    winner = qualified[0]
    overlaps = [r["family"] for r in qualified[1:] if
                r["p95_min_session"] <= winner["p95_max_session"] and r["p95_max_session"] >= winner["p95_min_session"]]
    speed_winner = ranked[0]
    record = dict(source_split="val", frozen_heads_sha=sha(out / "heads/freeze.json"),
        historical_freeze_sha=sha(out / "stock/freeze.json"), registration_sha=sha(out / "registration.json"),
        validation_table_sha=sha(out / "tables/val-paths.csv"), quality_gates=deltas,
        fastest_eligible=winner["family"], overlapping_eligible=overlaps,
        fastest_observed=speed_winner["family"], speed_order=[r["family"] for r in ranked],
        seed_speed_order=[[r["family"] for r in sorted(summary, key=lambda r:r[f"seed{s}_ms"])] for s in range(3)],
        fastest_overlaps=[r["family"] for r in ranked[1:] if r["p95_min_session"] <= speed_winner["p95_max_session"]])
    path = out / "stock/selection.json"
    if path.exists() and read(path) != record:
        raise ValueError("Existing validation recommendation differs.")
    if not path.exists():
        put(path, record)
    return record


def make_tables(data, artifacts, out, split):
    paths = detail(data, artifacts, out, split)
    table(out / "tables" / f"{split}-paths.csv", paths)
    rows = stock_rows(paths)
    stock = aggregate(rows)
    table(out / "tables" / f"{split}-families.csv", stock)
    if split == "val":
        selection(out, rows, stock)
    by_head = {}
    for row in paths:
        if row["model"] in "ABCDE":
            by_head.setdefault((row["model"], row["runtime"], row["head"]), []).append(row)
    comparisons = []
    for model, runtime, head in by_head:
        if head != "one":
            continue
        one = sorted(by_head[model,runtime,"one"],key=lambda r:r["seed"])
        dense = sorted(by_head[model,runtime,"dense"],key=lambda r:r["seed"])
        comparisons.append(dict(split=split,model=model,runtime=runtime,
            one_p95_ms=mean(r["p95_ms"] for r in one), dense_p95_ms=mean(r["p95_ms"] for r in dense),
            dense_minus_one_ms=mean(d["p95_ms"]-o["p95_ms"] for d,o in zip(dense,one,strict=True)),
            one_recall=mean(r["macro_recall"] for r in one),dense_recall=mean(r["macro_recall"] for r in dense),
            one_fp_image=mean(r["fp_per_image"] for r in one),dense_fp_image=mean(r["fp_per_image"] for r in dense)))
    table(out / "tables" / f"{split}-heads.csv", comparisons)
    return stock
