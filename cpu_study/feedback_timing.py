"""Recompute paired, archived CPU session percentiles without executing inference."""
import csv
import gzip
from pathlib import Path
from statistics import mean, median

import numpy as np

from steel.bench import image_order
from steel.common import digest, read, sha
from .benchmark import matrix, name
from .policy_timing import schedule


HEAD_RUN = "head_specific"
REPEAT_RUN = "policy_repeat"


def checked_csv(path, checksum):
    if sha(path) != checksum:
        raise ValueError(f"Changed timing table: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def percentile(path, rec, session, slot, rows, order, affinity, before, after):
    raw = path / f"timings-s{session}.csv.gz"
    if sha(raw) != rec["observations_sha"] or len(order) != 1200:
        raise ValueError(f"Changed timing file or image count at {raw}")
    rt, summary = rec["runtime"], rec["summary"]
    if ((rt["device"], rt["threads"], rt["streams"], rt["precision"]) != (["CPU"], 4, 1, "f32")
            or not(before["ac_online"] and after["ac_online"]) or before["power_profile"] != after["power_profile"]
            or len(set(affinity)) != 4 or (summary["boundary"], summary["session"], summary["warmups"],
                summary["timed"]) != ("pipeline", session, 200, 1000)):
        raise ValueError(f"Unexpected operation or hardware conditions at {raw}")
    elapsed = []
    with gzip.open(raw, "rt", encoding="utf-8", newline="") as source:
        for i, (event, image) in enumerate(zip(csv.DictReader(source), order[200:], strict=True)):
            if ((int(event["repetition"]), event["image_id"], int(event["invocation_order"]),
                 int(event["session"]), event["boundary"]) != (i, rows[image]["id"], slot, session, "pipeline")):
                raise ValueError(f"Unpaired repetition or image in {raw}")
            value = int(event["elapsed_ns"])
            if value <= 0:
                raise ValueError(f"Nonpositive latency in {raw}")
            elapsed.append(value)
    # Linear 95th sample percentile, interpolated independently in nanoseconds.
    ordered = sorted(elapsed)
    rank = (len(ordered)-1)*95
    lower, remainder = divmod(rank, 100)
    p95 = ((100-remainder)*ordered[lower] + remainder*ordered[lower+1]) / 1e8
    if abs(p95 - summary["p95_ms"]) > 1e-10:
        raise ValueError(f"Recorded p95 differs from raw calls at {raw}")
    return dict(p95_ms=p95, slot=slot, order_sha=digest(order.tolist()), affinity=tuple(affinity),
                profile=before["power_profile"], raw_sha256=rec["observations_sha"])


def collect(splits, out, conditions, freeze, freeze_sha):
    stock_freeze = sha(out / "stock/freeze.json")
    head_freeze = sha(out / "heads/freeze.json")
    root = Path(__file__).resolve().parent
    original_code, repeat_code = sha(root / "benchmark.py"), sha(root / "policy_timing.py")
    original_table_sha = read(out / "complete.json")["tables"]
    original_order, repeat_order = {}, {}
    entries = matrix(out)
    for session in range(3):
        permutation = np.random.default_rng(np.random.SeedSequence([20261125, session])).permutation(len(entries))
        for slot, index in enumerate(permutation):
            original_order[name(entries[index]), session] = slot
    for session, slot, entry in schedule(freeze):
        repeat_order[entry["graph"], entry["policy"], session] = slot
    repeated = read(out / "policy-timing/verification.json")
    if repeated["freeze_sha"] != freeze_sha or repeated["implementation"] != repeat_code:
        raise ValueError("Repeat timing and frozen policies have different identities")
    review = checked_csv(out / "policy-timing/timing.csv", repeated["table_sha"])
    known_repeat = {(r["model"], r["head"], r["policy"], int(r["side"]), int(r["seed"])): r for r in review}
    cells, sources = {}, {}
    for split, rows in splits.items():
        original_rows = checked_csv(out / f"tables/{split}-paths.csv", original_table_sha[f"{split}-paths.csv"])
        known_orig = {(v["model"], v["head"], int(v["seed"])): v for v in original_rows if v["runtime"] == "openvino"}
        for condition, (model, head, policy) in conditions.items():
            for seed in (0,1,2):
                graph = f"{model}{seed}-openvino-{head}-480"
                fit = freeze["fits"][graph]
                for session in range(3):
                    order = image_order(len(rows), 1200, session)
                    order_sha = digest(order.tolist())
                    if policy == "global":
                        loc = out / "timing" / split / graph
                        rec = read(loc / f"session-{session}.json")
                        k = rec["key"]
                        slot = original_order[graph, session]
                        if ((k["head_freeze"], k["historical_freeze"], k["implementation"], k["split"], k["session"],
                                k["invocation"], k["artifact"], k["cutoff"], rec["order_sha"], k["observations"],
                                k["warmups"], k["boundary"]) != (head_freeze, stock_freeze, original_code, split,
                                session, slot, fit["artifact_sha"], fit["thresholds"][policy][0], order_sha,
                                1000, 200, "decoded image to final boxes")):
                            raise ValueError(f"Independent-head timing identity differs at {loc}")
                        value = percentile(loc, rec, session, slot, rows, order, k["affinity"],
                                           rec["hardware_before"], rec["hardware_after"])
                        if abs(value["p95_ms"] - float(known_orig[model, head, seed][f"session{session}_ms"])) > 1e-10:
                            raise ValueError(f"Original released p95 differs at {loc}")
                        cells[split, HEAD_RUN, condition, seed, session] = value
                        sources[(loc / f"timings-s{session}.csv.gz").relative_to(out).as_posix()] = value["raw_sha256"]
                    if split != "test" or (graph, policy, session) not in repeat_order:
                        continue
                    loc = out / "policy-timing" / f"{graph}-{policy}"
                    rec = read(loc / f"session-{session}.json")
                    identity = rec["identity"]
                    slot = repeat_order[graph, policy, session]
                    c = identity["condition"]
                    if ((identity["freeze_sha"],identity["code_sha"],identity["split"],identity["session"],
                            identity["slot"],identity["order_sha"],c["graph"],c["policy"],c["thresholds"],
                            c["artifact_sha"],c["checkpoint_sha"]) != (freeze_sha,repeat_code,split,session,slot,
                            order_sha,graph,policy,fit["thresholds"][policy],fit["artifact_sha"],fit["checkpoint_sha"])):
                        raise ValueError(f"Repeat timing identity differs at {loc}")
                    value = percentile(loc, rec, session, slot, rows, order, rec["affinity"],
                                       rec["before"], rec["after"])
                    if abs(value["p95_ms"] - float(known_repeat[model,head,policy,480,seed][f"session{session}_ms"])) > 1e-10:
                        raise ValueError(f"Repeat released p95 differs at {loc}")
                    cells[split, REPEAT_RUN, condition, seed, session] = value
                    sources[(loc / f"timings-s{session}.csv.gz").relative_to(out).as_posix()] = value["raw_sha256"]
    return cells, sources


def timing_tables(splits, out, conditions, contrasts, freeze, freeze_sha):
    cells, sources = collect(splits, out, conditions, freeze, freeze_sha)
    sessions, summaries, repeats = [], [], []
    for split, runs in (("val", (HEAD_RUN,)), ("test", (HEAD_RUN, REPEAT_RUN))):
        by_run = {}
        for run in runs:
            available = {k[2] for k in cells if k[:2] == (split, run)}
            for contrast, (candidate, reference) in contrasts.items():
                if not {candidate, reference} <= available:
                    continue
                context = dict(split=split, run=run, contrast=contrast, candidate=candidate, reference=reference)
                per_seed, all_pairs = [], []
                for s in (0,1,2):
                    pairs = []
                    for i in (0,1,2):
                        a, b = (cells[split,run,c,s,i] for c in (candidate, reference))
                        if ((a["order_sha"],a["affinity"],a["profile"]) != (b["order_sha"],b["affinity"],b["profile"])
                                or a["slot"] == b["slot"]):
                            raise ValueError(f"Different image stream or CPU setup for {context}/{s}/{i}")
                        record = dict(**context, seed=s, session=i, candidate_slot=a["slot"], reference_slot=b["slot"],
                                      candidate_p95_ms=a["p95_ms"], reference_p95_ms=b["p95_ms"],
                                      difference_ms=a["p95_ms"]-b["p95_ms"], order_sha256=a["order_sha"])
                        sessions.append(record)
                        pairs.append(record)
                    av, bv = (median(p[k] for p in pairs) for k in ("candidate_p95_ms", "reference_p95_ms"))
                    ds = [p["difference_ms"] for p in pairs]
                    per_seed.append(dict(**context, seed=s, candidate_ms=av, reference_ms=bv,
                        difference_of_path_medians_ms=av-bv, median_paired_session_difference_ms=median(ds),
                        mean_paired_session_difference_ms=mean(ds), min_paired_difference_ms=min(ds),
                        max_paired_difference_ms=max(ds), faster_sessions=sum(d<0 for d in ds), paired_sessions=3))
                    all_pairs.extend(pairs)
                ds = [p["difference_ms"] for p in all_pairs]
                summaries.extend(per_seed)
                summaries.append(dict(**context, seed="paired mean", candidate_ms=mean(r["candidate_ms"] for r in per_seed),
                    reference_ms=mean(r["reference_ms"] for r in per_seed),
                    difference_of_path_medians_ms=mean(r["difference_of_path_medians_ms"] for r in per_seed),
                    median_paired_session_difference_ms=median(ds), mean_paired_session_difference_ms=mean(ds),
                    min_paired_difference_ms=min(ds), max_paired_difference_ms=max(ds),
                    faster_sessions=sum(d<0 for d in ds), paired_sessions=len(ds)))
                by_run[run, contrast] = all_pairs
        # Match streams, seeds and session identifiers across rotations. The hardware clocks were recorded
        # sequentially at distinct slots. These differences are descriptive run-to-run changes.
        for contrast, (candidate, reference) in contrasts.items():
            if not all((r, contrast) in by_run for r in (HEAD_RUN, REPEAT_RUN)):
                continue
            for original, repeat in zip(by_run[HEAD_RUN,contrast], by_run[REPEAT_RUN,contrast], strict=True):
                if any(original[k] != repeat[k] for k in ("split","contrast","seed","session","order_sha256")):
                    raise ValueError("Session pairing across runs is incomplete")
                s, i = original["seed"], original["session"]
                for c in (candidate,reference):
                    first, second = (cells[split,r,c,s,i] for r in (HEAD_RUN,REPEAT_RUN))
                    if first["profile"] != second["profile"] or first["affinity"] != second["affinity"]:
                        raise ValueError("CPU profile or affinity differs across repeated runs")
                repeats.append(dict(split=split, contrast=contrast, candidate=candidate, reference=reference, seed=s,
                    session=i, order_sha256=original["order_sha256"], head_candidate_slot=original["candidate_slot"],
                    head_reference_slot=original["reference_slot"], repeat_candidate_slot=repeat["candidate_slot"],
                    repeat_reference_slot=repeat["reference_slot"], head_candidate_ms=original["candidate_p95_ms"],
                    head_reference_ms=original["reference_p95_ms"], repeat_candidate_ms=repeat["candidate_p95_ms"],
                    repeat_reference_ms=repeat["reference_p95_ms"],
                    candidate_run_shift_ms=repeat["candidate_p95_ms"]-original["candidate_p95_ms"],
                    reference_run_shift_ms=repeat["reference_p95_ms"]-original["reference_p95_ms"],
                    head_difference_ms=original["difference_ms"], repeat_difference_ms=repeat["difference_ms"],
                    change_in_difference_ms=repeat["difference_ms"]-original["difference_ms"]))
    return {"timing-sessions":sessions, "timing-summary":summaries, "timing-repeat":repeats}, sources
