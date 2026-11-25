"""Freeze independent sampling units and score a paired, coil-level acceptance screen."""
from collections import Counter
from datetime import date
from math import ceil
from random import Random

from scipy.stats import beta, binomtest

from steel.common import digest, read, sha


MODELS = ("B0", "A0")
CRITICAL = ("crazing", "rolled_in_scale")
TIMING = ("capture_to_controller_p95_ms", "completed_fps", "deadline_exceedance_fraction")


def required_text(record, keys):
    if any(not isinstance(record.get(k), str) or not record[k].strip() for k in keys):
        raise ValueError(f"Supply nonempty values for {', '.join(keys)}.")


def enrollment(policy, spec_sha, roster_path):
    roster = read(roster_path)
    campaign = roster["campaign"]
    required_text(campaign, ("site", "line", "surface", "start_date", "end_date", "reference_inspector", "clock_id"))
    rigs = campaign["rig_ids"]
    sampling, timing = policy["sampling"], policy["timing"]
    if (roster.get("policy_sha256") != spec_sha or roster.get("independent_blinded_census") is not True
            or date.fromisoformat(campaign["start_date"]) > date.fromisoformat(campaign["end_date"])
            or campaign["frame_pitch_mm"] != sampling["frame_pitch_mm"]
            or campaign["nominal_strip_speed_m_s"] != sampling["nominal_strip_speed_m_s"]
            or campaign.get("full_width_nonoverlapping") is not True or campaign.get("matched_isolated_rigs") is not True
            or len(rigs) != 2 or any(not isinstance(r, str) or not r.strip() for r in rigs) or rigs[0] == rigs[1]):
        raise ValueError("Invalid blinded sampling frame, acquisition geometry or shadow rig declaration.")
    timed_frames = timing["duration_seconds"] * timing["offered_fps"]
    coils = {}
    strips = set()
    for coil in roster["coils"]:
        required_text(coil, ("acquisition_group", "strip_id", "grade", "shift"))
        cid, strip = coil["acquisition_group"], coil["strip_id"]
        if (cid in coils or strip in strips or type(coil["whole_strip_clean"]) is not bool
                or type(coil["n_frames"]) is not int or coil["n_frames"] < max(timed_frames, sampling["clean_window_frames"])):
            raise ValueError(f"Duplicated physical unit or ineligible strip at {cid}.")
        targets = coil["critical_objects"]
        if not isinstance(targets, list) or (targets and coil["whole_strip_clean"]):
            raise ValueError(f"Inconsistent blinded object census at {cid}.")
        ids = set()
        for target in targets:
            required_text(target, ("object_id", "class"))
            if (target["object_id"] in ids or target["class"] not in CRITICAL
                    or type(target["frame_index"]) is not int or not 0 <= target["frame_index"] < coil["n_frames"]):
                raise ValueError(f"Invalid unique physical target or canonical exposure at {cid}.")
            ids.add(target["object_id"])
        coils[cid] = coil
        strips.add(strip)

    def rank(*keys):
        return digest([sampling["seed"], *keys])

    def draw(arm, pool):
        count = sampling["coils"][arm]
        if len(pool) < count:
            raise ValueError(f"{arm} requires {count} independent eligible coils, found {len(pool)}.")
        return sorted(pool, key=lambda cid: rank("draw", arm, cid))[:count]

    # Timing samples the natural operating mix before any enrichment by reference class.
    timed = draw("timing", list(coils))
    pools = {arm: [] for arm in ("clean", *CRITICAL)}
    for cid in coils.keys() - set(timed):
        coil = coils[cid]
        labels = {t["class"] for t in coil["critical_objects"]}
        if coil["whole_strip_clean"]:
            pools["clean"].append(cid)
        else:
            arm = min(CRITICAL, key=lambda label: rank("stratum", cid, label))
            if arm in labels:
                pools[arm].append(cid)
    selected = {"timing": timed, **{arm: draw(arm, pool) for arm, pool in pools.items()}}
    units = {}
    for arm, ids in selected.items():
        for slot, cid in enumerate(ids):
            coil = coils[cid]
            unit = {k: coil[k] for k in ("strip_id", "grade", "shift")}
            unit["arm"] = arm
            if arm in CRITICAL:
                targets = [t for t in coil["critical_objects"] if t["class"] == arm]
                target = min(targets, key=lambda t: rank("object", cid, t["object_id"]))
                unit.update(start=target["frame_index"], count=1, object_id=target["object_id"])
            else:
                n = timed_frames if arm == "timing" else sampling["clean_window_frames"]
                start = Random(rank("start", cid, arm)).randrange(coil["n_frames"] - n + 1)
                unit.update(start=start, count=n)
                if arm == "timing":
                    unit["rig_by_model"] = {model: rigs[(slot+i) % 2] for i, model in enumerate(MODELS)}
            units[cid] = unit

    def mix(items):
        return {field: dict(sorted(Counter(row[field] for row in items).items())) for field in ("grade", "shift")}

    return dict(version=policy["version"], policy_sha256=spec_sha, roster_sha256=sha(roster_path),
                campaign=campaign, eligible_coils=len(coils), eligible_mix=mix(coils.values()),
                selected_mix={arm: mix([units[cid] for cid in ids]) for arm, ids in selected.items()}, units=units)


def exact_screen(events, required, limit, side, confidence):
    """A one-sided Clopper-Pearson bound with one Bernoulli outcome per independent coil."""
    n, k = len(events), sum(events)
    if n > required:
        raise ValueError("The observed denominator exceeds the frozen enrollment.")
    bound = None
    if n == required and side == "lower":
        bound = float(beta.ppf(1-confidence, k, n-k+1)) if k else 0.0
    elif n == required and side == "upper":
        bound = float(beta.ppf(confidence, k+1, n-k)) if k < n else 1.0
    elif side not in ("upper", "lower"):
        raise ValueError("Unknown binomial bound.")
    met = (bound >= limit if side == "lower" else bound <= limit) if n == required else None
    return dict(events=k, observed=n, planned=required, estimate=k/n if n else None, side=side,
                confidence=confidence, bound=bound, limit=limit, met=met)


def paired_binary(candidate, stock):
    if candidate.keys() != stock.keys():
        raise ValueError("The two systems were scored on different coils.")
    n = len(candidate)
    both = sum(candidate[k] and stock[k] for k in candidate)
    b = sum(candidate.values()) - both
    a = sum(stock.values()) - both
    p = float(binomtest(b, b+a, .5).pvalue) if b+a else (1.0 if n else None)
    return dict(coils=n, both=both, B0_only=b, A0_only=a, neither=n-both-b-a,
                B0_minus_A0_pp=100*(b-a)/n if n else None, mcnemar_two_sided_p=p)


def timing_block(samples, model, timing):
    """Score a complete ordered trace. None is an explicitly logged dropped decision."""
    duration = timing["duration_seconds"]
    n = duration * timing["offered_fps"]
    if len(samples) != n:
        raise ValueError("A timing block must contain every scheduled trigger.")
    start = samples[0][0]
    captures = [s[0] for s in samples]
    offsets = [abs(t-start-i*1e9/timing["offered_fps"]) for i, t in enumerate(captures)]
    if any(b <= a for a, b in zip(captures, captures[1:])) or max(offsets) > 1e6*timing["capture_jitter_tolerance_ms"]:
        raise ValueError("Camera captures depart from the preregistered offered load.")
    latencies = []
    completed = 0
    for capture, acknowledgments in samples:
        ack = acknowledgments[model]
        if ack is not None:
            latencies.append((ack-capture)/1e6)
            completed += ack <= start + duration*10**9
    drops = n-len(latencies)
    rank = ceil(.95*n)
    p95 = sorted(latencies)[rank-1] if len(latencies) >= rank else None
    late = drops + sum(t > timing["frame_deadline_ms"] for t in latencies)
    fps = completed/duration
    return dict(scheduled=n, dropped=drops, completed_in_window=completed, late_or_dropped=late,
                capture_to_controller_p95_ms=p95, p95_is_infinite=p95 is None, completed_fps=fps,
                deadline_exceedance_fraction=late/n, passed={
                    TIMING[0]: p95 is not None and p95 <= timing["maximum_capture_to_controller_p95_ms"],
                    TIMING[1]: fps >= timing["minimum_completed_fps"],
                    TIMING[2]: late/n <= timing["maximum_deadline_exceedance_fraction"]})
