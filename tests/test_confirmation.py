"""Check coil-level bounds, enrollment and paired shadow telemetry with software fixtures."""
from contextlib import ExitStack
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cpu_study import confirmation as trial
from cpu_study.confirmation_design import exact_screen, timing_block
from steel.common import put, read, sha


class BoundsTest(unittest.TestCase):
    def test_exact_critical_clean_and_timing_boundaries(self):
        for n, limit, side, passed, failed in (
                (100, .90, "lower", 96, 95), (300, .01, "upper", 0, 1), (60, .95, "lower", 60, 59)):
            for k, expected in ((passed, True), (failed, False)):
                end = exact_screen([True]*k + [False]*(n-k), n, limit, side, .95)
                self.assertEqual(end["met"], expected)
                b = end["bound"]
                states = range(k, n+1) if side == "lower" else range(k+1)
                tail = sum(math.comb(n,i)*b**i*(1-b)**(n-i) for i in states)
                self.assertAlmostEqual(tail, .05, places=9)
        for n in (298, 299):
            end = exact_screen([False]*n, n, .01, "upper", .95)
            self.assertAlmostEqual(end["bound"], 1-.05**(1/n), places=12)
            self.assertEqual(end["met"], n == 299)
        partial = exact_screen([False]*299, 300, .01, "upper", .95)
        self.assertIsNone(partial["bound"])
        self.assertIsNone(partial["met"])

    def test_timing_boundaries_keep_drops_and_deadlines_in_denominator(self):
        rules = read(trial.SPEC)["timing"]
        def block(drops, delayed=0):
            rows = []
            for i in range(3000):
                capture = i * 20_000_000
                ack = None if i < drops else capture+20_000_000+int(i < drops+delayed)
                rows.append((capture, {"B0": ack}))
            return timing_block(rows, "B0", rules)
        clean = block(0)
        self.assertEqual(clean["completed_fps"], 50)
        self.assertTrue(all(clean["passed"].values()))
        one = block(1)
        self.assertFalse(one["passed"]["completed_fps"])
        self.assertTrue(one["passed"]["deadline_exceedance_fraction"])
        self.assertTrue(block(0,30)["passed"]["deadline_exceedance_fraction"])
        self.assertFalse(block(0,31)["passed"]["deadline_exceedance_fraction"])
        self.assertEqual(block(150)["capture_to_controller_p95_ms"], 20)
        self.assertTrue(block(151)["p95_is_infinite"])

    def test_all_six_gates_require_their_planned_independent_coils(self):
        policy = read(trial.SPEC)
        policy["timing"].update(duration_seconds=1, offered_fps=2, minimum_completed_fps=2)
        units, observed = {}, {}
        for arm, count in policy["sampling"]["coils"].items():
            for i in range(count):
                cid = f"{arm}-{i}"
                units[cid] = dict(arm=arm, start=0, count=2 if arm == "timing" else 1)
                if arm == "timing":
                    observed[cid] = {j:(j*500_000_000, {m:j*500_000_000+10_000_000 for m in trial.MODELS})
                                     for j in range(2)}
                else:
                    observed[cid] = {0:{m:arm != "clean" for m in trial.MODELS}}
        observed["timing-0"][0][1]["A0"] = None
        result = trial.summarize(policy, units, observed)
        self.assertTrue(result["paired_study_complete"])
        self.assertTrue(result["models"]["B0"]["passes_supervised_screen"])
        self.assertFalse(result["models"]["A0"]["passes_supervised_screen"])
        self.assertEqual(result["paired_coil_outcomes"]["completed_fps"]["B0_only"],1)


class PairedTrialTest(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.data = self.root/"benchmark"
        for split in ("train", "val", "test"):
            put(self.data/f"{split}.json", [])
        self.p = read(trial.SPEC)
        self.p["sampling"].update(coils=dict(clean=1,crazing=1,rolled_in_scale=1,timing=2), clean_window_frames=2)
        self.p["timing"].update(duration_seconds=1,offered_fps=4,minimum_completed_fps=4)
        spec = self.root/"protocol.json"
        put(spec,self.p)
        self.resources.enter_context(patch.object(trial,"SPEC",spec))
        self.resources.enter_context(patch.object(trial,"verify",return_value=self.p))
        self.resources.enter_context(patch.object(trial,"Engine"))
        self.resources.enter_context(patch.object(trial,"pin_cpu"))
        self.resources.enter_context(patch.object(trial,"pipeline",side_effect=self.predict))
        self.roster, self.plan = self.root/"roster.json", self.root/"enrollment.json"
        campaign = dict(site="fixture",line="one",surface="upper",start_date="2026-01-01",end_date="2026-01-02",
                        reference_inspector="blinded",clock_id="common",rig_ids=["left","right"],frame_pitch_mm=10,
                        nominal_strip_speed_m_s=.5,full_width_nonoverlapping=True,matched_isolated_rigs=True)
        coils = []
        for i in range(50):
            targets = [] if i < 10 else [{"object_id":c,"class":c,"frame_index":9} for c in trial.CRITICAL]
            coils.append(dict(acquisition_group=f"coil{i}",strip_id=f"strip{i}",grade="G1",shift="day",
                              n_frames=50,whole_strip_clean=i<10,critical_objects=targets))
        put(self.roster,dict(policy_sha256=sha(spec),independent_blinded_census=True,campaign=campaign,coils=coils))
        self.plan_sha = trial.freeze_roster(self.p,self.roster,self.plan)
        self.units = read(self.plan)["units"]
        self.frames, self.runs = [], {}
        for cid, unit in self.units.items():
            arm = unit["arm"]
            if arm == "timing":
                self.runs[cid] = dict(rig_by_model=unit["rig_by_model"],clock_id="common",
                                      graph_sha256={m:a["graph_sha256"] for m,a in self.p["models"].items()})
            for i in range(unit["count"]):
                code = {"clean":i+1, "crazing":3, "rolled_in_scale":4, "timing":5}[arm]
                image = self.root/f"{code}.png"
                cv2.imwrite(str(image),np.full((9,9),code,np.uint8))
                row = dict(id=f"{cid}-{i}",acquisition_group=cid,strip_id=unit["strip_id"],
                           frame_index=unit["start"]+i,image=image.name,image_sha=sha(image))
                if arm in trial.CRITICAL:
                    label = trial.CLASSES.index(arm)
                    row.update(classes=[label,label],object_ids=["other",unit["object_id"]],boxes=[[1,1,4,4],[5,5,8,8]])
                elif arm == "clean":
                    row.update(classes=[],object_ids=[],boxes=[])
                else:
                    t = 1000000+i*250000000
                    row.update(capture_ns=t,controllers={m:dict(status="ack",ack_ns=t+12_000_000,alarm=False)
                                                         for m in trial.MODELS})
                self.frames.append(row)

    @staticmethod
    def predict(engine, gray, entry, cutoffs):
        assert cutoffs == [entry["all_class_cutoff"]]
        code, candidate = int(gray[0,0]), entry["checkpoint"] == "B0"
        rows = []
        if code in (3,4):
            coords = [5,5,8,8] if (code == 3) == candidate else [1,1,4,4]
            rows = [[*coords,.90,0 if code == 3 else 4]]
        elif code == 2 and candidate:
            rows = [[1,1,4,4,.91,1]]
        return np.array(rows,np.float32).reshape(-1,6)

    def score(self):
        manifest = self.root/"observations.json"
        put(manifest,dict(policy_sha256=sha(trial.SPEC),enrollment=self.plan.name,enrollment_sha256=self.plan_sha,
                          roster=self.roster.name,frames=self.frames,timing_runs=self.runs))
        return trial.evaluate(self.data,self.root,manifest)

    def test_paired_primary_objects_clean_windows_and_complete_blocks(self):
        out = self.score()
        pairs = out["paired_coil_outcomes"]
        self.assertEqual((pairs["crazing"]["B0_only"], pairs["rolled_in_scale"]["A0_only"]), (1,1))
        self.assertEqual(pairs["clean"]["B0_minus_A0_pp"],100)
        self.assertEqual(pairs["completed_fps"]["both"],2)
        self.assertTrue(out["paired_study_complete"])
        self.assertFalse(out["models"]["B0"]["passes_supervised_screen"])
        self.assertGreater(out["repeated_image_content"],0)
        self.assertEqual(len({r["rig_by_model"]["B0"] for r in self.runs.values()}),2)
        self.assertEqual(trial.freeze_roster(self.p,self.roster,self.plan),self.plan_sha)

    def test_missing_trace_is_model_specific_and_partial_pairs_are_indeterminate(self):
        row = next(r for r in self.frames if "controllers" in r)
        del row["controllers"]["A0"]
        out = self.score()
        self.assertFalse(out["paired_study_complete"])
        self.assertIsNone(out["models"]["A0"]["passes_supervised_screen"])
        self.assertEqual(out["models"]["B0"]["endpoints"]["completed_fps"]["observed"],2)
        self.assertIsNone(out["paired_coil_outcomes"]["completed_fps"]["mcnemar_two_sided_p"])
        self.frames = [r for r in self.frames if not (r["id"].endswith("-1") and r.get("object_ids") == [])]
        clean = self.score()["models"]["B0"]["endpoints"]["clean"]
        self.assertEqual(clean["observed"],0)
        self.assertIsNone(clean["met"])

    def test_invalid_trace_identity_and_benchmark_overlap(self):
        row = next(r for r in self.frames if "controllers" in r)
        with patch.dict(row["controllers"]["B0"], alarm=True):
            with self.assertRaisesRegex(ValueError,"alarm"):
                self.score()
        with patch.dict(row, capture_ns=-1):
            with self.assertRaisesRegex(ValueError,"trigger"):
                self.score()
        self.frames.append(deepcopy(self.frames[0]))
        with self.assertRaisesRegex(ValueError,"duplicate coil"):
            self.score()
        self.frames.pop()
        put(self.data/"train.json",[dict(image_sha=self.frames[0]["image_sha"])])
        with self.assertRaisesRegex(ValueError,"benchmark image"):
            self.score()

    def test_enrollment_substitution_or_unfilled_pool_fails(self):
        roster = read(self.roster)
        roster["campaign"]["site"] = "changed after enrollment"
        put(self.roster,roster)
        with self.assertRaisesRegex(ValueError,"frozen selection"):
            self.score()
        with self.assertRaisesRegex(ValueError,"different design"):
            trial.freeze_roster(self.p,self.roster,self.plan)
        roster["coils"] = [r for r in roster["coils"] if r["whole_strip_clean"]]
        put(self.roster,roster)
        with self.assertRaisesRegex(ValueError,"requires"):
            trial.freeze_roster(self.p,self.roster,self.root/"second.json")
