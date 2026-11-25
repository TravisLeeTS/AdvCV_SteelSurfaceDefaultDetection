"""Exercise independent trial denominators and the missing-evidence cases."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from cpu_study import confirmation
from steel.common import put, read, sha


class ConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.data = self.root / "benchmark"
        for split in ("train","val","test"):
            put(self.data / f"{split}.json", [])
        policy = dict(read(confirmation.SPEC), clean_window_frames=2)
        self.resources.enter_context(patch.object(confirmation, "verify", return_value=policy))
        self.resources.enter_context(patch.object(confirmation, "pin_cpu"))
        self.resources.enter_context(patch.object(confirmation, "Engine"))
        self.resources.enter_context(patch.object(confirmation, "pipeline", side_effect=self.prediction))
        self.frames = []
        for index, labels in enumerate(([], [], [0], [4])):
            image = self.root / f"{index}.png"
            cv2.imwrite(str(image), np.full((8,8),index,np.uint8))
            self.frames.append(dict(id=f"frame{index}", acquisition_group=f"coil{index//2}", strip_id=f"strip{index//2}",
                frame_index=index%2, clean_strip=index<2, image=image.name, image_sha=sha(image), classes=labels,
                boxes=[[1,1,4,4] for _ in labels], capture_ns=100000000*index,
                controller_ack_ns=100000000*index+12000000, controller_alarm=index!=0))

    @staticmethod
    def prediction(engine, gray, entry, cuts):
        label = {0:None, 1:1, 2:0, 3:4}[int(gray[0,0])]
        return np.array([[1,1,4,4,.9,label]] if label is not None else [],np.float32).reshape(-1,6)

    def score(self):
        manifest = self.root / "manifest.json"
        put(manifest, dict(policy_sha256=sha(confirmation.SPEC), acquisition_source="unit test fixture",
                           independently_acquired=True, frames=self.frames))
        return confirmation.evaluate(self.data, self.root, manifest)

    def test_whole_strip_and_clock_denominators(self):
        result = self.score()
        self.assertEqual(result["object_recall"]["crazing"]["recall"], 1)
        self.assertEqual(result["object_recall"]["rolled_in_scale"]["recall"], 1)
        self.assertEqual((result["clean_strips"],result["rejected_clean_strips"],result["clean_rejection_fraction"]), (1,1,1))
        self.assertEqual(result["capture_to_controller_p95_ms"],12)
        self.assertFalse(result["all_criteria_met"])

    def test_unobserved_endpoints_remain_missing(self):
        self.frames = [self.frames[2]]
        for name in ("controller_alarm","capture_ns","controller_ack_ns"):
            del self.frames[0][name]
        result = self.score()
        self.assertIsNone(result["clean_rejection_fraction"])
        self.assertIsNone(result["capture_to_controller_p95_ms"])
        self.assertIsNone(result["criteria_met"]["rolled_in_scale"])
        self.assertIsNone(result["all_criteria_met"])

    def test_rejects_incomplete_window_and_benchmark_overlap(self):
        with patch.object(self,"frames",self.frames[1:]):
            with self.assertRaisesRegex(ValueError,"complete predeclared"):
                self.score()
        put(self.data/"train.json",[dict(image_sha=self.frames[0]["image_sha"])])
        with self.assertRaisesRegex(ValueError,"reused benchmark"):
            self.score()
