import unittest

import numpy as np
import torch

from cpu_study.head_calibration import select_budgets, verify_floor
from cpu_study.models import DenseOutput
from steel.metrics import choose_threshold, events


def calibration_sample(score, false_score, count):
    rows, boxes = [], []
    for i in range(60):
        label = i % 6
        rows.append(dict(id=str(i), classes=[label], boxes=[[0,0,10,10]]))
        detections = [[0,0,10,10,score,label]]
        if i < count:
            detections.append([14,14,18,18,false_score,label])
        boxes.append(np.asarray(sorted(detections,key=lambda d:-d[4]),np.float32))
    return rows, boxes


class ThresholdsAndDenseDecode(unittest.TestCase):
    def test_heads_can_use_different_cutoffs_at_the_same_fp_cap(self):
        for level in (.60,.75):
            rows, predictions = calibration_sample(level,.91,6)
            options = select_budgets(rows,predictions)
            selected = next(r for r in options if r["fp_budget_per_image"] == .10)
            self.assertEqual(selected["validation_fp"],6)
            self.assertEqual(selected["validation_macro_recall"],1)
            self.assertEqual(selected["cutoff"],float(np.float32(level)))
            self.assertEqual(selected["cutoff"],choose_threshold(events(rows,predictions)))

    def test_an_entire_tied_score_group_must_fit_the_budget(self):
        rows, predictions = calibration_sample(.6,.6,7)
        selected = next(r for r in select_budgets(rows,predictions) if r["fp_budget_per_image"] == .10)
        self.assertGreater(selected["cutoff"],1)
        self.assertEqual(selected["validation_fp"],0)
        self.assertEqual(selected["validation_macro_recall"],0)

    def test_retention_limit_must_not_clip_the_operating_point(self):
        rows, predictions = calibration_sample(.7,.9,6)
        predictions[0]=np.repeat(predictions[0][:1],300,axis=0)
        with self.assertRaisesRegex(ValueError,"cap"):
            verify_floor(rows,predictions,.7)

    def test_original_v5_multiplies_by_objectness_before_filtering(self):
        raw = torch.tensor([[[2.,3.,4.,4.,.25,.8,.4,.2,.6,.12,.16]]])
        class Detector(torch.nn.Module):
            def forward(self, image):
                return raw, ()
        decoded=DenseOutput(Detector(),"yolov5s")(torch.zeros(1,3,1,1))
        self.assertEqual(tuple(decoded.shape),(1,10,1))
        torch.testing.assert_close(decoded[0,:4,0],raw[0,0,:4])
        torch.testing.assert_close(decoded[0,4:,0],raw[0,0,5:]*.25)
