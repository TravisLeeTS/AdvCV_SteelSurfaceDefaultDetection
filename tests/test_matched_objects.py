import unittest

import numpy as np

from scripts.matched_objects import CellMatches, choose_examples, example_detections, paired_counts
from steel.common import CLASSES


class MatchedObjectTests(unittest.TestCase):
    def cell(self, head, hits):
        meta = dict(split="test", seed=0, side=480, precision="FP32", split_sha="split",
                    model_id="E", runtime="openvino", head=head, cutoff=0.8,
                    checkpoint_sha="E0", artifact_sha=head)
        return CellMatches(meta, np.array(hits, dtype=bool), np.zeros(len(CLASSES), dtype=int), head)

    def test_equal_totals_can_hide_different_detected_objects(self):
        labels = np.zeros(6, dtype=int)
        dense = self.cell("dense", [True, True, False, False, True, False])
        one = self.cell("one", [True, False, True, False, False, True])
        result = list(paired_counts("head", "dense", "one", dense, one, labels))
        self.assertEqual(len(result), len(CLASSES))
        first = result[0]
        self.assertEqual((first["both"], first["reference_only"], first["candidate_only"], first["neither"]),
                         (1, 2, 2, 1))
        self.assertEqual((first["reference_tp"], first["candidate_tp"], first["recall_change_pp"]), (3, 3, 0))
        with self.assertRaises(ValueError):
            list(paired_counts("architecture", "dense", "one", dense, one, labels))

    def test_examples_follow_annotation_order_and_keep_both_directions(self):
        rows = [dict(id="first", classes=[1, 0, 0]), dict(id="second", classes=[4, 0])]
        dense = self.cell("dense", [True, False, True, False, True])
        one = self.cell("one", [False, True, False, True, False])
        winners = choose_examples(rows, dense, one)
        self.assertEqual(winners, {"one": (rows[0], 1), "dense": (rows[0], 2)})

    def test_score_floor_diagnostic_uses_the_requested_target(self):
        row = dict(classes=[0, 0], boxes=[[0, 0, 10, 10], [20, 20, 30, 30]])
        first = [0, 0, 10, 10, 0.95, 0]
        second = [20, 20, 30, 30, 0.79, 0]
        record = dict(operating=[first], low=[first, second])
        self.assertEqual(example_detections(row, 0, record, 0.8)["status"], "matched")
        other = example_detections(row, 1, record, 0.8)
        self.assertEqual(other["status"], "below cutoff")
        self.assertEqual(other["iou"], 1)
        self.assertAlmostEqual(other["score"], 0.79)
        self.assertIsNone(example_detections(row, 1, dict(operating=[], low=[first]), 0.8)["prediction"])
