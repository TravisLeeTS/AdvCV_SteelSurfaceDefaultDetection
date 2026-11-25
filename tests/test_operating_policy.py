"""Protect equal-score ties, shared FP allocation, and lexicographic threshold selection."""
from itertools import product
from fractions import Fraction
import unittest

import numpy as np

from cpu_study.class_thresholds import fit_vectors
from cpu_study.head_calibration import NO_DETECTIONS


class ThresholdTests(unittest.TestCase):
    def test_both_objectives_against_full_enumeration(self):
        # Two scores tie within a class and the same numeric score occurs in other classes.
        scores = np.array([.9,.8,.8,.7,.91,.8,.65,.93,.83,.76,.91,.77,.9,.8,.92,.82], np.float32)
        labels = np.repeat(range(6), [4,3,3,2,2,2])
        tp = np.array([0,1,0,1,1,0,1,0,1,1,1,0,0,1,1,1], bool)
        gt = np.array([3,4,6,3,2,2])
        items = [(scores, labels, tp[:,None], gt)]
        options = [[NO_DETECTIONS, *np.unique(scores[labels == c]).astype(float)] for c in range(6)]
        for budget in (0, 2, 3):
            best = {"class_macro": None, "class_minimum": None}
            for cuts in product(*options):
                keep = scores >= np.asarray(cuts)[labels]
                cost = int((keep & ~tp).sum())
                if cost > budget:
                    continue
                recall = [Fraction(int((keep & tp & (labels == c)).sum()), int(gt[c])) for c in range(6)]
                macro = sum(recall)
                for policy, value in (("class_macro", (macro,-cost,cuts)),
                                      ("class_minimum",(min(recall),macro,-cost,cuts))):
                    if best[policy] is None or value > best[policy]:
                        best[policy] = value
            actual = fit_vectors(items,budget)
            for policy in best:
                self.assertEqual(actual[policy], list(best[policy][-1]))

    def test_empty_predictions(self):
        items = [(np.array([]), np.array([],int), np.empty((0,1),bool), np.ones(6,int))]
        self.assertEqual(fit_vectors(items, 0)["class_minimum"], [NO_DETECTIONS]*6)
