import unittest
from pathlib import Path
import numpy as np

from steel.common import CLASSES,matrix
from steel.bench import reference_suppression
from steel.post import nms_inputs,suppress
from steel.geometry import from_yolo,letterbox,project,to_yolo
from steel.ir import bundle_directory,bundle_id
from steel.metrics import choose_threshold,events,summarize
from steel.report import decision,matched_heads
from steel.uncertainty import bootstrap_metrics
from steel.factorial import factorial_rows


class ProtocolTests(unittest.TestCase):
    def test_fusion_recovery_and_interaction_use_paired_seeds(self):
        cells={}
        for seed in range(3):
            for model,value in zip("ABCD",(.40,.36,.44,.40+.01*seed),strict=True):
                cells[("test",model,seed,"openvino","one",480)]=dict(
                    map5095=value,map50=value,pipeline_p95_ms=10,fp_per_image=.1,
                    cell_dir=f"{model}{seed}",checkpoint_sha=f"sha{model}{seed}",
                    **{c+"_recall":value for c in CLASSES})
        results=factorial_rows(cells,"test",[0,1,2])
        values={(r["contrast"],r["seed"]):r["estimate"] for r in results if r["metric"]=="map5095"}
        self.assertAlmostEqual(values["D-B",0],4)
        self.assertAlmostEqual(values["(D-B)-(C-A)",0],0)
        self.assertAlmostEqual(values["(D-B)-(C-A)","paired mean"],1)
        def draw(key):
            value=cells[key]["map5095"]
            return {metric:np.array([value-.02,value,value+.02])
                    for metric in ("map5095",*(c+"_recall" for c in CLASSES))}
        intervals=factorial_rows(cells,"test",[0,1,2],draw,group_count=3)
        interaction=next(r for r in intervals if r["contrast"]=="(D-B)-(C-A)" and
                         r["metric"]=="map5095" and r["seed"]=="paired mean")
        np.testing.assert_allclose([interaction["low"],interaction["high"]],[1,1])

    def test_nms_preserves_original_coordinate_precision_and_score_order(self):
        # Validation image pitted_surface_290: translating these boxes by a
        # class offset moves their FP32 IoU from 0.69999886 to 0.70000434.
        a=[408.46838,.034030914,491.87097,13.876141,.006984035,2]
        b=[424.66461,.16988945,487.28723,13.074589,.010317402,2]
        twin=[*b[:5],1]
        duplicate=[*b[:4],.005,2]
        rows=np.array([a,b,twin,duplicate],dtype=np.float32)
        for sample,expected in [(rows,[1,2,0]),(rows[:0],[])]:
            np.testing.assert_array_equal(suppress(nms_inputs(sample)).numpy(),expected)
            np.testing.assert_array_equal(reference_suppression(sample),expected)

    def test_ir_identity_includes_checkpoint_size_head_precision_and_weights(self):
        run=Path("runs/A0")
        shared=bundle_directory(run,"one",480,"a"*64)
        self.assertEqual(shared,run/"exports"/("a"*64)/"FP32/one-480")
        paths={shared,bundle_directory(run,"one",256,"a"*64),bundle_directory(run,"dense",480,"a"*64),
               bundle_directory(run,"one",480,"b"*64)}
        self.assertEqual(len(paths),4)
        ir={"xml_sha":"same topology","bin_sha":"weights one"}
        other=dict(ir,bin_sha="weights two")
        self.assertNotEqual(bundle_id("a"*64,"one",480,ir),bundle_id("a"*64,"one",480,other))

    def test_geometry_with_unequal_letterbox_rounding(self):
        gray=np.zeros((73,121),np.uint8)
        image,t=letterbox(gray,256)
        b=np.array([[0,0,121,73],[11,6,16,68]],np.float32)
        self.assertEqual(image.shape,(256,256))
        y=from_yolo(to_yolo(b,121,73),121,73)
        np.testing.assert_allclose(project(project(y,t),t,inverse=True),b,rtol=0,atol=1e-4)

    def test_iou_matching_and_tied_threshold_groups(self):
        rows=[dict(boxes=[[0,0,10,10]],classes=[i%6]) for i in range(12)]
        det=[[[0,0,10,10,.8,i%6]] for i in range(12)]
        # Two FPs appear at exactly the same score. Taking only one would fabricate
        # a threshold meeting the FP budget. The maximum-recall tie favors fewer FPs.
        det[1].extend([[20,20,25,25,.75,1],[30,30,35,35,.75,1]])
        event=events(rows,det)
        cutoff=choose_threshold(event)
        self.assertAlmostEqual(cutoff,.8)
        quality=summarize(event,cutoff)
        self.assertAlmostEqual(quality["map5095"],1)
        self.assertEqual(quality["fp_per_image"],0)
        repeated=bootstrap_metrics(event,cutoff,np.ones((1,len(rows)),np.int16))
        self.assertAlmostEqual(repeated["map5095"][0],1)

    def test_greedy_match_can_use_second_best_available_gt(self):
        rows=[dict(boxes=[[0,0,10,10],[2,0,12,10]],classes=[0,0])]
        det=[[[0,0,10,10,.9,0],[.5,0,10.5,10,.8,0]]]
        q=summarize(events(rows,det))
        self.assertEqual(q["classes"]["crazing"]["tp"],2)
        self.assertLess(q["map5095"],1)

    def test_group_bootstrap_weights_match_explicit_image_copies(self):
        rows=[dict(boxes=[[0,0,10,10]],classes=[i%6]) for i in range(12)]
        predictions=[[[13,13,20,20,.97-i*.012,i%6],[0,0,10-i%3,10,.33+i*.022,i%6]] for i in range(12)]
        items=events(rows,predictions)
        weights=np.array([0,2,2,1,1,1,1,1,1,2,1,1],dtype=np.int16)
        expanded=[event for event,weight in zip(items,weights,strict=True) for _ in range(weight)]
        explicit=summarize(expanded,.48)
        weighted=bootstrap_metrics(items,.48,weights[None,:])
        self.assertAlmostEqual(weighted["map5095"][0],explicit["map5095"])
        for defect in CLASSES:
            self.assertAlmostEqual(weighted[defect+"_recall"][0],explicit["classes"][defect]["recall"])
        for defect in ("crazing","rolled_in_scale"):
            self.assertAlmostEqual(weighted[defect+"_ap50"][0],explicit["classes"][defect]["ap50"])

    def test_matrix_and_selection_ties(self):
        self.assertEqual(len(set(matrix())),87)
        self.assertEqual(len(set(matrix(("A","B","E")))),57)
        self.assertEqual(len(set(matrix(("A","E")))),42)
        rows={}
        for m,k,rt,h,s in matrix():
            if (rt,h)!=("openvino","one"):continue
            p95={256:4,320:4.1,480:8,640:14}[s]-(.1 if m=="B" else 0)
            rows[("val",m,k,rt,h,s)]=dict(map5095=.50,fp_per_image=.09,crazing_ap50=.55,rolled_in_scale_ap50=.65,
                **{c+"_recall":.4 for c in CLASSES},pipeline_p95_ms=p95,p95_min_ms=p95-.2,p95_max_ms=p95+.2)
        picked=decision(rows,("A","B","E"),(0,1,2))
        self.assertEqual((picked["model"],picked["side"]),("A",256))
        rows[("val","A",0,"openvino","one",256)]["crazing_ap50"]-=.021
        self.assertEqual(decision(rows,("A","B","E"),(0,1,2))["side"],320)
        for v in rows.values(): v["fp_per_image"]=.2
        self.assertIsNone(decision(rows,("A","B","E"),(0,1,2))["model"])

    def test_saving_pairs_sessions_before_seed_summary(self):
        base=dict(cutoff=.2,checkpoint_sha="fixture",map5095=.5,crazing_ap50=.3,rolled_in_scale_ap50=.4,
                  fp_per_image=.1,nms_p50_ms=.01,nms_p95_ms=.02,head_p95_ms=.2,source="fixture",
                  **{c+"_recall":.5 for c in CLASSES})
        kd=("val","A",0,"openvino","dense",480)
        ko=("val","A",0,"openvino","one",480)
        rows={kd:dict(**base,pipeline_p95_ms=10),ko:dict(**base,pipeline_p95_ms=8)}
        times={kd:{"pipeline":[dict(p95_ms=x) for x in (7,10,20)]},ko:{"pipeline":[dict(p95_ms=x) for x in (8,1,22)]}}
        result=matched_heads(rows,times)[0]
        self.assertEqual(result["saving_ms"],-1)
        self.assertAlmostEqual(result["saving_percent"],-10)


if __name__=="__main__":
    unittest.main()
