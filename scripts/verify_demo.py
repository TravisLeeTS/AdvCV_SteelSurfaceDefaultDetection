"""Exercise the real demo controls against frozen predictions for all five models."""
import argparse
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from steel import demo as app
from steel.bench import read_predictions
from steel.common import put, read, sha


def check(data, out):
    study=out/"study"
    dest=study/"report"
    frozen=read(study/"freeze.json")
    cases=read(dest/"figures/demo_cases.json")
    sample=cases["cases"]["success"]["id"]
    models=frozen["models"]
    selected=frozen["selection"]["model"]
    side=frozen["selection"]["side"]
    record=[]
    predictions={}
    draw=app.pair_image
    def capture(row,gray,detections,title,footer):
        parts=[x.strip() for x in title.split("|")]
        fields=parts[1].split()
        run=fields[0]
        head="one" if "one-to-one" in fields else "dense"
        size=int(fields[fields.index("px")-1])
        cell=f"{run}-openvino-{head}-{size}"
        if cell not in predictions:
            predictions[cell]=dict(read_predictions(study/"test"/cell/"predictions.jsonl.gz","operating"))
        np.testing.assert_allclose(detections,predictions[cell][row["id"]],atol=1e-5,rtol=0)
        assert f"Cutoff {frozen['cutoffs'][run]:.4f}" in footer and "Decoded image to final boxes" in footer
        picture=draw(row,gray,detections,title,footer)
        first=not record
        record.append(dict(cell=cell,image=row["id"],boxes=len(detections),shape=picture.size,
                           cutoff=frozen["cutoffs"][run],title=title))
        if first:
            picture.save(dest/"demo.png")
        return picture
    def exercise(root,*_):
        root.withdraw()
        try:
            toolbar=next(w for w in root.winfo_children() if isinstance(w,ttk.Frame))
            image,model,head,size=[w for w in toolbar.winfo_children() if isinstance(w,ttk.Combobox)]
            assert tuple(model.cget("values"))==tuple(models)
            assert (model.get(),int(size.get()),image.get())==(selected,side,sample)
            def select(widget,value):
                widget.set(value);widget.event_generate("<<ComboboxSelected>>");root.update()
            for m in models:
                select(model,m)
                assert record[-1]["cell"]==f"{m}0-openvino-one-{side}"
            select(model,selected)
            select(size,"256");select(head,"dense")
            assert size.get()=="480" and record[-1]["cell"]==f"{selected}0-openvino-dense-480"
            select(head,"one");select(size,"320")
            select(image,cases["cases"]["miss"]["id"])
            if "false_positive" in cases["cases"]: select(image,cases["cases"]["false_positive"]["id"])
            before=image.get()
            next(w for w in toolbar.winfo_children() if isinstance(w,ttk.Button)).invoke()
            assert image.get()!=before and record[-1]["image"]==image.get()
        finally:
            root.destroy()
    args=SimpleNamespace(data=data,artifacts=out,phase="runs",split="test",models=models,seeds=[0],
                         side=None,head="one",image_id=sample,output=None,threads=4)
    with patch.object(app,"pair_image",capture),patch.object(tk.Tk,"mainloop",exercise):
        app.demo(args)
    put(dest/"demo-check.json",dict(selected_model=selected,selected_side=side,ui_models=models,
        transitions=record,headless_image="demo.png",output_sha=sha(dest/"demo.png"),
        evidence="Actual Tk model, image, head, size and Next controls, including dense at 480. Every displayed box matched the saved operating predictions."))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    args=parser.parse_args()
    check(args.data,args.artifacts)
