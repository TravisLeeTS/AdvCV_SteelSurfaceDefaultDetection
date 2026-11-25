from pathlib import Path
from time import perf_counter_ns

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .bench import Engine, calibration
from .common import CLASSES, read
from .data import as_arrays, gray_image, load_rows
from .geometry import prepare
from .hardware import pin_cpu
from .post import finish


def font(size):
    for candidate in ("C:/Windows/Fonts/segoeui.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(candidate,size)
        except OSError: pass
    return ImageFont.load_default(size=size)


def panel(gray, boxes, labels, scores=None, size=400):
    h,w=gray.shape
    im=Image.fromarray(gray).convert("RGB").resize((size,size),Image.Resampling.BILINEAR)
    draw=ImageDraw.Draw(im)
    color=(255,209,70) if scores is None else (42,224,222)
    annotations=[]
    occupied=[]
    f=font(13)
    for i,(box,c) in enumerate(zip(boxes,labels,strict=True)):
        x1,y1,x2,y2=np.clip(box*np.array([size/w,size/h,size/w,size/h]),0,size-1)
        draw.rectangle((x1,y1,x2,y2),outline=color,width=2)
        label=CLASSES[int(c)].replace("_"," ")
        if scores is not None: label+=f" {scores[i]:.2f}"
        tw=draw.textlength(label,font=f)
        x=max(0,min(x1,size-tw-8))
        possible=[y1-19,y1,y2-19,y2,*range(0,size-18,20)]
        def cost(y):
            return sum(max(0,min(x+tw+7,b)-max(x,a))*max(0,min(y+18,d)-max(y,c))
                       for a,c,b,d in occupied)
        y=min((max(0,min(p,size-19)) for p in possible),key=cost)
        occupied.append((x,y,x+tw+7,y+18))
        annotations.append((x,y,tw,label))
    for x,y,tw,label in annotations:
        draw.rectangle((x,y,x+tw+7,y+18),fill=(22,28,33))
        draw.text((x+3,y),label,fill=color,font=f)
    return im


def pair_image(row,gray,prediction,title,footer):
    boxes,labels=as_arrays(row)
    image=Image.new("RGB",(838,500),(249,249,246))
    draw=ImageDraw.Draw(image)
    draw.text((14,7),title,font=font(20),fill=(21,31,43))
    draw.text((14,40),f"Ground truth  ({len(boxes)})",font=font(15),fill=(30,40,49))
    draw.text((424,40),f"Predictions  ({len(prediction)})",font=font(15),fill=(30,40,49))
    image.paste(panel(gray,boxes,labels),(14,66))
    image.paste(panel(gray,prediction[:,:4],prediction[:,5],prediction[:,4]),(424,66))
    draw.text((14,475),footer,font=font(14),fill=(30,40,49))
    return image


def demo(args):
    data,out=args.data,args.artifacts
    study=out/("study" if args.phase=="runs" else "pilot-study")
    frozen=read(study/"freeze.json") if (study/"freeze.json").exists() else None
    if args.split=="test":
        from .report import verify_freeze
        if not frozen: raise ValueError("Test demo requires a frozen study")
        verify_freeze(data,out,frozen["models"],frozen["seeds"])
    selected=frozen["selection"]["model"] if frozen else None
    seed=args.seeds[0]
    available=[m for m in args.models if (out/args.phase/f"{m}{seed}"/"best.pt").is_file()]
    if not available:
        raise ValueError(f"No trained checkpoint for seed {seed} and requested models {args.models}")
    model=selected if selected in available else available[0]
    selected_side=frozen["selection"]["side"] if frozen and model==selected else None
    start_side=args.side or selected_side or 480
    if args.head=="dense" and args.side is None:
        start_side=480
    rows=load_rows(data,args.split)
    by_id={r["id"]:r for r in rows}
    initial_id=args.image_id or rows[0]["id"]
    if initial_id not in by_id:
        raise ValueError(f"{initial_id} is outside the frozen {args.split} split. An available image is {rows[0]['id']}.")
    pin_cpu(args.threads)
    engines,cutoffs={},{}
    def render(image_id,model_id,head,side):
        from .export import export_cell
        if head=="dense" and side!=480: raise ValueError("Dense-head demo uses the 480-pixel artifact")
        row=by_id[image_id]
        gray=gray_image(data,row)
        run=out/args.phase/f"{model_id}{seed}"
        key=(model_id,head,side)
        with torch.inference_mode():
            if model_id not in cutoffs:
                cutoffs[model_id]=(frozen["cutoffs"][run.name] if frozen else calibration(data,study,run,args.threads)["cutoff"])
            cutoff=cutoffs[model_id]
            if key not in engines:
                export_cell(data,run,head,side,args.threads)
                engine=engines[key]=Engine(run,"openvino",head,side,args.threads)
                x,_=prepare(gray,side)
                engine.forward(engine.tensor(x))
            engine=engines[key]
            start=perf_counter_ns()
            x,t=prepare(gray,side)
            pred,_=finish(engine.forward(engine.tensor(x)),head,cutoff,t,gray.shape)
            elapsed=(perf_counter_ns()-start)/1e6
        label="one-to-one" if head=="one" else "dense + NMS"
        title=f"{row['id']}   |   {model_id}{seed}   {label}   {side} px   OpenVINO FP32"
        footer=f"Decoded image to final boxes   {elapsed:.2f} ms (one frame)     Cutoff {cutoff:.4f} (shared native 480)     {args.split}"
        return pair_image(row,gray,pred,title,footer)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        render(initial_id,model,args.head,start_side).save(args.output)
        print(args.output)
        return
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk
    root=tk.Tk()
    root.title("MLR 512 | Steel surface inspection")
    toolbar=ttk.Frame(root,padding=10)
    toolbar.pack(fill="x")
    image_id=tk.StringVar(value=initial_id)
    architecture=tk.StringVar(value=model)
    head=tk.StringVar(value=args.head)
    side=tk.StringVar(value=str(start_side))
    controls=[("Image",image_id,list(by_id),22),("Model",architecture,available,4),("Detection head",head,["one","dense"],7),
              ("Input size",side,["256","320","480","640"],6)]
    view=ttk.Label(root)
    view.pack()
    def update(*_):
        if head.get()=="dense": side.set("480")
        pic=ImageTk.PhotoImage(render(image_id.get(),architecture.get(),head.get(),int(side.get())))
        view.configure(image=pic)
        view.image=pic
    for label,var,options,width in controls:
        ttk.Label(toolbar,text=label).pack(side="left",padx=5)
        widget=ttk.Combobox(toolbar,textvariable=var,values=options,width=width,state="readonly")
        widget.pack(side="left")
        widget.bind("<<ComboboxSelected>>",update)
    ttk.Button(toolbar,text="Next image",command=lambda:(image_id.set(rows[(list(by_id).index(image_id.get())+1)%len(rows)]["id"]),update())).pack(side="left",padx=12)
    update()
    root.mainloop()
