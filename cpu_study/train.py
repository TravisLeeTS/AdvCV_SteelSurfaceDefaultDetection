"""Train the two historical CPU comparators on the original grouped manifests."""
import argparse
import copy
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
from filelock import FileLock
import numpy as np
import torch

from steel.common import SCORE_FLOOR, digest, put, read, sha, table
from steel.data import gray_image, load_rows
from steel.geometry import letterbox
from steel.metrics import match_image, summarize
from steel.post import finish
from steel.train import atom_save, batches, image_batch, recover_pending, training_config
from .models import DenseOutput, WEIGHTS, V5_REV, criterion, initialize


@torch.inference_mode()
def validate(net, family, rows, prepared, batch, device):
    view = DenseOutput(net.eval(), family).eval()
    observations = []
    for start in range(0, len(rows), batch):
        chunk = prepared[start:start + batch]
        pixels = image_batch(np.stack([x for x, _ in chunk]), device)
        outputs = view(pixels).cpu().numpy()
        for row, raw, (_, transform) in zip(rows[start:start + batch], outputs, chunk, strict=True):
            det, _ = finish(raw[None], "dense", SCORE_FLOOR, transform, (row["height"], row["width"]))
            observations.append(match_image(row, det))
    return summarize(observations)


def fit(data, out, weights, family, seed, epochs=200, batch=16, device="cuda"):
    run = out / "runs" / f"{family}-{seed}"
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / "training.lock", timeout=0):
        return train_locked(data, out, weights, run, family, seed, epochs, batch, device)


def train_locked(data, out, weights, run, family, seed, epochs, batch, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base = Path(__file__).parent
    config = training_config(data, family, seed, epochs, batch, device, "cpu-extension")
    config.update(selection="highest native FP32 dense+NMS validation mAP50-95 at 480, earliest tie",
                  extension_code={p:sha(base/p) for p in ("models.py","train.py")},
                  protocol_sha=sha(base/"protocol.json"), pretrained_sha=WEIGHTS[family][2],
                  implementation=V5_REV if family=="yolov5s" else config["ultralytics"])
    key = digest(config)
    done, last, best = (run/n for n in ("complete.json","last.pt","best.pt"))
    if (run/"config.json").exists() and read(run/"config.json") != config:
        raise ValueError(f"Training configuration differs in {run}")
    if done.exists():
        record = read(done)
        if record["config_id"] != key or record["best_sha"] != sha(best):
            raise ValueError(f"Completed checkpoint has changed: {run}")
        return record
    put(run/"config.json", config)
    net, transfer = initialize(family, seed, out, weights)
    put(run/"transfer.json", transfer)
    net.to(device)
    compute_loss = criterion(net, family, out, epochs, batch)
    groups = [[], []]
    for param in net.parameters():
        if param.requires_grad:
            groups[int(param.ndim == 1)].append(param)
    opt = torch.optim.AdamW([{"params":groups[0]}, {"params":groups[1],"weight_decay":0.0}],
                             lr=config["lr"], weight_decay=config["weight_decay"], betas=tuple(config["betas"]))
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda",enabled=amp,init_scale=1024.0)
    rows, val = (load_rows(data, s) for s in ("train","val"))
    images = [gray_image(data, r) for r in rows]
    prepared = [letterbox(gray_image(data, r),480) for r in val]
    high, history, start = -1.0, [], 0
    recover_pending(last, key, net)
    if last.exists():
        state = torch.load(last,map_location="cpu",weights_only=True)
        if state["config_id"] != key:
            raise ValueError("Training resume configuration differs")
        net.load_state_dict(state["weights"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, high, start = state["history"], state["high"], state["epoch"] + 1
        torch.set_rng_state(state["cpu_rng"])
        if amp:
            torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    if amp:
        torch.cuda.reset_peak_memory_stats()
    steps = math.ceil(len(rows)/batch)
    total = steps * epochs
    warm = max(1, min(3*steps, int(total*0.10)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for epoch in range(start, epochs):
            tic = perf_counter()
            net.train()
            order = np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(rows))
            accumulated_loss, rejected = 0.0, np.zeros(6,np.int64)
            for j,(images_batch, classes, boxes, indices, dropped) in enumerate(batches(pool,order,batch,rows,images,seed,epoch)):
                step = epoch*steps+j
                scale = ((step+1)/warm if step<warm else
                         0.01+0.99*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))/2)
                for group in opt.param_groups:
                    group["lr"] = config["lr"]*scale
                sample = dict(img=image_batch(images_batch,device), cls=torch.from_numpy(classes[:,None]).to(device),
                              bboxes=torch.from_numpy(boxes).to(device), batch_idx=torch.from_numpy(indices).to(device))
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,enabled=amp):
                    loss, _ = compute_loss(sample)
                    loss = loss.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{run.name} nonfinite loss at epoch {epoch+1}, step {j+1}")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(),config["gradient_clip"])
                scaler.step(opt)
                scaler.update()
                accumulated_loss += float(loss.detach())
                rejected += dropped
            quality = validate(net,family,val,prepared,batch,device)
            score = quality["map5095"]
            row = dict(epoch=epoch+1, seconds=perf_counter()-tic, loss=accumulated_loss/steps,
                       map50=quality["map50"],map5095=score,lr=opt.param_groups[0]["lr"],
                       peak_vram_gib=torch.cuda.max_memory_allocated()/2**30 if amp else 0,
                       **{f"rejected_{i}":int(n) for i,n in enumerate(rejected)})
            history.append(row)
            snap = dict(family=family,seed=seed,config_id=key,epoch=epoch,yaml=copy.deepcopy(net.yaml),
                        weights=net.state_dict(),validation=quality)
            if score > high:
                high = score
                atom_save(best,snap)
            atom_save(last,dict(**snap,history=history,high=high,optimizer=opt.state_dict(),
                                scaler=scaler.state_dict(),cpu_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state() if amp else None))
            table(run/"epochs.csv",history)
            print(f"{run.name} epoch {epoch+1}/{epochs}: val mAP50-95 {score:.5f}, {row['seconds']:.1f}s",flush=True)
    record = dict(run=run.name,epochs=epochs,config_id=key,best_sha=sha(best),best_map5095=high)
    put(done,record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/cpu-extension"))
    parser.add_argument("--weights",type=Path,default=Path("weights"))
    parser.add_argument("--families",nargs="+",choices=list(WEIGHTS),default=list(WEIGHTS))
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args = parser.parse_args()
    for seed in args.seeds:
        for family in args.families:
            fit(args.data,args.out,args.weights,family,seed,args.epochs,args.batch,args.device)


if __name__ == "__main__":
    main()
