import copy
import math
import os
import random
import pickle
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2EDetectLoss

from .common import SCORE_FLOOR, digest, put, read, replace_file, sha, source_hash, table
from .data import as_arrays, gray_image, load_rows
from .geometry import augment, letterbox, to_yolo
from .metrics import match_image, summarize
from .models import initial
from .post import finish


def atom_save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    replace_file(tmp,path)


def recover_pending(last, config_id, net):
    temporary=last.with_suffix(".tmp")
    if not temporary.exists():
        return
    try:
        latest=torch.load(temporary,map_location="cpu",weights_only=True)
    except (OSError,RuntimeError,EOFError,pickle.UnpicklingError):
        if last.exists(): return  # an interrupted temporary write never displaces a complete file
        raise
    required={"optimizer","scaler","history","cpu_rng","cuda_rng","weights","epoch","config_id","high"}
    if not required <= latest.keys() or latest["config_id"]!=config_id or len(latest["history"])!=latest["epoch"]+1:
        raise ValueError(f"Invalid pending checkpoint at {temporary}")
    expected=net.state_dict()
    if latest["weights"].keys()!=expected.keys() or any(v.shape!=expected[k].shape for k,v in latest["weights"].items()):
        raise ValueError("Pending checkpoint tensor schema mismatch")
    previous=torch.load(last,map_location="cpu",weights_only=True)["epoch"] if last.exists() else -1
    if latest["epoch"]>previous:
        replace_file(temporary,last)
        print(f"Recovered completed epoch {latest['epoch']+1} from {temporary}",flush=True)


def training_config(data, model_id, seed, epochs, batch, device, phase):
    split = read(data / "split.json")
    for name in ("train", "val"):
        if sha(data / (name+".json")) != split["split_files"][name]:
            raise ValueError(f"Changed {name} manifest")
    return dict(model_id=model_id, seed=seed, epochs=epochs, batch=batch, device=device, phase=phase,
                split_sha=split["split_hash"], torch=str(torch.__version__), ultralytics=ultralytics.__version__,
                code_sha=source_hash("common.py", "models.py", "geometry.py", "metrics.py", "post.py", "train.py", "data.py"),
                optimizer="AdamW", lr=0.001, final_lr=1e-5, betas=[0.9, 0.999], weight_decay=0.0005,
                schedule="linear warmup up to 3 epochs (at most 10% of run), cosine to 1% of base",
                amp=device=="cuda", early_stopping=False, selection="highest native one-to-one FP32 validation mAP50-95 at 480, earliest tie",
                ema=False, gradient_clip=10.0, batch_order="numpy permutation SeedSequence([seed,epoch])",
                aug_rng="numpy SeedSequence([seed,epoch,train-manifest-index])",
                canvas=480, scale=0.20, translate=0.10, fliplr=0.5, brightness=0.15, contrast=0.15,
                multi_scale=False, mosaic=0, mixup=0, hsv=0, erasing=0, copy_paste=0,
                interpolation="bilinear", fill_gray=114, loader_threads=4)


def packed(ids, rows, images, seed, epoch):
    pixels, labels, targets, batch_idx = [], [], [], []
    rejected = np.zeros(6, np.int64)
    for j, i in enumerate(ids):
        box, cls = as_arrays(rows[i])
        canvas, box, cls, lost, _ = augment(images[i], box, cls, seed, epoch, int(i))
        pixels.append(canvas)
        labels.append(cls)
        targets.append(to_yolo(box, 480, 480))
        batch_idx.append(np.full(len(cls), j, dtype=np.int64))
        rejected += lost
    return (np.stack(pixels), np.concatenate(labels), np.concatenate(targets), np.concatenate(batch_idx), rejected)


def batches(pool, ids, size, rows, images, seed, epoch):
    chunks = iter(ids[i:i+size] for i in range(0, len(ids), size))
    waiting = deque()
    def queue(chunk):
        waiting.append(pool.submit(packed, chunk, rows, images, seed, epoch))
    for _ in range(min(4, math.ceil(len(ids)/size))):
        queue(next(chunks))
    while waiting:
        value = waiting.popleft().result()
        chunk = next(chunks, None)
        if chunk is not None:
            queue(chunk)
        yield value


def image_batch(array, device):
    return torch.from_numpy(array).to(device).unsqueeze(1).expand(-1, 3, -1, -1).contiguous().float().mul_(1/255)


@torch.inference_mode()
def validate(net, rows, prepared, batch, device):
    net.eval()
    net.end2end = True
    head = net.model[-1]
    if not head.end2end or head.export:
        raise ValueError("Invalid training-time validation branch")
    items = []
    for i in range(0, len(rows), batch):
        pack = prepared[i:i+batch]
        x = image_batch(np.stack([p[0] for p in pack]), device)
        raw, branches = net(x)
        if set(branches) != {"one2one", "one2many"} or raw.shape[-1] != 6:
            raise ValueError("Expected unfused native one-to-one validation output")
        for j, det in enumerate(raw.cpu().numpy()):
            row = rows[i+j]
            pred, _ = finish(det[None], "one", SCORE_FLOOR, pack[j][1], (row["height"], row["width"]))
            items.append(match_image(row, pred))
    return summarize(items)


def fit(data, out, weights, model_id, seed, epochs, batch=16, device="cuda", phase="runs"):
    from filelock import FileLock
    run=Path(out)/phase/f"{model_id}{seed}"
    run.mkdir(parents=True,exist_ok=True)
    with FileLock(run/"training.lock",timeout=0):
        return _fit(data,out,weights,model_id,seed,epochs,batch,device,phase)


def _fit(data, out, weights, model_id, seed, epochs, batch, device, phase):
    data, out = Path(data), Path(out)
    run = out / phase / f"{model_id}{seed}"
    run.mkdir(parents=True, exist_ok=True)
    config = training_config(data, model_id, seed, epochs, batch, device, phase)
    done, last, best = (run / n for n in ("complete.json", "last.pt", "best.pt"))
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
    net, transfer = initial(model_id, seed, weights)
    config["pretrained_sha"] = transfer["source_sha"]
    config["paired_initial_sha"] = transfer["paired_initial_sha"]
    implementation=config["code_sha"]
    if (run/"config.json").exists():
        previous=read(run/"config.json")
        if previous!=config:
            approved=read(Path(__file__).with_name("training_compatibility.json"))
            equivalent=any(r["from"]==previous["code_sha"] and r["to"]==implementation for r in approved)
            if not equivalent or dict(previous,code_sha=implementation)!=config:
                raise ValueError(f"Run inputs differ at {run}. Use a separate artifact root.")
            config=previous  # retain the exact checkpoint identity and record the execution revision separately
    key=digest(config)
    if done.exists() and read(done)["config_id"] == key and sha(best) == read(done)["best_sha"]:
        print(f"Verified completed {run}", flush=True)
        return read(done)
    put(run/"config.json", config)
    put(run/"transfer.json", transfer)
    net.args = get_cfg(overrides={"epochs": epochs, "imgsz": 480, "batch": batch, "scale": 0.2,
                                  "mosaic": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0})
    net.to(device)
    net.criterion = E2EDetectLoss(net) if model_id == "E" else net.init_criterion()
    criterion = net.criterion
    decay, plain = [], []
    for p in net.parameters():
        if p.requires_grad:
            (plain if p.ndim == 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay}, {"params": plain, "weight_decay": 0.0}],
                            lr=config["lr"], weight_decay=config["weight_decay"], betas=tuple(config["betas"]))
    amp = config["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    rows, val = load_rows(data, "train"), load_rows(data, "val")
    images = [gray_image(data, r) for r in rows]
    val_prepared = [letterbox(gray_image(data, r), 480) for r in val]
    history, high, start = [], -1.0, 0
    recover_pending(last,key,net)
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["config_id"] != key:
            raise ValueError("Checkpoint/config mismatch")
        net.load_state_dict(state["weights"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, high, start = state["history"], state["high"], state["epoch"] + 1
        torch.set_rng_state(state["cpu_rng"])
        if amp:
            torch.cuda.set_rng_state(state["cuda_rng"])
        if hasattr(criterion, "update"):
            for _ in range(start):
                criterion.update()
        del state
    put(run/"execution_revision.json",dict(original_code_sha=config["code_sha"],active_code_sha=implementation,
                                           starting_epoch=start+1,config_id=key))
    if amp:
        torch.cuda.reset_peak_memory_stats()
    probe = [p["path"]+".kernel.weight" for p in transfer["replacements"]]
    for gate in transfer.get("gates", []):
        probe.extend(gate["path"] + "." + tensor for tensor in
                     ("reduce.weight", "reduce.bias", "project.weight", "project.bias"))
    head_prefix = f"model.{len(net.model)-1}."
    probe += [head_prefix+s+".0.2.weight" for s in ("cv2", "one2one_cv2", "cv3", "one2one_cv3")]
    params = dict(net.named_parameters())
    if not all(k in params for k in probe):
        raise ValueError("Expected trainable probe paths are missing")
    need_gradient_check=start==0 or not (run/"gradient_check.json").exists()
    before = {k: params[k].detach().clone() for k in probe} if need_gradient_check else {}
    steps = math.ceil(len(rows)/batch)
    total_steps = epochs * steps
    warm = max(1, min(3*steps, int(total_steps*0.10)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for epoch in range(start, epochs):
            tic = perf_counter()
            net.train()
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(rows))
            sums, lost, gradient = {}, np.zeros(6, np.int64), {}
            for j, (x, c, b, bi, dropped) in enumerate(batches(pool, order, batch, rows, images, seed, epoch)):
                step = epoch*steps + j
                scale = ((step+1)/warm if step < warm else
                         0.01 + 0.99*(1+math.cos(math.pi*(step-warm)/max(1,total_steps-warm-1)))/2)
                for group in opt.param_groups:
                    group["lr"] = config["lr"] * scale
                batch_tensor = dict(img=image_batch(x, device), cls=torch.from_numpy(c[:, None]).to(device),
                                    bboxes=torch.from_numpy(b).to(device), batch_idx=torch.from_numpy(bi).to(device))
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=amp):
                    loss, parts = net.loss(batch_tensor)
                    total_loss = loss.sum()
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {j}")
                scaler.scale(total_loss).backward()
                scaler.unscale_(opt)
                if need_gradient_check and epoch==start and len(gradient)<len(probe):
                    for k in probe:
                        if k not in gradient and params[k].grad is not None:
                            norm = float(params[k].grad.float().norm())
                            if math.isfinite(norm) and norm > 0:
                                gradient[k] = norm
                torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"])
                scaler.step(opt)
                scaler.update()
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                lost += dropped
            quality = validate(net, val, val_prepared, batch, device)
            seconds = perf_counter()-tic
            score = quality["map5095"]
            record = dict(epoch=epoch+1, seconds=seconds, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30 if amp else 0,
                          map50=quality["map50"], map5095=score, lr=opt.param_groups[0]["lr"],
                          **{k:v/steps for k, v in sums.items()}, **{f"rejected_{i}":int(v) for i,v in enumerate(lost)})
            history.append(record)
            if need_gradient_check and epoch==start:
                updated = {k: float((params[k].detach()-before[k]).abs().max()) for k in probe}
                if len(gradient) != len(probe) or not all(v > 0 for v in updated.values()):
                    raise ValueError(f"Missing finite gradients or weight updates: {gradient}, {updated}")
                put(run/"gradient_check.json", dict(epoch=epoch+1, max_weight_changes=updated, grad_norm=gradient,
                                                    criterion=type(criterion).__name__, validation_head="one2one"))
                before.clear()
            if hasattr(criterion, "update"):
                criterion.update()
            snapshot = dict(model_id=model_id, seed=seed, yaml=copy.deepcopy(net.yaml), config_id=key, epoch=epoch,
                            weights=net.state_dict(), validation=quality)
            if score > high:
                high = score
                atom_save(best, snapshot)
            atom_save(last, dict(**snapshot, history=history, high=high, optimizer=opt.state_dict(),
                                 scaler=scaler.state_dict(), cpu_rng=torch.get_rng_state(),
                                 cuda_rng=torch.cuda.get_rng_state() if amp else None))
            table(run/"epochs.csv", history)
            print(f"{phase}/{model_id}{seed} epoch {epoch+1}/{epochs}  {seconds:.1f}s  val mAP50-95 {score:.4f}", flush=True)
    result = dict(config_id=key, best_sha=sha(best), best_map5095=high, epochs=epochs, run=f"{phase}/{model_id}{seed}")
    put(done, result)
    return result
