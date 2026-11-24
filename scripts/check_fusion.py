"""Verify identity initialization and learned branch weights on real steel images."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
from ultralytics.cfg import get_cfg
from steel.common import CLASSES, put, read, sha
from steel.data import gray_image, load_rows
from steel.geometry import prepare
from steel.models import initial, restore


def check_outputs(left, right):
    """Check decoded boxes, both raw head dictionaries, and all feature levels."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        return 1
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return sum(check_outputs(left[key], right[key]) for key in left)
    return sum(check_outputs(a, b) for a, b in zip(left, right, strict=True))


def observe(net, images):
    observations = {index: [] for index in (12, 15)}
    def capture(index):
        def hook(module, inputs):
            observations[index].append(module.branch_weights(inputs[0]).cpu().numpy())
        return hook
    hooks = [net.model[i].register_forward_pre_hook(capture(i)) for i in observations]
    try:
        output = net.eval()(images)
    finally:
        for hook in hooks:
            hook.remove()
    return output, {i: np.concatenate(values) for i, values in observations.items()}


@torch.inference_mode()
def check(data, out, weights, learned, models=("C", "D"), seeds=(0, 1, 2)):
    torch.set_num_threads(2)
    split = "val" if learned else "train"
    rows = load_rows(data, split)
    chosen = [r for c in range(len(CLASSES)) for r in
              [x for x in rows if x["stratum"] == c][:(6 if learned else 1)]]
    images = torch.cat([torch.from_numpy(prepare(gray_image(data, r), 480)[0]) for r in chosen])
    results = {}
    loss_path=out/"audit/resolved_losses.json"
    losses=read(loss_path) if loss_path.exists() else {}
    for seed in seeds:
        gates_at_start = {}
        for model_id, control in (("C", "A"), ("D", "B")):
            if model_id not in models:
                continue
            run = out / "runs" / f"{model_id}{seed}"
            net, transfer = initial(model_id, seed, weights)
            stock, stock_transfer = initial(control, seed, weights)
            assert transfer["paired_initial_sha"] == stock_transfer["paired_initial_sha"]
            control_record = out / "runs" / f"{control}{seed}" / "transfer.json"
            if control_record.exists():
                assert transfer["paired_initial_sha"] == read(control_record)["paired_initial_sha"]
            assert transfer["replacements"] == stock_transfer["replacements"]
            paired_state = net.state_dict()
            for key, tensor in stock.state_dict().items():
                torch.testing.assert_close(paired_state[key], tensor, rtol=0, atol=0)
            for index in (12, 15):
                for key, tensor in net.model[index].state_dict().items():
                    name = f"{index}.{key}"
                    if name in gates_at_start:
                        torch.testing.assert_close(tensor, gates_at_start[name], rtol=0, atol=0)
                    else:
                        gates_at_start[name] = tensor.clone()
            if learned:
                net, _ = restore(run / "best.pt")
                gradient = read(run / "gradient_check.json")
                if seed==seeds[0]:
                    net.args=get_cfg(overrides={"epochs":200,"imgsz":480,"batch":16})
                    objective=net.init_criterion()
                    detail=dict(head_type=type(net.model[-1]).__name__,criterion=type(objective).__name__,
                        loss_gains={k:getattr(net.args,k) for k in ("box","cls","dfl")},
                        native_one2one_selection=net.model[-1].end2end,network_yaml=net.yaml,
                        fusion_sites=transfer["gates"],partial_convolutions=transfer["replacements"],
                        assignment_weight_schedule=dict(dense_initial=objective.o2m,dense_final=objective.final_o2m,
                            one_to_one="one minus dense",updates="after each epoch"))
                    for branch in ("one2many","one2one"):
                        component=getattr(objective,branch)
                        detail[branch]=dict(reg_max=component.reg_max,assigner=type(component.assigner).__name__,
                            **{k:getattr(component.assigner,k) for k in ("topk","topk2","alpha","beta")})
                    losses[model_id]=detail
            output, values = observe(net, images)
            receipt = dict(control=control, paired_initial_sha=transfer["paired_initial_sha"],
                           sites=transfer["gates"], branch_weights={})
            if learned:
                receipt["checkpoint_sha"] = sha(run / "best.pt")
            else:
                assert set(output[1]) == {"one2one", "one2many"}
                receipt["exactly_equal_output_tensors"] = check_outputs(stock.eval()(images), output)
            for index, pair in values.items():
                np.testing.assert_allclose(pair.sum(axis=1), 2, rtol=0, atol=3e-7)
                if learned:
                    prefix = f"model.{index}."
                    for key in ("reduce.weight", "reduce.bias", "project.weight", "project.bias"):
                        assert gradient["max_weight_changes"][prefix + key] > 0
                        assert np.isfinite(gradient["grad_norm"][prefix + key]) and gradient["grad_norm"][prefix + key] > 0
                    assert np.ptp(pair[:, 0]) > 0 and np.any(pair != 1)
                else:
                    np.testing.assert_array_equal(pair, np.ones_like(pair))
                receipt["branch_weights"][index] = dict(min=pair.min(axis=0).tolist(),
                    max=pair.max(axis=0).tolist(), mean=pair.mean(axis=0).tolist(),
                    per_image=pair.tolist())
            results[f"{model_id}{seed}"] = receipt
    dest = out / "audit" / ("fusion-learned.json" if learned else "fusion-initial.json")
    put(dest, dict(split=split, images=[r["id"] for r in chosen], results=results))
    if learned: put(loss_path,losses)
    print(f"Verified {len(results)} paired models across both gates on {len(chosen)} {split} images: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--learned", action="store_true")
    args = parser.parse_args()
    check(args.data, args.artifacts, args.weights, args.learned)
