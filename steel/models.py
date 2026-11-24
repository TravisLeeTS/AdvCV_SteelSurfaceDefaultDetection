import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.modules import Bottleneck, Concat
from ultralytics.nn.tasks import DetectionModel, torch_safe_load

from .common import CLASSES, WEIGHTS, fetch


class PartialConv(nn.Module):
    def __init__(self, dense):
        super().__init__()
        self.width = dense.in_channels // 4
        self.kernel = nn.Conv2d(self.width, self.width, 3, padding=1, bias=False)
        with torch.no_grad():
            self.kernel.weight.copy_(dense.weight[:self.width, :self.width])

    def forward(self, x):
        return torch.cat((self.kernel(x[:, :self.width]), x[:, self.width:]), dim=1)


class FusionConcat(nn.Module):
    """Gate the two input branches per image, preserving the stock concatenation."""

    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Linear(channels, max(1, channels // 16))
        self.project = nn.Linear(self.reduce.out_features, 2)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def branch_weights(self, branches):
        descriptors = torch.cat([x.mean(dim=(2, 3)) for x in branches], dim=1)
        return self.project(torch.relu(self.reduce(descriptors))).softmax(dim=1) * 2

    def forward(self, branches):
        weights = self.branch_weights(branches).to(branches[0].dtype)
        weighted = [x * weights[:, i:i+1, None, None] for i, x in enumerate(branches)]
        return torch.cat(weighted, dim=1)


def add_fusion(net):
    gates = []
    for index, lateral in ((12, 6), (15, 4)):
        old = net.model[index]
        if type(old) is not Concat or old.d != 1 or old.f != [-1, lateral]:
            raise ValueError(f"Unexpected top-down concatenation at layer {index}")
        channels = net.model[index + 1].cv1.conv.in_channels
        gate = FusionConcat(channels)
        gate.i, gate.f, gate.type = old.i, old.f, "FusionConcat"
        gate.np = sum(p.numel() for p in gate.parameters())
        net.model[index] = gate
        gates.append(dict(path=f"model.{index}", inputs=old.f, descriptor_channels=channels,
                          hidden_channels=gate.reduce.out_features, parameters=gate.np))
    return gates


def replace_pconv(net):
    records = []
    for stage in (6, 8):
        blocks = [(n, m) for n, m in net.model[stage].named_modules() if type(m) is Bottleneck]
        sites = []
        for name, block in blocks:
            conv = block.cv2.conv
            if (block.cv1.conv.kernel_size == (3, 3) and conv.kernel_size == (3, 3) and
                    conv.stride == (1, 1) and conv.groups == 1 and conv.in_channels == conv.out_channels):
                if conv.bias is not None or conv.dilation != (1, 1) or conv.padding != (1, 1):
                    raise ValueError("Unexpected geometry at PConv site")
                path = f"model.{stage}.{name}.cv2.conv"
                block.cv2.conv = PartialConv(conv)
                sites.append(dict(path=path, dense_shape=list(conv.weight.shape),
                                  new_shape=list(block.cv2.conv.kernel.weight.shape),
                                  source_slice=f"[:{conv.in_channels//4}, :{conv.in_channels//4}, :, :]"))
        if not sites:
            raise ValueError(f"No eligible bottleneck at stage {stage}")
        records.extend(sites)
    return records


def build(model, cfg):
    net = DetectionModel(copy.deepcopy(cfg), nc=6, verbose=False)
    replacements = replace_pconv(net) if model in ("B", "D") else []
    if model in ("C", "D"):
        add_fusion(net)
    net.names = dict(enumerate(CLASSES))
    net.end2end = True
    head = net.model[-1]
    if not head.end2end or any(v is None for v in (*head.one2one.values(), *head.one2many.values())):
        raise ValueError("The original dual-head structure is required")
    return net, replacements


def initial(model, seed, weights):
    name, url, file_sha = WEIGHTS["E" if model == "E" else "A"]
    path = fetch(Path(weights)/name, url, file_sha)
    # Publisher files are verified before deserializing their module objects.
    checkpoint, _ = torch_safe_load(path)
    source = (checkpoint.get("ema") or checkpoint["model"]).float()
    torch.manual_seed(seed)
    net, _ = build("E" if model == "E" else "A", source.yaml)
    target = net.state_dict()
    compatible = {k: v for k, v in source.state_dict().items() if k in target and target[k].shape == v.shape}
    missing = net.load_state_dict(compatible, strict=False).missing_keys
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(v.cpu().contiguous().numpy().tobytes())
    records = replace_pconv(net) if model in ("B", "D") else []
    transfer = dict(source=url, source_sha=file_sha, paired_initial_sha=h.hexdigest(),
                     copied_tensors=len(compatible), freshly_initialized=missing, replacements=records)
    if model in ("C", "D"):
        # Pair both gate MLPs across the two backbones independently of PConv allocation.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(512000 + seed)
            transfer["gates"] = add_fusion(net)
        transfer["gate_seed"] = 512000 + seed
    return net, transfer


def restore(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net, _ = build(state["model_id"], state["yaml"])
    net.load_state_dict(state["weights"], strict=True)
    return net.float().eval(), state


class Neck(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.layers = net.model[:-1]
        self.keep = set(net.save)
        self.indices = net.model[-1].f

    def forward(self, x):
        saved = []
        for m in self.layers:
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if m.i in self.keep else None)
        return tuple(saved[i] for i in self.indices)


class SelectedHead(nn.Module):
    def __init__(self, head, branch):
        super().__init__()
        self.head = head
        self.one = branch == "one"
        head.end2end = self.one
        head.export = True
        head.format = "openvino"
        head.xyxy = False

    def forward(self, p3, p4, p5):
        h = self.head
        selected = h.one2one if self.one else h.one2many
        raw = h.forward_head([p3, p4, p5], **selected)
        decoded = h._inference(raw)
        return h.postprocess(decoded.permute(0, 2, 1)) if self.one else decoded


class SingleBranch(nn.Module):
    def __init__(self, net, branch):
        super().__init__()
        self.neck = Neck(net)
        self.head = SelectedHead(net.model[-1], branch)

    def forward(self, x):
        return self.head(*self.neck(x))
