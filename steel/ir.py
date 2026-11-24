"""Resolve and verify immutable, fixed-shape OpenVINO export bundles."""
from collections import Counter
from pathlib import Path
import re

import openvino as ov

from .common import CLASSES, MAX_DET, SIDES, digest, read, sha


PRECISION="FP32"
LEVEL_NAME=re.compile(r"^__module\.head\.head\.((?:one2one_)?cv[23]\.[012])\.")


def bundle_directory(run, head, side, checkpoint_sha=None):
    run=Path(run)
    if head not in ("one","dense") or side not in SIDES:
        raise ValueError(f"Unsupported IR head or input side: {head}, {side}")
    checkpoint_sha=checkpoint_sha or sha(run/"best.pt")
    return run/"exports"/checkpoint_sha/PRECISION/f"{head}-{side}"


def inspect_ir(xml, head, side):
    """Read the serialized XML and weights and inspect operations on the actual output path."""
    xml=Path(xml)
    graph=ov.Core().read_model(str(xml),str(xml.with_suffix(".bin")))
    if len(graph.inputs)!=1 or len(graph.outputs)!=1:
        raise ValueError(f"Unexpected IR interface in {xml}")
    input_port,output_port=graph.input(),graph.output()
    if not input_port.partial_shape.is_static or not output_port.partial_shape.is_static:
        raise ValueError(f"Expected a static export in {xml}")
    input_shape=list(input_port.shape)
    if input_shape!=[1,3,side,side]:
        raise ValueError(f"Actual IR input shape {input_shape} does not match side {side}: {xml}")
    if input_port.element_type.get_type_name()!="f32" or output_port.element_type.get_type_name()!="f32":
        raise ValueError(f"Expected FP32 inputs and outputs in {xml}")
    ops=graph.get_ordered_ops()
    levels=set()
    for node in ops:
        if node.get_type_name() in ("Convolution","GroupConvolution"):
            match=LEVEL_NAME.match(node.get_friendly_name())
            if match: levels.add(match.group(1))
    prefix="one2one_" if head=="one" else ""
    expected={f"{prefix}cv{part}.{level}" for part in (2,3) for level in range(3)}
    if levels!=expected:
        raise ValueError(f"Wrong learned head branches in {xml}. Observed {sorted(levels)}, expected {sorted(expected)}")
    types=Counter(node.get_type_name() for node in ops)
    suppressions=sum(n for op,n in types.items() if "nms" in op.lower() or "nonmaxsuppression" in op.lower())
    if suppressions or types["TopK"]!=(2 if head=="one" else 0):
        raise ValueError(f"Unexpected selection or suppression operations in {xml}")
    output_shape=list(output_port.shape)
    anchors=sum((side//stride)**2 for stride in (8,16,32))
    expected_output=[1,MAX_DET,6] if head=="one" else [1,4+len(CLASSES),anchors]
    if output_shape!=expected_output:
        raise ValueError(f"Unexpected IR output {output_shape} in {xml}")
    constants=Counter(node.get_output_element_type(0).get_type_name() for node in ops if node.get_type_name()=="Constant")
    if {"f16","bf16"}.intersection(constants):
        raise ValueError(f"Low-precision weights in FP32 bundle {xml}")
    result=dict(input_shape=input_shape,output_shape=output_shape,input_type="f32",output_type="f32",
                head=head,learned_branches=sorted(levels),topk_operations=types["TopK"],
                nms_operations=suppressions,constant_types=dict(constants))
    gates={}
    for index in (12,15):
        prefix=f"__module.neck.layers.{index}"
        local=[node for node in ops if node.get_friendly_name().startswith((prefix+"/",prefix+"."))]
        if not any(node.get_friendly_name().startswith(prefix+"/aten::softmax/") for node in local):
            continue
        counts=Counter(node.get_type_name() for node in local)
        if counts["ReduceMean"]!=2 or counts["MatMul"]!=2 or counts["Softmax"]!=1 or counts["Multiply"]<2:
            raise ValueError(f"Incomplete learned fusion gate at layer {index} in {xml}")
        gates[str(index)]={op:counts[op] for op in ("ReduceMean","MatMul","Softmax","Multiply")}
    if gates: result["fusion_gates"]=gates
    return result


def bundle_id(checkpoint_sha,head,side,ir):
    return digest(dict(checkpoint_sha=checkpoint_sha,head=head,side=side,precision=PRECISION,
                       xml_sha=ir["xml_sha"],bin_sha=ir["bin_sha"]))


def checked_bundle(run,head,side,checkpoint_sha=None):
    run=Path(run)
    checkpoint_sha=checkpoint_sha or sha(run/"best.pt")
    folder=bundle_directory(run,head,side,checkpoint_sha)
    record=read(folder/"manifest.json")
    if (record["checkpoint_sha"],record["head"],record["side"],record["precision"])!=(checkpoint_sha,head,side,PRECISION):
        raise ValueError(f"IR manifest identity mismatch in {folder}")
    xml=folder/"model.xml"
    ir=record["ir"]
    if (sha(xml),sha(xml.with_suffix(".bin")))!=(ir["xml_sha"],ir["bin_sha"]):
        raise ValueError(f"IR XML or weights changed in {folder}")
    if record["bundle_sha"]!=bundle_id(checkpoint_sha,head,side,ir) or record["reopened_ir"]!=inspect_ir(xml,head,side):
        raise ValueError(f"Reopened IR disagrees with its export manifest in {folder}")
    expected={"12","15"} if record["trace"]["model"] in ("C","D") else set()
    if set(record["reopened_ir"].get("fusion_gates",{}))!=expected:
        raise ValueError(f"Reopened IR has unexpected fusion sites in {folder}")
    return folder,record
