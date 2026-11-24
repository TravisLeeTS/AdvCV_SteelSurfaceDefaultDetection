import argparse
import os
from pathlib import Path
from .common import MODELS


def main():
    parser=argparse.ArgumentParser(description="MLR 512 steel detector experiments")
    parser.add_argument("stage",choices=["prepare","environment","train","export","benchmark","freeze","report","verify","smoke","demo","all"])
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts"))
    parser.add_argument("--weights",type=Path,default=Path("weights"))
    parser.add_argument("--groups",type=Path,help="Optional acquisition mapping CSV with id,group columns")
    parser.add_argument("--models",nargs="+",default=MODELS,choices=MODELS)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--phase",choices=["pilot","runs"],default="runs")
    parser.add_argument("--epochs",type=int,default=200)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--split",choices=["val","test"],default="val")
    parser.add_argument("--cells",nargs="+",help="Restrict benchmark recovery to named cells, e.g. A0-openvino-one-480")
    parser.add_argument("--output",type=Path,help="Report directory or headless demo output image")
    parser.add_argument("--image-id",help="Image ID for the demo")
    parser.add_argument("--side",type=int,choices=[256,320,480,640])
    parser.add_argument("--head",choices=["one","dense"],default="one")
    parser.add_argument("--tables-only",action="store_true")
    args=parser.parse_args()
    if args.stage=="all" and args.phase!="runs":
        parser.error("The all command runs the full research study. Use the train, benchmark and report commands for a pilot.")
    os.environ.setdefault("YOLO_CONFIG_DIR",str(Path(".ultralytics").resolve()))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    stage=args.stage
    data,out=args.data,args.artifacts
    if stage in ("prepare","all"):
        from .data import prepare_data
        from .augmentation_audit import audit_augmentation
        prepare_data(data,out,args.groups)
        audit_augmentation(data,out)
    if stage in ("environment","all"):
        from .hardware import inventory
        print(inventory(out))
    if stage in ("train","all"):
        from .train import fit
        for seed in args.seeds:
            for model in args.models:
                fit(data,out,args.weights,model,seed,args.epochs,args.batch,args.device,args.phase)
    if stage=="export":
        from .export import export_cell
        from .hardware import pin_cpu
        import torch
        pin_cpu(args.threads)
        with torch.inference_mode():
            for seed in args.seeds:
                for model in args.models:
                    for head,side in [("one",256),("one",320),("one",480),("one",640),("dense",480)]:
                        export_cell(data,out/args.phase/f"{model}{seed}",head,side,args.threads)
    if stage in ("benchmark","all"):
        from .bench import benchmark
        if args.phase=="runs" and (stage=="all" or args.split=="val"):
            from .hardware import pin_cpu
            from .smoke import reopen_exports
            gated_models=[m for m in args.models if m in ("C","D")]
            if gated_models:
                from scripts.check_fusion import check
                from .common import read,sha
                expected={f"{m}{k}" for m in gated_models for k in args.seeds}
                for learned,label in ((False,"initial"),(True,"learned")):
                    path=out/"audit"/f"fusion-{label}.json"
                    saved=read(path)["results"] if path.exists() else {}
                    if not expected.issubset(saved) or (learned and any(
                            saved[r]["checkpoint_sha"]!=sha(out/"runs"/r/"best.pt") for r in expected)):
                        check(data,out,args.weights,learned,gated_models,args.seeds)
            receipt=out/"study/reopened-before-timing.json"
            if not receipt.exists():
                pin_cpu(args.threads)
                reference=out/"runs"/f"{args.models[0]}{args.seeds[0]}"
                reopen_exports(data,out,reference,args.threads,receipt=receipt)
            for gated in ("C","D"):
                if gated in args.models and 0 in args.seeds:
                    receipt=out/"study"/f"reopened-before-timing-{gated}0.json"
                    if not receipt.exists():
                        pin_cpu(args.threads)
                        reopen_exports(data,out,out/"runs"/f"{gated}0",args.threads,receipt=receipt)
        timing={} if args.phase=="runs" else dict(warm=12,count=36,sessions=1)
        benchmark(data,out,args.split if stage=="benchmark" else "val",args.phase,args.models,args.seeds,
                  args.threads,only=args.cells,**timing)
        if args.phase=="runs":
            from .stages import stage_traces
            stage_traces(data,out,args.split if stage=="benchmark" else "val",args.phase,args.models,args.seeds,args.threads,args.cells)
    if stage in ("freeze","all"):
        from .report import freeze
        print(freeze(data,out,args.models,args.seeds)["selection"])
    if stage=="all":
        from .bench import benchmark
        benchmark(data,out,"test","runs",args.models,args.seeds,args.threads)
        stage_traces(data,out,"test","runs",args.models,args.seeds,args.threads)
    if stage in ("report","all"):
        from .report import tables
        study=out/("study" if args.phase=="runs" else "pilot-study")
        dest=args.output or study/"report"
        splits=("val","test") if args.phase=="runs" else ("val",)
        rows=tables(study,dest,args.models,args.seeds,splits)
        if not args.tables_only and args.phase=="runs":
            from .figures import research_figures
            from .materials import package
            from .uncertainty import intervals
            intervals(data,study,rows,dest)
            research_figures(data,out,rows,dest)
            package(data,out,dest)
    if stage in ("verify","all"):
        from .checks import verify_results
        verify_results(data,out,args.output or out/"study/report",args.models,args.seeds)
    if stage=="smoke":
        from .smoke import cpu_smoke
        cpu_smoke(data,out,args.weights,args.threads)
    if stage=="demo":
        from .demo import demo
        demo(args)


if __name__=="__main__":
    main()
