from pathlib import Path
import numpy as np
import cv2

from .common import CLASSES, digest, put, table
from .data import as_arrays, gray_image, load_rows
from .geometry import augment


def audit_augmentation(data,artifacts):
    rows=load_rows(data,"train")
    images=[gray_image(data,r) for r in rows]
    out=Path(artifacts)/"audit"
    cv2.setNumThreads(1)
    observations,summary=[],[]
    for seed in (0,1,2):
        order=np.random.default_rng(np.random.SeedSequence([seed,0])).permutation(len(rows))
        records=[]
        rejected=np.zeros(6,dtype=int)
        for position,i in enumerate(order):
            box,cls=as_arrays(rows[i])
            gray,target,labels,dropped,params=augment(images[i],box,cls,seed,0,int(i))
            if (gray.shape!=(480,480) or not 0.8<=params["scale"]<=1.2 or
                    max(abs(params["dx"]),abs(params["dy"]))>48 or not np.all((target>=0)&(target<=480)) or
                    len(labels)+sum(dropped)!=len(cls)):
                raise ValueError("Training augmentation check failed")
            rec=dict(seed=seed,epoch=1,position=position,image_id=rows[i]["id"],**params,
                     boxes_in=len(cls),boxes_retained=len(labels),**{f"rejected_{c}":int(dropped[k]) for k,c in enumerate(CLASSES)})
            records.append(rec)
            rejected+=dropped
        scales=[r["scale"] for r in records]
        summary.append(dict(seed=seed,images=len(records),order_and_transform_sha=digest(records),
                            scale_min=min(scales),scale_mean=float(np.mean(scales)),scale_max=max(scales),
                            flip_fraction=float(np.mean([r["flip"] for r in records])),rejected_by_class=rejected.tolist()))
        observations.extend(records)
    table(out/"augmentation_epoch1.csv",observations)
    put(out/"augmentation.json",dict(ordered_classes=CLASSES,summary=summary))
