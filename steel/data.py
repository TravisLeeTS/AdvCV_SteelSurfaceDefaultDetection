from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import csv
from itertools import product

import cv2
import numpy as np

from .common import CLASSES, SPLIT_SEED, digest, fetch, put, read, sha, table
from .geometry import augment, from_yolo, letterbox, project, to_yolo

ARCHIVE_URL = "https://drive.usercontent.google.com/download?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k&export=download&confirm=t"
ARCHIVE_SHA = "cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5"
SPLITS = ("train", "val", "test")


def canonical(name):
    return CLASSES.index(name.strip().lower().replace("-", "_").replace(" ", "_"))


def as_arrays(row):
    return np.asarray(row["boxes"], dtype=np.float32).reshape(-1, 4), np.asarray(row["classes"], dtype=np.int64)


def load_rows(data, split):
    return read(Path(data) / f"{split}.json")


def gray_image(data, row):
    path = Path(data) / row["image"]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (row["height"], row["width"]):
        raise ValueError(f"Missing or changed image {path}")
    return image


def indexed(archive, extension):
    found = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix.lower() == extension:
            if path.stem in found:
                raise ValueError(f"Repeated filename in archive: {path.stem}")
            found[path.stem] = name
    return found


def ingest(data, archive):
    images, xmls = indexed(archive, ".jpg"), indexed(archive, ".xml")
    if images.keys() != xmls.keys() or len(images) != 1800:
        raise ValueError(f"Unpaired or unexpected archive: {len(images)} images, {len(xmls)} XMLs")
    (data / "images").mkdir(parents=True, exist_ok=True)
    rows, hashes = [], []
    for name in sorted(images):
        image_bytes = archive.read(images[name])
        pixels = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        xml = archive.read(xmls[name])
        ann = ET.fromstring(xml)
        width, height = (int(ann.findtext("size/" + key)) for key in ("width", "height"))
        if pixels is not None and pixels.ndim == 3 and (pixels == pixels[:, :, :1]).all():
            pixels = pixels[:, :, 0]
        if pixels is None or pixels.shape != (height, width) or (height, width) != (200, 200):
            raise ValueError(f"Corrupt, non-grayscale or unexpected dimensions: {name}")
        if PurePosixPath(ann.findtext("filename")).stem != name:
            raise ValueError(f"Image/XML name mismatch for {name}")
        boxes, classes = [], []
        for obj in ann.findall("object"):
            x1, y1, x2, y2 = [int(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            if not (1 <= x1 <= x2 <= width and 1 <= y1 <= y2 <= height):
                raise ValueError(f"Out of bounds VOC box in {name}")
            boxes.append([x1 - 1, y1 - 1, x2, y2])
            classes.append(canonical(obj.findtext("name")))
        if not boxes:
            raise ValueError(f"No annotated object in {name}")
        path = data / "images" / (name + ".jpg")
        file_hash = sha256(image_bytes).hexdigest()
        if not path.exists() or sha(path) != file_hash:
            path.write_bytes(image_bytes)
        resized = cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_AREA)
        low = cv2.dct(resized.astype(np.float32))[:8, :8].ravel()[1:]
        hashes.append(int.from_bytes(np.packbits(low > np.median(low)).tobytes(), "big"))
        rows.append(dict(id=name, image=path.relative_to(data).as_posix(), width=width, height=height,
                         image_sha=file_hash, xml_sha=sha256(xml).hexdigest(),
                         pixel_sha=sha256(pixels.tobytes()).hexdigest(), stratum=canonical(name.rsplit("_", 1)[0]),
                         boxes=boxes, classes=classes))
    return rows, hashes


def shifted_pairs(data, rows):
    from threadpoolctl import threadpool_limits
    pixels = np.stack([gray_image(data, r) for r in rows])
    thumbs = np.stack([cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA) for a in pixels])

    def normalized(crops):
        x = crops.reshape(len(rows), -1).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

    center = normalized(thumbs[:, 8:24, 8:24])
    candidates = set()
    with threadpool_limits(limits=4):
        for dx, dy in product(range(0, 17, 2), repeat=2):
            ii, jj = np.where(center @ normalized(thumbs[:, dy:dy+16, dx:dx+16]).T >= 0.88)
            candidates.update((min(i, j), max(i, j)) for i, j in zip(ii, jj) if i != j)
    accepted = {}
    for i, j in sorted(candidates):
        a, b = pixels[i], pixels[j]
        scores = cv2.matchTemplate(b, a[50:150, 50:150], cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < 0.95:
            continue
        dx, dy = x-50, y-50
        ax, ay, bx, by = max(-dx, 0), max(-dy, 0), max(dx, 0), max(dy, 0)
        w, h = 200-abs(dx), 200-abs(dy)
        overlap = float(np.corrcoef(a[ay:ay+h, ax:ax+w].ravel(), b[by:by+h, bx:bx+w].ravel())[0, 1])
        if overlap >= 0.92:
            accepted[i, j] = (round(score, 6), round(overlap, 6), dx, dy)
    return accepted


def groups_for(data, rows, hashes, group_file):
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    def union(i, j):
        a, b = sorted((root(i), root(j)))
        parents[b] = a

    if group_file:
        lookup = {r["id"]: i for i, r in enumerate(rows)}
        first = {}
        with Path(group_file).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                i = lookup[row["id"]]
                union(i, first.setdefault(row["group"], i))
    shifted = shifted_pairs(data, rows)
    candidates = []
    for i, h in enumerate(hashes):
        for j in range(i):
            distance = (h ^ hashes[j]).bit_count()
            exact = rows[i]["pixel_sha"] == rows[j]["pixel_sha"]
            if exact or distance <= 8 or (j, i) in shifted:
                union(i, j)
                central, whole, dx, dy = shifted.get((j, i), (None, None, None, None))
                candidates.append(dict(first=rows[j]["id"], second=rows[i]["id"], hamming=distance,
                                       exact_pixels=exact, center_ncc=central, overlap_ncc=whole,
                                       dx=dx, dy=dy, decision="group together before splitting"))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        group = rows[root(i)]["id"]
        row["group"] = group
        groups[group].append(i)
    return groups, candidates


def assign(rows, groups):
    rng = np.random.default_rng(SPLIT_SEED)
    # Image strata have hard capacities where a whole group can fit. Object counts
    # provide a secondary stratification target, including cross-class annotations.
    vectors = {}
    for group, members in groups.items():
        image_counts = np.bincount([rows[i]["stratum"] for i in members], minlength=6)
        object_counts = np.bincount([c for i in members for c in rows[i]["classes"]], minlength=6)
        vectors[group] = np.r_[image_counts, object_counts]
    target = np.outer([0.6, 0.2, 0.2], sum(vectors.values()))
    counts = np.zeros_like(target)
    order = sorted(rng.permutation(list(groups)), key=lambda g: -len(groups[g]))
    for group in order:
        v = vectors[group]
        feasible = ((counts[:, :6] + v[:6]) <= target[:, :6] + 1e-8).all(axis=1)
        loss = ((2 * (counts - target) * v + v * v) / target) @ np.r_[np.ones(6), np.full(6, 0.15)]
        if feasible.any():
            loss[~feasible] = np.inf
        s = int(np.argmin(loss + rng.random(3) * 1e-9))
        counts[s] += v
        for i in groups[group]:
            rows[i]["split"] = SPLITS[s]


def draw_boxes(image, boxes):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2)-1, round(y2)-1), (85, 230, 255), 1, cv2.LINE_AA)
    return canvas


def caption(image, label):
    image = cv2.resize(image, (250, 250))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(38, 38, 38))
    cv2.putText(image, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def audit_views(data, out, rows, candidates):
    lookup = {r["id"]: r for r in rows}
    # Display the hash pairs plus a sample of registered overlaps from each defect.
    samples = [p for p in candidates if p["hamming"] <= 8]
    for c in range(6):
        shifted = [p for p in candidates if p["hamming"] > 8 and lookup[p["first"]]["stratum"] == c]
        if shifted:
            samples.append(min(shifted, key=lambda p: p["overlap_ncc"]))
    near = [np.hstack([caption(gray_image(data, lookup[p[k]]), p[k]) for k in ("first", "second")]) for p in samples]
    if near:
        cv2.imwrite(str(out / "near_pairs.jpg"), np.vstack(near))
    train = [r for r in rows if r["split"] == "train"]
    examples = {}
    max_error = 0.0
    for i, r in enumerate(train):
        b, c = as_arrays(r)
        image = gray_image(data, r)
        _, t = letterbox(image, 480)
        back = project(project(from_yolo(to_yolo(b, 200, 200), 200, 200), t), t, inverse=True)
        max_error = max(max_error, float(np.abs(back-b).max()))
        widths = b[:, 2:] - b[:, :2]
        overlap = np.minimum(b[:, None, 2:], b[None, :, 2:]) - np.maximum(b[:, None, :2], b[None, :, :2])
        tags = {"thin": (widths.max(1) / widths.min(1) >= 6).any(),
                "border": (b[:, :2] == 0).any() or (b[:, 2:] == 200).any(),
                "touching": np.triu((overlap >= 0).all(2), k=1).any()}
        for tag, match in tags.items():
            if match and tag not in examples and not any(tag2 in examples and ok for tag2, ok in tags.items()):
                aug, target, _, _, params = augment(image, b, c, 0, 0, i)
                examples[tag] = np.hstack((caption(draw_boxes(image, b), f"{tag}  {r['id']}"),
                                           caption(draw_boxes(aug, target), f"augmented  {params['scale']:.3f}x")))
    if max_error > 1e-3 or len(examples) != 3:
        raise ValueError("Coordinate round-trip or overlay coverage check failed")
    cv2.imwrite(str(out / "training_geometry.jpg"), np.vstack(list(examples.values())))
    return max_error


def prepare_data(data, artifacts, group_file=None):
    data, out = Path(data), Path(artifacts) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    path = fetch(data / "downloads/NEU-DET.zip", ARCHIVE_URL, ARCHIVE_SHA)
    with ZipFile(path) as archive:
        rows, hashes = ingest(data, archive)
    groups, candidates = groups_for(data, rows, hashes, group_file)
    assign(rows, groups)
    counts, manifest, split_files = [], [], {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        put(data / f"{split}.json", part)
        split_files[split] = sha(data / f"{split}.json")
        objects = Counter(c for row in part for c in row["classes"])
        images = Counter(c for row in part for c in set(row["classes"]))
        for c, name in enumerate(CLASSES):
            counts.append(dict(split=split, defect=name, filename_stratum_images=sum(r["stratum"] == c for r in part),
                               images_with_class=images[c], objects=objects[c]))
        manifest.extend({k: v for k, v in r.items() if k not in ("boxes", "classes")} for r in part)
    table(out / "manifest.csv", manifest)
    table(out / "class_counts.csv", counts)
    table(out / "near_pairs.csv", candidates)
    error = audit_views(data, out, rows, candidates)
    meta = dict(archive_sha=ARCHIVE_SHA, archive_url=ARCHIVE_URL, publisher="https://faculty.neu.edu.cn/songkc/en/zdylm/263265/",
                images=len(rows), boxes=sum(len(r["boxes"]) for r in rows), groups=len(groups),
                exact_duplicate_extras=len(rows)-len(set(r["pixel_sha"] for r in rows)),
                near_pairs=len(candidates), grouping="Union exact pixels, 63-bit DCT hash Hamming <=8, and translated matches (thumbnail NCC >=0.88; 100x100 center NCC >=0.95; full overlapping region NCC >=0.92).",
                acquisition_groups=sha(group_file) if group_file else "Acquisition identifiers absent from publisher XML.",
                channels="Publisher JPEGs decode to three equal channels. Training and inference repeat one grayscale plane.",
                coordinates="VOC 1-based inclusive endpoints to zero-based half-open [xmin-1,ymin-1,xmax,ymax].",
                max_roundtrip_error_px=error, split_seed=SPLIT_SEED, split_files=split_files, split_hash=digest(split_files),
                counts={s: sum(r["split"] == s for r in rows) for s in SPLITS})
    put(data / "split.json", meta)
    put(out / "audit.json", meta)
    print(meta, flush=True)
