#!/usr/bin/env python3
"""Instance-segment crowded olives with SAM 2.1 and conservative post-processing.

The output masks describe the *visible* part of every olive.  Specular pixels are
kept inside the olive masks; they are not treated as a separate object or as a
hole.  A single still image cannot recover the invisible part of an olive that
is hidden by another olive.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM2Predictor


PIPELINE_DIR = Path(__file__).resolve().parent


@dataclass
class InstanceInfo:
    instance_id: int
    score: float
    area_px: int
    bbox_xyxy: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]


class DenseSAM2Predictor(SAM2Predictor):
    """SAM 2 predictor with denser sampling and less aggressive box NMS.

    Ultralytics' standard configuration is intended for general scenes.  A
    crowded olive image needs many sampling points and a high box-NMS IoU:
    adjacent olives have strongly overlapping bounding boxes even though their
    masks are distinct.
    """

    def __init__(
        self,
        *args,
        points_stride: int = 32,
        stability_threshold: float = 0.80,
        min_region_area: int = 64,
        **kwargs,
    ) -> None:
        self.points_stride = points_stride
        self.stability_threshold = stability_threshold
        self.min_region_area = min_region_area
        super().__init__(*args, **kwargs)

    def inference(
        self,
        im,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        *args,
        **kwargs,
    ):
        if all(prompt is None for prompt in (bboxes, points, masks)):
            return self.generate(
                im,
                points_stride=self.points_stride,
                conf_thres=self.args.conf,
                stability_score_thresh=self.stability_threshold,
                min_mask_region_area=self.min_region_area,
            )
        return super().inference(
            im,
            bboxes,
            points,
            labels,
            masks,
            multimask_output,
            *args,
            **kwargs,
        )


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def fill_mask(mask: np.ndarray) -> np.ndarray:
    """Close tiny boundary cracks and fill internal holes, including glare holes."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    # Morphology on a tight crop is much faster than repeating full-resolution
    # connected-component and flood-fill operations for every proposal.
    pad = 6
    x1, x2 = max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)
    y1, y2 = max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)
    crop = keep_largest_component(mask[y1:y2, x1:x2])
    area = int(crop.sum())
    if area == 0:
        return np.zeros_like(mask, dtype=bool)

    # Scale the closing kernel to the object, not to the image resolution.
    radius = int(np.clip(round(np.sqrt(area) * 0.018), 1, 4))
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(
        crop.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    )

    # Fill holes from the exterior.  This changes only enclosed pixels; it does
    # not use a convex hull and therefore does not invent hidden olive parts.
    padded = cv2.copyMakeBorder(closed, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    exterior = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(exterior, flood_mask, (0, 0), 1)
    holes = exterior == 0
    filled = (padded > 0) | holes
    output = np.zeros_like(mask, dtype=bool)
    output[y1:y2, x1:x2] = filled[1:-1, 1:-1]
    return output


def mask_geometry(mask: np.ndarray) -> Tuple[int, Tuple[int, int, int, int], float, float]:
    ys, xs = np.nonzero(mask)
    area = len(xs)
    if area == 0:
        return 0, (0, 0, 0, 0), 0.0, 0.0

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    width, height = x2 - x1, y2 - y1
    aspect = max(width, height) / max(1, min(width, height))

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / max(1.0, hull_area)
    return area, (x1, y1, x2, y2), aspect, solidity


def plausible_olive(
    mask: np.ndarray,
    value_channel: np.ndarray,
    reference_image_area: int | None = None,
) -> bool:
    """Reject tiny texture masks, dark gaps, and implausibly broad regions."""
    height, width = mask.shape
    image_area = reference_image_area or height * width
    area, bbox, aspect, solidity = mask_geometry(mask)

    min_area = max(80, round(image_area * 0.00035))
    max_area = round(image_area * 0.06)
    if not (min_area <= area <= max_area):
        return False
    if aspect > 3.6 or solidity < 0.62:
        return False

    x1, y1, x2, y2 = bbox
    extent = area / max(1, (x2 - x1) * (y2 - y1))
    if extent < 0.34:
        return False

    values = value_channel[mask]
    # Dark inter-olive gaps occasionally form stable SAM regions.  Real green,
    # brown, and purple olives all remain comfortably above this conservative
    # brightness test in the supplied images.
    if float(np.median(values)) < 48 or float(np.mean(values > 55)) < 0.55:
        return False
    return True


def suppress_duplicate_masks(
    candidates: Sequence[Tuple[np.ndarray, float]],
    iou_threshold: float = 0.66,
    containment_threshold: float = 0.88,
) -> List[Tuple[np.ndarray, float]]:
    """Mask NMS that preserves adjacent olives with overlapping bounding boxes."""
    kept: List[Tuple[np.ndarray, float]] = []
    kept_geometry: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for mask, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        area, bbox, _, _ = mask_geometry(mask)
        x1, y1, x2, y2 = bbox
        duplicate = False
        for (other, _), (other_area, other_bbox) in zip(kept, kept_geometry):
            ox1, oy1, ox2, oy2 = other_bbox
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            intersection = int(
                np.logical_and(
                    mask[iy1:iy2, ix1:ix2], other[iy1:iy2, ix1:ix2]
                ).sum()
            )
            if intersection == 0:
                continue
            union = area + other_area - intersection
            iou = intersection / max(1, union)
            containment = intersection / max(1, min(area, other_area))
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((mask, score))
            kept_geometry.append((area, bbox))
    return kept


def specular_mask(bgr: np.ndarray) -> np.ndarray:
    """Create a diagnostic glare mask; segmentation itself uses the original image."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    value_limit = max(175.0, float(np.percentile(value, 88)))
    saturation_limit = min(135.0, float(np.percentile(saturation, 58)))
    glare = ((value >= value_limit) & (saturation <= saturation_limit)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, kernel)
    return glare * 255


def palette_color(index: int) -> Tuple[int, int, int]:
    # Golden-ratio hue stepping gives stable, well-separated neighboring colors.
    hue = (0.11 + index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def extract_candidates(
    result,
    bgr: np.ndarray,
    reference_image_area: int | None = None,
) -> List[Tuple[np.ndarray, float]]:
    """Convert one SAM result into refined, plausible local olive masks."""
    if result.masks is None or result.boxes is None:
        return []

    height, width = bgr.shape[:2]
    raw_masks = result.masks.data.cpu().numpy().astype(bool)
    scores = result.boxes.conf.cpu().numpy()
    value_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates: List[Tuple[np.ndarray, float]] = []

    for raw_mask, score in zip(raw_masks, scores):
        if raw_mask.shape != (height, width):
            raw_mask = cv2.resize(
                raw_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        refined = fill_mask(raw_mask)
        if plausible_olive(refined, value_channel, reference_image_area):
            candidates.append((refined, float(score)))
    return candidates


def tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    """Return evenly spaced tile origins with at least the requested overlap."""
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    tile_count = int(np.ceil((length - tile_size) / stride)) + 1
    return [
        int(round(value))
        for value in np.linspace(0, length - tile_size, tile_count)
    ]


def touches_internal_tile_edge(
    mask: np.ndarray,
    tile_xyxy: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> bool:
    """Detect a proposal cut by a tile edge that is not an image edge."""
    _, bbox, _, _ = mask_geometry(mask)
    x1, y1, x2, y2 = bbox
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_xyxy
    image_height, image_width = image_shape
    tile_height, tile_width = mask.shape
    margin = max(2, round(min(tile_height, tile_width) * 0.003))

    return (
        (tile_x1 > 0 and x1 <= margin)
        or (tile_y1 > 0 and y1 <= margin)
        or (tile_x2 < image_width and x2 >= tile_width - margin)
        or (tile_y2 < image_height and y2 >= tile_height - margin)
    )


def predict_tiled(
    bgr: np.ndarray,
    predictor: DenseSAM2Predictor,
    tile_size: int,
    overlap: int,
) -> Tuple[List[Tuple[np.ndarray, float]], int]:
    """Run SAM per overlapping tile and map accepted masks to global coordinates."""
    image_height, image_width = bgr.shape[:2]
    x_starts = tile_starts(image_width, tile_size, overlap)
    y_starts = tile_starts(image_height, tile_size, overlap)
    tile_boxes = [
        (
            x,
            y,
            min(x + tile_size, image_width),
            min(y + tile_size, image_height),
        )
        for y in y_starts
        for x in x_starts
    ]

    print(
        f"  tiled inference: {len(tile_boxes)} tiles, "
        f"tile_size={tile_size}px, overlap={overlap}px"
    )
    global_candidates: List[Tuple[np.ndarray, float]] = []
    for tile_index, (x1, y1, x2, y2) in enumerate(tile_boxes, start=1):
        tile = bgr[y1:y2, x1:x2]
        result = predictor(source=tile)[0]
        # Use tile-relative size thresholds. Besides reducing memory, tiling is
        # meant to reveal olives that are too small in the full image.
        local_candidates = extract_candidates(result, tile)

        accepted_in_tile = 0
        for local_mask, score in local_candidates:
            if touches_internal_tile_edge(
                local_mask,
                (x1, y1, x2, y2),
                (image_height, image_width),
            ):
                continue
            global_mask = np.zeros((image_height, image_width), dtype=bool)
            global_mask[y1:y2, x1:x2] = local_mask
            global_candidates.append((global_mask, score))
            accepted_in_tile += 1

        print(
            f"    tile {tile_index}/{len(tile_boxes)} "
            f"({x1}:{x2}, {y1}:{y2}): {accepted_in_tile} candidates"
        )

    return global_candidates, len(tile_boxes)


def build_outputs(
    image_path: Path,
    result,
    output_root: Path,
    draw_ids: bool,
    candidates: List[Tuple[np.ndarray, float]] | None = None,
    tiled: bool = False,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
) -> int:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # Include the extension because `olives.jpg` and `olives.png` otherwise
    # share the same stem and would overwrite one another.
    tiled_suffix = "_tiled" if tiled else ""
    image_key = (
        f"{image_path.stem}_{image_path.suffix.lower().lstrip('.')}{tiled_suffix}"
    )
    image_dir = output_root / image_key
    image_dir.mkdir(parents=True, exist_ok=True)
    height, width = bgr.shape[:2]

    if candidates is None:
        candidates = extract_candidates(result, bgr)
    if not candidates:
        raise RuntimeError(f"SAM produced no valid olive masks for {image_path}")

    instances = suppress_duplicate_masks(candidates)
    # Predictions are ordered by confidence. The first/highest-confidence mask
    # owns a pixel in the rare overlap between two visible predictions.
    label_map = np.zeros((height, width), dtype=np.uint16)
    for instance_id, (mask, _) in enumerate(instances, start=1):
        label_map[mask & (label_map == 0)] = instance_id

    overlay = bgr.copy()
    tint = bgr.copy()
    metadata: List[InstanceInfo] = []
    for instance_id, (mask, score) in enumerate(instances, start=1):
        visible = label_map == instance_id
        if not np.any(visible):
            continue
        color = palette_color(instance_id)
        tint[visible] = color
        contour_mask = visible.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

        ys, xs = np.nonzero(visible)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        cx, cy = float(xs.mean()), float(ys.mean())
        metadata.append(
            InstanceInfo(
                instance_id=instance_id,
                score=round(score, 6),
                area_px=int(len(xs)),
                bbox_xyxy=(x1, y1, x2, y2),
                centroid_xy=(round(cx, 2), round(cy, 2)),
            )
        )

        if draw_ids:
            distance = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
            _, _, _, text_point = cv2.minMaxLoc(distance)
            font_scale = max(0.35, min(height, width) / 1100.0)
            cv2.putText(
                overlay,
                str(instance_id),
                text_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                max(1, round(font_scale * 2)),
                cv2.LINE_AA,
            )

    union = label_map > 0
    overlay = cv2.addWeighted(overlay, 0.72, tint, 0.28, 0.0)
    cutout = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = union.astype(np.uint8) * 255

    cv2.imwrite(str(image_dir / "instance_labels.png"), label_map)
    cv2.imwrite(str(image_dir / "olive_mask.png"), union.astype(np.uint8) * 255)
    cv2.imwrite(str(image_dir / "olive_cutout.png"), cutout)
    cv2.imwrite(str(image_dir / "overlay.png"), overlay)
    cv2.imwrite(str(image_dir / "specular_diagnostic.png"), specular_mask(bgr))
    with (image_dir / "instances.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(image_path),
                "width": width,
                "height": height,
                "instance_count": len(metadata),
                "tiled_inference": tiled,
                "tile_size": tile_size if tiled else None,
                "tile_overlap": tile_overlap if tiled else None,
                "instances": [asdict(item) for item in metadata],
            },
            handle,
            indent=2,
        )
    return len(metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment visible olive instances, including specular pixels."
    )
    parser.add_argument("images", nargs="+", type=Path, help="Input JPG/PNG image(s)")
    parser.add_argument("--output", type=Path, default=PIPELINE_DIR / "results")
    parser.add_argument(
        "--model", default=str(PIPELINE_DIR / "sam2.1_t.pt"), help="SAM 2.1 weights"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA id")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--points-stride", type=int, default=32)
    parser.add_argument("--confidence", type=float, default=0.80)
    parser.add_argument("--stability", type=float, default=0.90)
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Enable overlapping tiled inference (disabled by default)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Square tile size in original-image pixels (default: 1024)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help="Overlap between neighboring tiles in pixels (default: 128)",
    )
    parser.add_argument("--no-ids", action="store_true", help="Do not draw instance IDs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input image(s): " + ", ".join(missing))
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be greater than zero")
    if not 0 <= args.tile_overlap < args.tile_size:
        raise ValueError("--tile-overlap must satisfy 0 <= overlap < tile size")

    device = choose_device(args.device)
    overrides = {
        "model": args.model,
        "task": "segment",
        "mode": "predict",
        "imgsz": args.imgsz,
        "conf": args.confidence,
        # Dense scenes contain adjacent objects whose *boxes* overlap heavily.
        "iou": 0.85,
        "device": device,
        "verbose": False,
        "save": False,
    }
    predictor = DenseSAM2Predictor(
        overrides=overrides,
        points_stride=args.points_stride,
        stability_threshold=args.stability,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    # The current SAM predictor accepts one source image per call, but the model
    # remains loaded in the predictor and is reused across calls.
    for path in args.images:
        image_start = time.perf_counter()
        if args.tile:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Could not read image: {path}")
            candidates, _ = predict_tiled(
                bgr,
                predictor,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
            )
            result = None
        else:
            result = predictor(source=str(path))[0]
            candidates = None
        inference_end = time.perf_counter()
        count = build_outputs(
            path,
            result,
            args.output,
            draw_ids=not args.no_ids,
            candidates=candidates,
            tiled=args.tile,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
        processing_end = time.perf_counter()

        inference_seconds = inference_end - image_start
        export_seconds = processing_end - inference_end
        total_seconds = processing_end - image_start
        tiled_suffix = "_tiled" if args.tile else ""
        image_key = (
            f"{path.stem}_{path.suffix.lower().lstrip('.')}{tiled_suffix}"
        )
        print(f"{path}: {count} visible olive instances -> {args.output / image_key}")
        print(
            f"  time: inference={inference_seconds:.2f}s, "
            f"post-processing/export={export_seconds:.2f}s, total={total_seconds:.2f}s"
        )


if __name__ == "__main__":
    main()
