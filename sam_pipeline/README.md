# Olive instance segmentation

This project segments the **visible region of each olive** in crowded images.
It keeps white specular highlights inside the olive masks and separates adjacent
or partially occluded olives into different instance IDs.

## Recommended pipeline

1. **SAM 2.1 dense automatic proposals** at 1024 px. A high box-NMS threshold
   is important because neighboring olives have overlapping bounding boxes.
2. **Olive/shape filtering** removes tiny texture masks, dark gaps, very large
   regions, and implausible shapes.
3. **Mask NMS** removes duplicate proposals using mask overlap rather than box
   overlap.
4. **Per-instance closing and hole filling** makes the mask continuous through
   specular reflections without using a convex hull or hallucinating the hidden
   portion of an occluded olive.
5. Export a binary foreground mask, 16-bit instance-label image, transparent
   cutout, diagnostic glare mask, overlay, and JSON measurements.

The model works directly on the original image. Inpainting glare before
segmentation is deliberately avoided: it can erase true boundaries near wet,
touching olives. The `specular_diagnostic.png` output is for inspection only.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r sam_pipeline/requirements.txt

python sam_pipeline/olive_instance_segmentation.py \
  images/olives.jpg images/olives.png
```

### Optional overlapping tiles

Add the boolean `--tile` flag to segment a large image as overlapping tiles.
Tile size and overlap are expressed in pixels of the original image:

```bash
python sam_pipeline/olive_instance_segmentation.py images/olives.jpg \
  --tile --tile-size 1024 --tile-overlap 128
```

Tiling is disabled unless `--tile` is present. Proposals cut by an internal
tile boundary are rejected; the overlap gives a neighboring tile the chance to
capture the complete olive. Duplicate masks from overlapping tiles are merged
with mask-level NMS. Tiled outputs use an additional `_tiled` suffix so they do
not overwrite full-image results.

Use a smaller tile, such as `--tile-size 512 --tile-overlap 96`, when olives are
very small in the original photograph. Smaller tiles enlarge those objects for
SAM but increase the number of inference calls and therefore execution time.

`sam2.1_t.pt` is downloaded automatically on first use. On Apple Silicon the
script selects MPS; on NVIDIA it selects CUDA; otherwise it uses CPU.

Each image receives its own output directory under `sam_pipeline/results/`
(`olives_jpg` and `olives_png` for the supplied files):

- `instance_labels.png`: 16-bit image; 0 is background and 1..N are olives.
- `olive_mask.png`: binary union of all selected olives.
- `olive_cutout.png`: original RGB image with a transparent background.
- `overlay.png`: colored contours/tint and instance IDs.
- `instances.json`: score, area, bounding box, and centroid for each instance.
- `specular_diagnostic.png`: detected bright/low-saturation glare pixels.

For every input image, the command also prints separate wall-clock times for
SAM inference, post-processing/export, and the complete operation.

The masks cover only pixels visible in the photograph. Recovering the hidden
part of an olive underneath another olive is an amodal-segmentation problem and
cannot be measured reliably from one image without a trained amodal model or
multiple views.
