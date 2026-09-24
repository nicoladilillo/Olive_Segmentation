# Classical computer-vision pipeline

This directory contains approaches that do **not** use SAM or another neural
segmentation model.

## Variants

- `olive_segmentation_specular.py`: Lab thresholding, specular inpainting,
  black-hat/edge separators, and watershed.
- `olive_segmentation_otsu.py`: adaptive Otsu threshold on Lab `b*`, followed
  by separators and watershed.
- `olive_segmentation_knn.py`: pseudo-labelled KNN pixel classification before
  watershed instance separation.
- `olive_segmentation_knn_filled.py`: KNN pipeline plus per-instance external
  contour reconstruction and hole filling. This is the most complete classical
  variant.

## Install

From the repository root:

```bash
source .venv/bin/activate
pip install -r computer_vision_pipeline/requirements.txt
```

Alternatively:

```bash
conda env create -f computer_vision_pipeline/environment.yml
conda activate olive-segmentation-cv
```

## Run

```bash
python computer_vision_pipeline/olive_segmentation_specular.py images/olives.png
python computer_vision_pipeline/olive_segmentation_otsu.py images/olives.png
python computer_vision_pipeline/olive_segmentation_knn.py images/olives.png
python computer_vision_pipeline/olive_segmentation_knn_filled.py images/olives.png
```

Default outputs remain inside `computer_vision_pipeline/results/`, with one
subdirectory per method. Existing JPG and PNG experiments are stored there as
well.

## Limitation

These methods depend on hand-selected color and size thresholds. They are less
robust than the SAM pipeline when resolution, olive color, lighting, glare, or
occlusion changes. Watershed separates touching foreground regions but may
over-split strong highlights or miss heavily occluded olives.
