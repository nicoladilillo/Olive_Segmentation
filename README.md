# Olive segmentation

The repository is divided into two independent implementations that use the
same source images from [`images/`](images/).

Install the complete environment for both pipelines with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Smaller, pipeline-specific requirement files are also provided in each
pipeline directory.

```text
Olive_Segmentation/
├── images/                         # Shared JPG and PNG inputs
├── sam_pipeline/                   # Recommended learned instance segmentation
│   ├── olive_instance_segmentation.py
│   ├── sam2.1_t.pt
│   ├── requirements.txt
│   ├── results/
│   └── README.md
└── computer_vision_pipeline/       # Classical computer-vision experiments
    ├── olive_segmentation_specular.py
    ├── olive_segmentation_otsu.py
    ├── olive_segmentation_knn.py
    ├── olive_segmentation_knn_filled.py
    ├── requirements.txt
    ├── environment.yml
    ├── results/
    └── README.md
```

## Recommended: SAM 2.1

This is the stronger pipeline for crowded and partially occluded olives. It
produces individual instance labels and keeps specular highlights inside each
olive mask.

```bash
source .venv/bin/activate
python sam_pipeline/olive_instance_segmentation.py \
  images/olives.jpg images/olives.png
```

See [`sam_pipeline/README.md`](sam_pipeline/README.md) for output details and
parameters.

## Classical computer vision

These versions use only image processing and conventional machine learning:
HSV/Lab features, inpainting, morphology, edge detection, KNN or Otsu, distance
transform, and watershed. They are useful for comparison and for environments
where a neural model cannot be used.

```bash
source .venv/bin/activate
python computer_vision_pipeline/olive_segmentation_knn_filled.py \
  images/olives.png
```

See [`computer_vision_pipeline/README.md`](computer_vision_pipeline/README.md)
for all four variants.
