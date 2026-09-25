import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from sklearn.neighbors import KNeighborsClassifier


def run_pipeline(input_path: str, output_dir: str):
    pipeline_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------
    rgb = np.array(Image.open(input_path).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    # ---------------------------------------------------------
    # 1) SPECULAR REFLECTION DETECTION
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    specular = ((V > 195) & (S < 145)).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    specular = cv2.morphologyEx(specular, cv2.MORPH_OPEN, k3, iterations=1)
    specular = cv2.dilate(specular, k5, iterations=1)

    # ---------------------------------------------------------
    # 2) REMOVE SPECULAR HIGHLIGHTS
    # ---------------------------------------------------------
    clean_bgr = cv2.inpaint(
        bgr,
        specular,
        7,
        cv2.INPAINT_TELEA
    )

    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 3) EDGE-PRESERVING SMOOTHING
    #    Use bilateral filtering instead of strong Gaussian blur.
    # ---------------------------------------------------------
    filtered = cv2.bilateralFilter(
        clean_bgr,
        d=12,
        sigmaColor=35,
        sigmaSpace=35
    )

    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    # Optional mild Mean Shift after bilateral filtering
    mean_shift = cv2.pyrMeanShiftFiltering(
        filtered,
        sp=8,
        sr=20,
        maxLevel=1
    )

    mean_shift_rgb = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 4) BUILD PIXEL FEATURES FOR KNN
    #    Features = Lab + normalized XY position
    # ---------------------------------------------------------
    lab = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    yy, xx = np.mgrid[0:h, 0:w]

    x_norm = (xx / max(w - 1, 1)).astype(np.float32)
    y_norm = (yy / max(h - 1, 1)).astype(np.float32)

    # Spatial features are deliberately weak compared with Lab.
    spatial_weight = 20.0

    features = np.column_stack([
        L.ravel().astype(np.float32),
        a.ravel().astype(np.float32),
        b.ravel().astype(np.float32),
        spatial_weight * x_norm.ravel(),
        spatial_weight * y_norm.ravel(),
    ])

    # ---------------------------------------------------------
    # 5) CREATE HIGH-CONFIDENCE PSEUDO-LABELS
    #
    # KNN is supervised. Since no hand-labelled mask is provided,
    # use conservative color rules only to create training seeds.
    #
    # Olive seed:
    #   relatively yellow Lab b* and not extremely dark.
    #
    # Background seed:
    #   dark regions / weak yellow response.
    # ---------------------------------------------------------
    olive_seed = (
        (b > 165) &
        (L > 65)
    )

    background_seed = (
        (L < 80) |
        (b < 145)
    )

    # Avoid overlap
    background_seed = background_seed & (~olive_seed)

    olive_idx = np.flatnonzero(olive_seed.ravel())
    background_idx = np.flatnonzero(background_seed.ravel())

    if len(olive_idx) < 100 or len(background_idx) < 100:
        raise RuntimeError(
            "Not enough confident seed pixels for KNN. "
            "Adjust the olive/background seed thresholds."
        )

    # Limit training size for speed and balance the classes.
    rng = np.random.default_rng(0)
    max_per_class = 12000

    if len(olive_idx) > max_per_class:
        olive_idx = rng.choice(olive_idx, max_per_class, replace=False)

    if len(background_idx) > max_per_class:
        background_idx = rng.choice(background_idx, max_per_class, replace=False)

    train_idx = np.concatenate([olive_idx, background_idx])

    X_train = features[train_idx]
    y_train = np.concatenate([
        np.ones(len(olive_idx), dtype=np.uint8),
        np.zeros(len(background_idx), dtype=np.uint8)
    ])

    # ---------------------------------------------------------
    # 6) KNN PIXEL CLASSIFICATION
    # ---------------------------------------------------------
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1
    )

    knn.fit(X_train, y_train)

    # Classify in chunks to avoid unnecessary memory peaks.
    prediction = np.zeros(h * w, dtype=np.uint8)
    chunk = 50000

    for start in range(0, len(features), chunk):
        end = min(start + chunk, len(features))
        prediction[start:end] = knn.predict(features[start:end])

    knn_mask = prediction.reshape(h, w) * 255

    knn_mask = cv2.morphologyEx(
        knn_mask,
        cv2.MORPH_CLOSE,
        k5,
        iterations=1
    )
    
    # Conservative cleanup; do not fill large holes yet.
    knn_mask = cv2.morphologyEx(
        knn_mask,
        cv2.MORPH_OPEN,
        k5,
        iterations=1
    )
    

    # ---------------------------------------------------------
    # 7) BLACK-HAT FOR DARK GAPS BETWEEN OLIVES
    # ---------------------------------------------------------
    L_knn = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2LAB)[:, :, 0]

    bh_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (13, 13)
    )

    blackhat = cv2.morphologyEx(
        L_knn,
        cv2.MORPH_BLACKHAT,
        bh_kernel
    )

    _, blackhat_mask = cv2.threshold(
        blackhat,
        18,
        255,
        cv2.THRESH_BINARY
    )

    blackhat_mask = cv2.morphologyEx(
        blackhat_mask,
        cv2.MORPH_OPEN,
        k3,
        iterations=1
    )

    # ---------------------------------------------------------
    # 8) SCHARR + CANNY SUPPORT
    # ---------------------------------------------------------
    gray = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    magnitude = cv2.magnitude(gx, gy)

    magnitude_u8 = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    strong_gradient = (magnitude_u8 > 85).astype(np.uint8) * 255
    canny = cv2.Canny(gray, 50, 120)

    edge_support = cv2.bitwise_or(
        strong_gradient,
        canny
    )

    edge_near = cv2.dilate(
        edge_support,
        k5,
        iterations=1
    )

    separator = cv2.bitwise_and(
        blackhat_mask,
        edge_near
    )

    separator = cv2.dilate(
        separator,
        k3,
        iterations=1
    )

    # ---------------------------------------------------------
    # 9) APPLY SEPARATORS TO KNN MASK
    # ---------------------------------------------------------
    separated = knn_mask.copy()
    separated[separator > 0] = 0

    separated = cv2.morphologyEx(
        separated,
        cv2.MORPH_CLOSE,
        k5,
        iterations=1
    )
    
    separated = cv2.morphologyEx(
        separated,
        cv2.MORPH_OPEN,
        k5,
        iterations=1
    )
    

    binary = separated > 0

    # ---------------------------------------------------------
    # 10) DISTANCE TRANSFORM + WATERSHED
    # ---------------------------------------------------------
    distance = ndi.distance_transform_edt(binary)

    distance_smooth = ndi.gaussian_filter(
        distance,
        sigma=2.0
    )

    coords = peak_local_max(
        distance_smooth,
        min_distance=15,
        threshold_abs=4.0,
        labels=binary
    )

    marker_img = np.zeros((h, w), dtype=np.int32)

    for i, (r, c) in enumerate(coords, start=1):
        marker_img[r, c] = i

    markers = ndi.label(marker_img > 0)[0]

    labels = watershed(
        -distance_smooth,
        markers,
        mask=binary,
        watershed_line=True
    )

    # ---------------------------------------------------------
    # 11) KEEP PLAUSIBLE OLIVE INSTANCES
    # ---------------------------------------------------------
    accepted = np.zeros_like(labels, dtype=np.int32)
    new_id = 1

    for region in regionprops(labels):
        area = region.area

        if area < 350 or area > 3800:
            continue

        minr, minc, maxr, maxc = region.bbox
        bh = maxr - minr
        bw = maxc - minc

        aspect = max(bh, bw) / max(1, min(bh, bw))

        if not (1.0 <= aspect <= 2.8):
            continue

        if region.solidity < 0.72:
            continue

        if not (0.15 <= region.eccentricity <= 0.95):
            continue

        accepted[labels == region.label] = new_id
        new_id += 1

    n = new_id - 1

    # ---------------------------------------------------------
    # 12) MORPHOLOGY ON EACH OLIVE SEPARATELY
    # ---------------------------------------------------------
    final = np.zeros_like(accepted, dtype=np.int32)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    for obj_id in range(1, n + 1):
        obj = (accepted == obj_id).astype(np.uint8) * 255

        ys, xs = np.where(obj > 0)

        if len(xs) == 0:
            continue

        pad = 8

        x0 = max(xs.min() - pad, 0)
        x1 = min(xs.max() + pad + 1, w)
        y0 = max(ys.min() - pad, 0)
        y1 = min(ys.max() + pad + 1, h)

        crop = obj[y0:y1, x0:x1]

        closed = cv2.morphologyEx(
            crop,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        )

        padded = cv2.copyMakeBorder(
            closed,
            1, 1, 1, 1,
            cv2.BORDER_CONSTANT,
            value=0
        )

        flood = padded.copy()

        flood_mask = np.zeros(
            (padded.shape[0] + 2, padded.shape[1] + 2),
            dtype=np.uint8
        )

        cv2.floodFill(
            flood,
            flood_mask,
            (0, 0),
            255
        )

        holes = cv2.bitwise_not(flood)

        filled = cv2.bitwise_or(
            padded,
            holes
        )[1:-1, 1:-1]

        # Restrict expansion to the neighborhood of the detected instance.
        allowable = cv2.dilate(
            crop,
            k3,
            iterations=2
        )

        filled = cv2.bitwise_and(
            filled,
            allowable
        )

        sub = final[y0:y1, x0:x1]
        add = (filled > 0) & (sub == 0)
        sub[add] = obj_id
        final[y0:y1, x0:x1] = sub

    # ---------------------------------------------------------
    # 13) FINAL OVERLAY
    # ---------------------------------------------------------
    overlay = rgb.copy()

    for obj_id in range(1, n + 1):
        mask_i = (final == obj_id).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_i,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 0, 0),
            2
        )

        ys, xs = np.where(final == obj_id)

        if len(xs):
            cx = int(xs.mean())
            cy = int(ys.mean())

            cv2.putText(
                overlay,
                str(obj_id),
                (cx - 6, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    # ---------------------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------------------
    cv2.imwrite(str(output_dir / "01_specular_mask.png"), specular)
    cv2.imwrite(str(output_dir / "02_specular_removed.png"), clean_bgr)
    cv2.imwrite(str(output_dir / "03_bilateral_mean_shift.png"), mean_shift)
    cv2.imwrite(str(output_dir / "04_knn_mask.png"), knn_mask)
    cv2.imwrite(str(output_dir / "05_blackhat.png"), blackhat)
    cv2.imwrite(str(output_dir / "06_separator.png"), separator)
    cv2.imwrite(str(output_dir / "07_separated_knn_mask.png"), separated)
    cv2.imwrite(
        str(output_dir / "08_final_mask.png"),
        (final > 0).astype(np.uint8) * 255
    )
    cv2.imwrite(
        str(output_dir / "09_final_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    # ---------------------------------------------------------
    # COMPARISON FIGURE
    # ---------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(14, 9))

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(clean_rgb)
    ax[0, 1].set_title("Specular removal")

    ax[0, 2].imshow(mean_shift_rgb)
    ax[0, 2].set_title("Bilateral + Mean Shift")

    ax[1, 0].imshow(knn_mask, cmap="gray")
    ax[1, 0].set_title("KNN olive/background mask")

    ax[1, 1].imshow(separator, cmap="gray")
    ax[1, 1].set_title("Black-hat + derivative separators")

    ax[1, 2].imshow(overlay)
    ax[1, 2].set_title(f"Final selected olives: {n}")

    for axis in ax.ravel():
        axis.axis("off")

    plt.tight_layout()

    comparison_path = output_dir / "10_pipeline_comparison.png"

    plt.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"KNN olive seed pixels: {len(olive_idx)}")
    print(f"KNN background seed pixels: {len(background_idx)}")
    print(f"Selected olive regions: {n}")
    print(f"Results saved in: {output_dir.resolve()}")
    print(f"Total processing time: {time.perf_counter() - pipeline_start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Olive segmentation with pseudo-labelled KNN pixel classification, "
            "specular removal, black-hat separators and watershed."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image."
    )

    parser.add_argument(
        "--output",
        default=Path(__file__).resolve().parent / "results" / "knn",
        help="Output directory. Default: computer_vision_pipeline/results/knn"
    )

    args = parser.parse_args()

    run_pipeline(
        args.image,
        args.output
    )
