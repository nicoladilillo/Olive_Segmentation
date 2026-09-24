import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from pathlib import Path


def run_pipeline(input_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(input_path).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    # 1) Detect specular reflection in HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    specular = ((V > 185) & (S < 145)).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    specular = cv2.morphologyEx(
        specular, cv2.MORPH_OPEN, k3, iterations=1
    )
    specular = cv2.dilate(
        specular, k5, iterations=1
    )

    # 2) Remove highlights by Telea inpainting
    clean_bgr = cv2.inpaint(
        bgr,
        specular,
        5,
        cv2.INPAINT_TELEA
    )
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

    # 3) Gaussian blur + Mean Shift
    blur = cv2.GaussianBlur(clean_bgr, (9, 9), 0)

    ms = cv2.pyrMeanShiftFiltering(
        blur,
        sp=12,
        sr=30,
        maxLevel=1
    )
    ms_rgb = cv2.cvtColor(ms, cv2.COLOR_BGR2RGB)

    # 4) Lab olive-like foreground mask, no Otsu
    lab = cv2.cvtColor(ms, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    base = ((b > 148) & (L > 45)).astype(np.uint8) * 255

    # 5) Black-hat transform to detect dark gaps
    bh_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (13, 13)
    )

    blackhat = cv2.morphologyEx(
        L,
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

    # 6) Scharr + Canny on cleaned image
    gray = cv2.cvtColor(ms, cv2.COLOR_BGR2GRAY)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

    mag = cv2.magnitude(gx, gy)

    mag_u8 = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    strong_grad = (mag_u8 > 85).astype(np.uint8) * 255

    canny = cv2.Canny(
        gray,
        50,
        120
    )

    edge_support = cv2.bitwise_or(
        strong_grad,
        canny
    )

    # 7) Combine black-hat + derivative support
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

    # 8) Build separated foreground
    separated = base.copy()
    separated[separator > 0] = 0

    separated = cv2.morphologyEx(
        separated,
        cv2.MORPH_OPEN,
        k3,
        iterations=1
    )

    binary = separated > 0

    # 9) Distance transform + watershed
    distance = ndi.distance_transform_edt(binary)

    distance_smooth = ndi.gaussian_filter(
        distance,
        sigma=2.2
    )

    coords = peak_local_max(
        distance_smooth,
        min_distance=16,
        threshold_abs=4.5,
        labels=binary
    )

    marker_img = np.zeros(
        (h, w),
        dtype=np.int32
    )

    for i, (r, c) in enumerate(coords, start=1):
        marker_img[r, c] = i

    markers = ndi.label(
        marker_img > 0
    )[0]

    labels = watershed(
        -distance_smooth,
        markers,
        mask=binary,
        watershed_line=True
    )

    # 10) Keep only plausible olive regions
    accepted = np.zeros_like(
        labels,
        dtype=np.int32
    )

    new_id = 1

    for region in regionprops(labels):
        area = region.area

        if area < 400 or area > 3600:
            continue

        minr, minc, maxr, maxc = region.bbox

        bh = maxr - minr
        bw = maxc - minc

        aspect = max(bh, bw) / max(
            1,
            min(bh, bw)
        )

        if not (1.0 <= aspect <= 2.7):
            continue

        if region.solidity < 0.74:
            continue

        if not (0.20 <= region.eccentricity <= 0.93):
            continue

        accepted[labels == region.label] = new_id
        new_id += 1

    n = new_id - 1

    # 11) Morphology on each olive after identification
    final = np.zeros_like(
        accepted,
        dtype=np.int32
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    for obj_id in range(1, n + 1):
        obj = (
            accepted == obj_id
        ).astype(np.uint8) * 255

        ys, xs = np.where(
            obj > 0
        )

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

        # Hole filling
        padded = cv2.copyMakeBorder(
            closed,
            1, 1, 1, 1,
            cv2.BORDER_CONSTANT,
            value=0
        )

        flood = padded.copy()

        ffmask = np.zeros(
            (
                padded.shape[0] + 2,
                padded.shape[1] + 2
            ),
            np.uint8
        )

        cv2.floodFill(
            flood,
            ffmask,
            (0, 0),
            255
        )

        holes = cv2.bitwise_not(
            flood
        )

        filled = cv2.bitwise_or(
            padded,
            holes
        )[1:-1, 1:-1]

        # Restrict expansion around original detected instance
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

        add = (
            (filled > 0) &
            (sub == 0)
        )

        sub[add] = obj_id
        final[y0:y1, x0:x1] = sub

    # 12) Final overlay
    overlay = rgb.copy()

    for obj_id in range(1, n + 1):
        mask_i = (
            final == obj_id
        ).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(
            mask_i,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            cnts,
            -1,
            (255, 0, 0),
            2
        )

        ys, xs = np.where(
            final == obj_id
        )

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

    # Save outputs
    cv2.imwrite(
        str(output_dir / "01_specular_mask.png"),
        specular
    )

    cv2.imwrite(
        str(output_dir / "02_specular_removed.png"),
        clean_bgr
    )

    cv2.imwrite(
        str(output_dir / "03_mean_shift.png"),
        ms
    )

    cv2.imwrite(
        str(output_dir / "04_blackhat.png"),
        blackhat
    )

    cv2.imwrite(
        str(output_dir / "05_separator.png"),
        separator
    )

    cv2.imwrite(
        str(output_dir / "06_final_mask.png"),
        (final > 0).astype(np.uint8) * 255
    )

    cv2.imwrite(
        str(output_dir / "07_final_overlay.png"),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    # Comparison figure
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(14, 9)
    )

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Original")

    ax[0, 1].imshow(
        specular,
        cmap="gray"
    )
    ax[0, 1].set_title("Specular reflection mask")

    ax[0, 2].imshow(clean_rgb)
    ax[0, 2].set_title("After Telea inpainting")

    ax[1, 0].imshow(ms_rgb)
    ax[1, 0].set_title("Blur + Mean Shift")

    ax[1, 1].imshow(
        separator,
        cmap="gray"
    )
    ax[1, 1].set_title("Black-hat + derivative separators")

    ax[1, 2].imshow(overlay)
    ax[1, 2].set_title(
        f"Final selected olives: {n}"
    )

    for axis in ax.ravel():
        axis.axis("off")

    plt.tight_layout()

    comparison_path = output_dir / "08_pipeline_comparison.png"

    plt.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Selected olive regions: {n}")
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Olive segmentation with specular reflection removal."
    )

    parser.add_argument(
        "image",
        help="Path to the input image."
    )

    parser.add_argument(
        "--output",
        default=Path(__file__).resolve().parent / "results" / "specular",
        help="Output directory. Default: computer_vision_pipeline/results/specular"
    )

    args = parser.parse_args()

    run_pipeline(
        args.image,
        args.output
    )
