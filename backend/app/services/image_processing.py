"""
CoffeeLeaf AI - Image Processing

Responsibilities:
- Validate uploaded image
- Prepare tensor for EfficientNet-B0
- Segment the main foreground leaf
- Generate affected-region visualization
- Calculate affected-area percentage

Important:
- The classifier preprocessing remains 224x224 + ImageNet normalization.
- Leaf segmentation is traditional computer vision.
- Affected-area calculation is a computer-vision heuristic.
- This is NOT a trained disease-segmentation model.
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.utils.exceptions import InvalidImageError


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
}

# Maximum image dimension used by the CV segmentation branch.
MAX_SEGMENTATION_SIZE = 900

# Ignore very tiny connected components.
MIN_LEAF_COMPONENT_AREA = 500


# ============================================================
# MODEL PREPROCESSING
# ============================================================

_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ]
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def _validate_decodable(
    file_bytes: bytes,
) -> None:
    """
    Verify that OpenCV can decode the uploaded image.
    """

    if not file_bytes:
        raise InvalidImageError(
            "Empty file uploaded."
        )

    arr = np.frombuffer(
        file_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise InvalidImageError(
            "Uploaded file is not a valid image."
        )


def _open_image(
    file_bytes: bytes,
) -> Image.Image:
    """
    Open and validate uploaded image using Pillow.
    """

    if not file_bytes:
        raise InvalidImageError(
            "Empty file uploaded."
        )

    try:
        probe = Image.open(
            io.BytesIO(file_bytes)
        )

        probe.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:

        raise InvalidImageError(
            f"Corrupted image: {exc}"
        ) from exc

    try:
        image = Image.open(
            io.BytesIO(file_bytes)
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:

        raise InvalidImageError(
            f"Unable to open image: {exc}"
        ) from exc

    fmt = (
        image.format or ""
    ).upper()

    if fmt not in SUPPORTED_FORMATS:

        raise InvalidImageError(
            f"Unsupported format '{fmt}'. "
            f"Supported: "
            f"{', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return image.convert("RGB")


# ============================================================
# MODEL INPUT
# ============================================================

def preprocess_image(
    file_bytes: bytes,
) -> torch.Tensor:
    """
    Prepare uploaded image for EfficientNet-B0.

    Output:
        Tensor shape = (1, 3, 224, 224)
    """

    if not file_bytes:
        raise InvalidImageError(
            "Empty file uploaded."
        )

    _validate_decodable(
        file_bytes
    )

    image = _open_image(
        file_bytes
    )

    tensor = _transform(
        image
    )

    return tensor.unsqueeze(0)


# ============================================================
# SEGMENTATION RESIZE
# ============================================================

def _resize_for_segmentation(
    original: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Resize large images before segmentation.

    Returns:
        resized_image,
        scale_factor
    """

    height, width = original.shape[:2]

    largest_dimension = max(
        height,
        width,
    )

    scale = min(
        1.0,
        MAX_SEGMENTATION_SIZE
        / float(largest_dimension),
    )

    if scale < 1.0:

        small = cv2.resize(
            original,
            (
                max(
                    1,
                    int(round(width * scale)),
                ),
                max(
                    1,
                    int(round(height * scale)),
                ),
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:

        small = original.copy()

    return small, scale


# ============================================================
# GREEN CANDIDATE
# ============================================================

def _create_green_candidate(
    image: np.ndarray,
) -> np.ndarray:
    """
    Create a conservative green-leaf candidate mask.

    This is a hint for segmentation.
    It is NOT the final leaf mask.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # Coffee leaves can vary from yellow-green
    # to dark green.
    lower_green = np.array(
        [18, 30, 25],
        dtype=np.uint8,
    )

    upper_green = np.array(
        [105, 255, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green,
    )

    # Remove isolated noise.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    # Connect nearby leaf pixels.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    return mask


# ============================================================
# CENTER PRIOR
# ============================================================

def _create_center_prior(
    height: int,
    width: int,
) -> np.ndarray:
    """
    Create a smooth center-weighted prior.

    The application expects the main leaf to generally
    appear near the center/foreground of the uploaded image.
    """

    yy, xx = np.mgrid[
        0:height,
        0:width,
    ]

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    nx = (
        xx - cx
    ) / max(
        width,
        1,
    )

    ny = (
        yy - cy
    ) / max(
        height,
        1,
    )

    distance = np.sqrt(
        nx * nx
        + ny * ny
    )

    prior = np.exp(
        -(
            distance ** 2
        )
        / (
            2.0 * 0.42 ** 2
        )
    )

    return prior


# ============================================================
# CONNECTED COMPONENT CLEANUP
# ============================================================

def _select_main_component(
    mask: np.ndarray,
    green_mask: np.ndarray,
    center_prior: np.ndarray,
) -> np.ndarray:
    """
    Select the component most likely to be the main
    foreground leaf.

    Selection considers:
    - component area
    - center proximity
    - green overlap
    - border contact
    """

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return np.zeros_like(mask)

    image_height, image_width = mask.shape

    image_area = (
        image_height
        * image_width
    )

    candidates = []

    for label_index in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label_index,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < MIN_LEAF_COMPONENT_AREA:
            continue

        component_mask = (
            labels == label_index
        )

        # ----------------------------------------------------
        # Green overlap
        # ----------------------------------------------------

        component_pixels = np.count_nonzero(
            component_mask
        )

        green_pixels = np.count_nonzero(
            green_mask[
                component_mask
            ]
        )

        green_ratio = (
            green_pixels
            / max(
                component_pixels,
                1,
            )
        )

        # ----------------------------------------------------
        # Center proximity
        # ----------------------------------------------------

        component_center_x = (
            float(
                centroids[
                    label_index,
                    0,
                ]
            )
        )

        component_center_y = (
            float(
                centroids[
                    label_index,
                    1,
                ]
            )
        )

        normalized_x = (
            component_center_x
            - image_width / 2.0
        ) / max(
            image_width / 2.0,
            1.0,
        )

        normalized_y = (
            component_center_y
            - image_height / 2.0
        ) / max(
            image_height / 2.0,
            1.0,
        )

        center_distance = np.sqrt(
            normalized_x ** 2
            + normalized_y ** 2
        )

        centrality = max(
            0.05,
            1.0 - center_distance,
        )

        # ----------------------------------------------------
        # Average center-prior support
        # ----------------------------------------------------

        prior_values = center_prior[
            component_mask
        ]

        prior_score = float(
            np.mean(
                prior_values
            )
        ) if prior_values.size else 0.0

        # ----------------------------------------------------
        # Border contact penalty
        # ----------------------------------------------------

        x = int(
            stats[
                label_index,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label_index,
                cv2.CC_STAT_TOP,
            ]
        )

        component_width = int(
            stats[
                label_index,
                cv2.CC_STAT_WIDTH,
            ]
        )

        component_height = int(
            stats[
                label_index,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        touches_border = (
            x <= 1
            or y <= 1
            or (
                x + component_width
                >= image_width - 1
            )
            or (
                y + component_height
                >= image_height - 1
            )
        )

        border_penalty = (
            0.72
            if touches_border
            else 1.0
        )

        # ----------------------------------------------------
        # Area score
        # ----------------------------------------------------

        area_ratio = (
            area
            / max(
                image_area,
                1,
            )
        )

        # Very tiny components have already been removed.
        # Avoid allowing a giant background component to win.
        area_score = min(
            1.0,
            np.sqrt(
                area_ratio * 8.0
            ),
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        score = (
            area_score
            * (
                0.35
                + 0.65 * centrality
            )
            * (
                0.40
                + 0.60 * green_ratio
            )
            * (
                0.40
                + 0.60 * prior_score
            )
            * border_penalty
        )

        candidates.append(
            (
                float(score),
                label_index,
                area,
            )
        )

    if not candidates:
        return np.zeros_like(mask)

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_label = candidates[0][1]

    selected = np.zeros_like(
        mask
    )

    selected[
        labels == best_label
    ] = 255

    return selected


# ============================================================
# LEAF MASK
# ============================================================

def _create_leaf_mask(
    original: np.ndarray,
) -> np.ndarray:
    """
    Segment the main foreground leaf.

    Improved pipeline:

        Green candidate
              ↓
        Center prior
              ↓
        GrabCut initialization
              ↓
        GrabCut
              ↓
        Green support filtering
              ↓
        Connected component scoring
              ↓
        Main component selection
              ↓
        Morphological cleanup
              ↓
        Contour refinement
              ↓
        Original resolution
    """

    height, width = original.shape[:2]

    small, scale = (
        _resize_for_segmentation(
            original
        )
    )

    h, w = small.shape[:2]

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2HSV,
    )

    green_mask = (
        _create_green_candidate(
            small
        )
    )

    center_prior = (
        _create_center_prior(
            h,
            w,
        )
    )

    # ========================================================
    # GRABCUT INITIALIZATION
    # ========================================================

    gc_mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Definite background border
    # --------------------------------------------------------

    border = max(
        4,
        int(
            min(h, w)
            * 0.035
        ),
    )

    gc_mask[
        :border,
        :
    ] = cv2.GC_BGD

    gc_mask[
        -border:,
        :
    ] = cv2.GC_BGD

    gc_mask[
        :,
        :border
    ] = cv2.GC_BGD

    gc_mask[
        :,
        -border:
    ] = cv2.GC_BGD

    # ========================================================
    # PROBABLE FOREGROUND
    # ========================================================

    probable_foreground = (
        (green_mask > 0)
        & (
            center_prior > 0.12
        )
    )

    strong_foreground = (
        (green_mask > 0)
        & (
            center_prior > 0.38
        )
        & (
            hsv[:, :, 1] > 40
        )
        & (
            hsv[:, :, 2] > 35
        )
    )

    gc_mask[
        probable_foreground
    ] = cv2.GC_PR_FGD

    gc_mask[
        strong_foreground
    ] = cv2.GC_FGD

    # ========================================================
    # CENTRAL SEED
    # ========================================================

    seed_h = max(
        12,
        int(h * 0.16),
    )

    seed_w = max(
        12,
        int(w * 0.16),
    )

    cx = int(
        (w - 1) / 2.0
    )

    cy = int(
        (h - 1) / 2.0
    )

    x1 = max(
        0,
        cx - seed_w // 2,
    )

    x2 = min(
        w,
        cx + seed_w // 2,
    )

    y1 = max(
        0,
        cy - seed_h // 2,
    )

    y2 = min(
        h,
        cy + seed_h // 2,
    )

    central_green = (
        green_mask[
            y1:y2,
            x1:x2
        ] > 0
    )

    if np.count_nonzero(
        central_green
    ) > 20:

        region = gc_mask[
            y1:y2,
            x1:x2
        ]

        region[
            central_green
        ] = cv2.GC_FGD

        gc_mask[
            y1:y2,
            x1:x2
        ] = region

    # ========================================================
    # CORNER BACKGROUND
    # ========================================================

    corner = max(
        5,
        int(
            min(h, w)
            * 0.10
        ),
    )

    gc_mask[
        :corner,
        :corner
    ] = cv2.GC_BGD

    gc_mask[
        :corner,
        -corner:
    ] = cv2.GC_BGD

    gc_mask[
        -corner:,
        :corner
    ] = cv2.GC_BGD

    gc_mask[
        -corner:,
        -corner:
    ] = cv2.GC_BGD

    # ========================================================
    # VERIFY FOREGROUND
    # ========================================================

    has_foreground = np.any(
        (
            gc_mask
            == cv2.GC_FGD
        )
        |
        (
            gc_mask
            == cv2.GC_PR_FGD
        )
    )

    if not has_foreground:

        fallback = (
            green_mask.copy()
        )

        if scale < 1.0:

            fallback = cv2.resize(
                fallback,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        return fallback

    # ========================================================
    # GRABCUT
    # ========================================================

    bgd_model = np.zeros(
        (1, 65),
        np.float64,
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64,
    )

    try:

        cv2.grabCut(
            small,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            6,
            cv2.GC_INIT_WITH_MASK,
        )

    except cv2.error as exc:

        logger.warning(
            "GrabCut failed: %s",
            exc,
        )

        fallback = (
            green_mask.copy()
        )

        if scale < 1.0:

            fallback = cv2.resize(
                fallback,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        return fallback

    # ========================================================
    # GRABCUT → BINARY MASK
    # ========================================================

    leaf_mask_small = np.where(
        (
            (
                gc_mask
                == cv2.GC_FGD
            )
            |
            (
                gc_mask
                == cv2.GC_PR_FGD
            )
        ),
        255,
        0,
    ).astype(
        np.uint8
    )

    # ========================================================
    # GREEN SUPPORT
    # ========================================================

    green_dilated = cv2.dilate(
        green_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (13, 13),
        ),
        iterations=1,
    )

    supported = cv2.bitwise_and(
        leaf_mask_small,
        green_dilated,
    )

    original_area = cv2.countNonZero(
        leaf_mask_small
    )

    supported_area = cv2.countNonZero(
        supported
    )

    # Only replace with the green-supported mask
    # if it still contains a meaningful amount
    # of the original segmentation.
    if (
        original_area > 0
        and supported_area
        >= original_area * 0.25
    ):

        leaf_mask_small = supported

    # ========================================================
    # MORPHOLOGICAL CLEANUP
    # ========================================================

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    leaf_mask_small = cv2.morphologyEx(
        leaf_mask_small,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    leaf_mask_small = cv2.morphologyEx(
        leaf_mask_small,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    # ========================================================
    # SELECT MAIN LEAF COMPONENT
    # ========================================================

    leaf_mask_small = (
        _select_main_component(
            leaf_mask_small,
            green_mask,
            center_prior,
        )
    )

    # ========================================================
    # IF COMPONENT SELECTION FAILED
    # ========================================================

    if cv2.countNonZero(
        leaf_mask_small
    ) < MIN_LEAF_COMPONENT_AREA:

        fallback = (
            _select_main_component(
                green_mask,
                green_mask,
                center_prior,
            )
        )

        if cv2.countNonZero(
            fallback
        ) >= MIN_LEAF_COMPONENT_AREA:

            leaf_mask_small = fallback

        else:

            leaf_mask_small = (
                green_mask.copy()
            )

    # ========================================================
    # CONTOUR REFINEMENT
    # ========================================================

    contours, _ = cv2.findContours(
        leaf_mask_small,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:

        contours = [
            contour
            for contour in contours
            if cv2.contourArea(
                contour
            ) >= MIN_LEAF_COMPONENT_AREA
        ]

        if contours:

            # Choose contour closest to image center
            # while preserving enough area.
            contour_candidates = []

            image_center = np.array(
                [
                    w / 2.0,
                    h / 2.0,
                ]
            )

            for contour in contours:

                area = cv2.contourArea(
                    contour
                )

                moments = cv2.moments(
                    contour
                )

                if moments["m00"] > 0:

                    contour_center = np.array(
                        [
                            moments["m10"]
                            / moments["m00"],
                            moments["m01"]
                            / moments["m00"],
                        ]
                    )

                else:

                    contour_center = image_center

                distance = np.linalg.norm(
                    contour_center
                    - image_center
                )

                centrality = 1.0 / (
                    1.0
                    + distance
                    / max(
                        min(w, h),
                        1,
                    )
                )

                contour_score = (
                    np.sqrt(
                        max(area, 1)
                    )
                    * (
                        0.60
                        + 0.40
                        * centrality
                    )
                )

                contour_candidates.append(
                    (
                        contour_score,
                        contour,
                    )
                )

            contour_candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            best_contour = (
                contour_candidates[0][1]
            )

            refined = np.zeros_like(
                leaf_mask_small
            )

            cv2.drawContours(
                refined,
                [
                    best_contour
                ],
                -1,
                255,
                thickness=cv2.FILLED,
            )

            leaf_mask_small = refined

    # ========================================================
    # FINAL SMOOTHING
    # ========================================================

    leaf_mask_small = cv2.morphologyEx(
        leaf_mask_small,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
        iterations=1,
    )

    # ========================================================
    # RESIZE TO ORIGINAL SIZE
    # ========================================================

    if scale < 1.0:

        leaf_mask = cv2.resize(
            leaf_mask_small,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    else:

        leaf_mask = (
            leaf_mask_small
        )

    # ========================================================
    # FINAL CLEANUP AT ORIGINAL RESOLUTION
    # ========================================================

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
        iterations=1,
    )

    # Remove tiny final components.
    leaf_mask = (
        _remove_small_components(
            leaf_mask,
            min_area=max(
                500,
                int(
                    height
                    * width
                    * 0.0002
                ),
            ),
        )
    )

    return leaf_mask


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def _remove_small_components(
    mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    Remove tiny disconnected regions.
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return mask

    cleaned = np.zeros_like(
        mask
    )

    for label_index in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label_index,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= min_area:

            cleaned[
                labels == label_index
            ] = 255

    return cleaned


# ============================================================
# AFFECTED REGION
# ============================================================

def _create_affected_mask(
    original: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """
    Estimate visually abnormal regions inside the detected leaf.

    This is a computer-vision heuristic.

    It detects:
    - brown/rust-like regions
    - reddish-brown regions
    - yellow/chlorotic regions
    """

    hsv = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2HSV,
    )

    # --------------------------------------------------------
    # Remove leaf boundary
    # --------------------------------------------------------

    interior_mask = cv2.erode(
        leaf_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
        iterations=1,
    )

    # ========================================================
    # BROWN / RUST
    # ========================================================

    lower_brown = np.array(
        [5, 70, 30],
        dtype=np.uint8,
    )

    upper_brown = np.array(
        [25, 255, 210],
        dtype=np.uint8,
    )

    brown_mask = cv2.inRange(
        hsv,
        lower_brown,
        upper_brown,
    )

    # ========================================================
    # RED
    # ========================================================

    lower_red_1 = np.array(
        [0, 80, 30],
        dtype=np.uint8,
    )

    upper_red_1 = np.array(
        [10, 255, 200],
        dtype=np.uint8,
    )

    red_mask_1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1,
    )

    lower_red_2 = np.array(
        [165, 80, 30],
        dtype=np.uint8,
    )

    upper_red_2 = np.array(
        [179, 255, 200],
        dtype=np.uint8,
    )

    red_mask_2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2,
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2,
    )

    # ========================================================
    # YELLOW / CHLOROSIS
    # ========================================================

    lower_yellow = np.array(
        [24, 80, 80],
        dtype=np.uint8,
    )

    upper_yellow = np.array(
        [38, 255, 255],
        dtype=np.uint8,
    )

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow,
    )

    # ========================================================
    # COMBINE
    # ========================================================

    affected_mask = cv2.bitwise_or(
        brown_mask,
        red_mask,
    )

    affected_mask = cv2.bitwise_or(
        affected_mask,
        yellow_mask,
    )

    # ========================================================
    # ONLY INSIDE LEAF
    # ========================================================

    affected_mask = cv2.bitwise_and(
        affected_mask,
        interior_mask,
    )

    # ========================================================
    # REMOVE NOISE
    # ========================================================

    affected_mask = cv2.morphologyEx(
        affected_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    affected_mask = cv2.morphologyEx(
        affected_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
        iterations=1,
    )

    # ========================================================
    # REMOVE TINY COMPONENTS
    # ========================================================

    image_area = (
        original.shape[0]
        * original.shape[1]
    )

    min_component_area = max(
        40,
        int(
            image_area
            * 0.00015
        ),
    )

    affected_mask = (
        _remove_small_components(
            affected_mask,
            min_component_area,
        )
    )

    # Never allow affected region outside leaf.
    affected_mask = cv2.bitwise_and(
        affected_mask,
        leaf_mask,
    )

    return affected_mask


# ============================================================
# MAIN SEGMENTATION PIPELINE
# ============================================================

def segment_leaf(
    file_bytes: bytes,
):
    """
    Generate:

    - original_bgr
    - segmented_bgr
    - affected_bgr
    - leaf_mask
    - affected_mask
    - stats
    """

    # ========================================================
    # VALIDATE
    # ========================================================

    if not file_bytes:

        raise InvalidImageError(
            "Empty file uploaded."
        )

    _validate_decodable(
        file_bytes
    )

    # ========================================================
    # DECODE
    # ========================================================

    arr = np.frombuffer(
        file_bytes,
        np.uint8,
    )

    original = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR,
    )

    if original is None:

        raise InvalidImageError(
            "Unable to decode uploaded image."
        )

    # ========================================================
    # LEAF SEGMENTATION
    # ========================================================

    leaf_mask = _create_leaf_mask(
        original
    )

    leaf_pixels = int(
        cv2.countNonZero(
            leaf_mask
        )
    )

    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    if leaf_pixels == 0:

        logger.warning(
            "Leaf segmentation returned an empty mask."
        )

        # Use green detection as a fallback.
        fallback = _create_green_candidate(
            original
        )

        leaf_mask = fallback

        leaf_pixels = int(
            cv2.countNonZero(
                leaf_mask
            )
        )

    # If still empty, use the whole image.
    # This prevents the API from crashing.
    if leaf_pixels == 0:

        leaf_mask = np.full(
            original.shape[:2],
            255,
            dtype=np.uint8,
        )

        leaf_pixels = int(
            cv2.countNonZero(
                leaf_mask
            )
        )

    # ========================================================
    # SEGMENTED IMAGE
    # ========================================================

    segmented = cv2.bitwise_and(
        original,
        original,
        mask=leaf_mask,
    )

    # ========================================================
    # AFFECTED REGION
    # ========================================================

    affected_mask = (
        _create_affected_mask(
            original,
            leaf_mask,
        )
    )

    affected_pixels = int(
        cv2.countNonZero(
            affected_mask
        )
    )

    affected_pixels = min(
        affected_pixels,
        leaf_pixels,
    )

    # ========================================================
    # AFFECTED PERCENTAGE
    # ========================================================

    if leaf_pixels > 0:

        percentage = (
            affected_pixels
            / leaf_pixels
        ) * 100.0

    else:

        percentage = 0.0

    percentage = max(
        0.0,
        min(
            percentage,
            100.0,
        ),
    )

    percentage = round(
        percentage,
        2,
    )

    # ========================================================
    # AFFECTED VISUALIZATION
    # ========================================================

    affected = (
        original.copy()
    )

    # OpenCV BGR:
    # (0, 0, 255) = red.
    red_overlay = np.zeros_like(
        original,
        dtype=np.uint8,
    )

    red_overlay[:, :] = (
        0,
        0,
        255,
    )

    alpha = 0.45

    blended = cv2.addWeighted(
        original,
        1.0 - alpha,
        red_overlay,
        alpha,
        0.0,
    )

    affected_pixels_mask = (
        affected_mask > 0
    )

    affected[
        affected_pixels_mask
    ] = blended[
        affected_pixels_mask
    ]

    # ========================================================
    # STATISTICS
    # ========================================================

    stats = {
        "leaf_pixels": leaf_pixels,
        "affected_pixels": affected_pixels,
        "percentage": percentage,
    }

    # ========================================================
    # RETURN
    # ========================================================

    return (
        original,
        segmented,
        affected,
        leaf_mask,
        affected_mask,
        stats,
    )


# ============================================================
# PNG ENCODING
# ============================================================

def encode_png(
    image: np.ndarray,
) -> bytes:
    """
    Encode an OpenCV image as PNG bytes.
    """

    if image is None:

        raise RuntimeError(
            "Failed to encode empty image."
        )

    ok, buffer = cv2.imencode(
        ".png",
        image,
    )

    if not ok:

        raise RuntimeError(
            "Failed to encode image."
        )

    return buffer.tobytes()