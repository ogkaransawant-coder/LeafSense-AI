"""
Feature extraction for Coffee Leaf AI.

Calculates:
- Color (HSV)
- Shape
- Texture (GLCM)
- Edge

IMPORTANT:
Leaf area is calculated directly from the same leaf_mask used
by the image segmentation / affected-area pipeline.

Therefore:

    Shape Leaf Area
        ==
    Leaf Mask Total Pixels

This keeps dashboard measurements consistent.

Note:
The affected-area percentage is calculated separately by
image_processing.py. It is not calculated by this module.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features(
    segmented_bgr: np.ndarray,
    leaf_mask: np.ndarray,
) -> dict:
    """
    Extract all dashboard features from the segmented leaf.

    Parameters
    ----------
    segmented_bgr:
        Segmented BGR image.

    leaf_mask:
        Binary leaf mask.

        0   = background
        >0  = leaf

    Returns
    -------
    dict
        Color, shape, texture and edge features.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if segmented_bgr is None:
        raise ValueError(
            "segmented_bgr cannot be None."
        )

    if leaf_mask is None:
        raise ValueError(
            "leaf_mask cannot be None."
        )

    if not isinstance(
        segmented_bgr,
        np.ndarray,
    ):
        raise TypeError(
            "segmented_bgr must be a NumPy array."
        )

    if not isinstance(
        leaf_mask,
        np.ndarray,
    ):
        raise TypeError(
            "leaf_mask must be a NumPy array."
        )

    if segmented_bgr.ndim != 3:
        raise ValueError(
            "segmented_bgr must be a 3-channel image."
        )

    if segmented_bgr.shape[2] != 3:
        raise ValueError(
            "segmented_bgr must contain exactly 3 channels."
        )

    if leaf_mask.ndim != 2:
        raise ValueError(
            "leaf_mask must be a single-channel mask."
        )

    if segmented_bgr.shape[:2] != leaf_mask.shape[:2]:
        raise ValueError(
            "segmented_bgr and leaf_mask must have "
            "the same height and width."
        )

    # ========================================================
    # BINARY LEAF MASK
    # ========================================================

    mask = leaf_mask > 0

    # Exact number of pixels belonging to the leaf.
    leaf_pixel_count = int(
        np.count_nonzero(mask)
    )

    # ========================================================
    # COLOR FEATURES
    # ========================================================

    hsv = cv2.cvtColor(
        segmented_bgr,
        cv2.COLOR_BGR2HSV,
    )

    if leaf_pixel_count > 0:

        mean_h = float(
            np.mean(
                hsv[:, :, 0][mask]
            )
        )

        mean_s = float(
            np.mean(
                hsv[:, :, 1][mask]
            ) / 255.0
        )

        mean_v = float(
            np.mean(
                hsv[:, :, 2][mask]
            ) / 255.0
        )

    else:

        mean_h = 0.0
        mean_s = 0.0
        mean_v = 0.0

    # ========================================================
    # SHAPE FEATURES
    # ========================================================

    # OpenCV findContours expects a uint8 binary image.
    contour_mask = np.where(
        mask,
        255,
        0,
    ).astype(
        np.uint8
    )

    contours, _ = cv2.findContours(
        contour_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # IMPORTANT:
    #
    # Do not use cv2.contourArea() for leaf area.
    #
    # contourArea() calculates geometric contour area,
    # which can differ from the actual mask pixel count.
    #
    # We use the exact leaf mask pixel count.
    area = float(
        leaf_pixel_count
    )

    perimeter = 0.0
    aspect_ratio = 0.0

    if contours:

        # Find the main leaf contour.
        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        # ----------------------------------------------------
        # Perimeter
        # ----------------------------------------------------

        perimeter = float(
            cv2.arcLength(
                largest_contour,
                True,
            )
        )

        # ----------------------------------------------------
        # Bounding box / aspect ratio
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        if h > 0:

            aspect_ratio = float(
                w / h
            )

    # ========================================================
    # TEXTURE FEATURES
    # ========================================================

    gray = cv2.cvtColor(
        segmented_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    # Keep only the leaf.
    gray_masked = gray.copy()

    gray_masked[
        ~mask
    ] = 0

    if leaf_pixel_count > 0:

        # ----------------------------------------------------
        # GLCM
        # ----------------------------------------------------

        glcm = graycomatrix(
            gray_masked,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
        )

        contrast = float(
            graycoprops(
                glcm,
                "contrast",
            )[0, 0]
        )

        correlation = float(
            graycoprops(
                glcm,
                "correlation",
            )[0, 0]
        )

        homogeneity = float(
            graycoprops(
                glcm,
                "homogeneity",
            )[0, 0]
        )

    else:

        contrast = 0.0
        correlation = 0.0
        homogeneity = 0.0

    # ========================================================
    # EDGE FEATURES
    # ========================================================

    if leaf_pixel_count > 0:

        # ----------------------------------------------------
        # Canny edge detection
        # ----------------------------------------------------

        edges = cv2.Canny(
            gray_masked,
            100,
            200,
        )

        # Only count edges inside the leaf.
        edges_inside_leaf = cv2.bitwise_and(
            edges,
            edges,
            mask=contour_mask,
        )

        total_edges = int(
            np.count_nonzero(
                edges_inside_leaf
            )
        )

        # ----------------------------------------------------
        # Edge density
        # ----------------------------------------------------

        edge_density = float(
            total_edges
            / leaf_pixel_count
        )

        # ----------------------------------------------------
        # Mean gradient
        # ----------------------------------------------------

        gradient = cv2.Laplacian(
            gray_masked,
            cv2.CV_64F,
        )

        gradient_inside_leaf = (
            np.abs(gradient)[mask]
        )

        if gradient_inside_leaf.size > 0:

            mean_gradient = float(
                np.mean(
                    gradient_inside_leaf
                )
            )

        else:

            mean_gradient = 0.0

    else:

        total_edges = 0
        edge_density = 0.0
        mean_gradient = 0.0

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "color": {
            "mean_hue": round(
                mean_h,
                2,
            ),
            "mean_saturation": round(
                mean_s,
                2,
            ),
            "mean_value": round(
                mean_v,
                2,
            ),
        },

        "shape": {
            # Exact pixel area from leaf_mask.
            "leaf_area": round(
                area,
                2,
            ),

            "perimeter": round(
                perimeter,
                2,
            ),

            "aspect_ratio": round(
                aspect_ratio,
                2,
            ),
        },

        "texture": {
            "contrast": round(
                contrast,
                3,
            ),
            "correlation": round(
                correlation,
                3,
            ),
            "homogeneity": round(
                homogeneity,
                3,
            ),
        },

        "edge": {
            "edge_density": round(
                edge_density,
                3,
            ),
            "total_edges": total_edges,
            "mean_gradient": round(
                mean_gradient,
                3,
            ),
        },
    }