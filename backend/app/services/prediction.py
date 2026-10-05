"""
Prediction service for the RoCoLe multiclass coffee-leaf model.

Pipeline:
    1. Validate uploaded image.
    2. Preprocess image for EfficientNet-B0.
    3. Load the 6-class model lazily.
    4. Run PyTorch inference.
    5. Interpret the predicted class.
    6. Segment the leaf.
    7. Estimate visually affected area.
    8. Extract HSV, shape, texture and edge features.
    9. Return a dashboard-ready response.

Model classes:
    0 -> healthy
    1 -> red_spider_mite
    2 -> rust_level_1
    3 -> rust_level_2
    4 -> rust_level_3
    5 -> rust_level_4

IMPORTANT:
    The disease prediction comes from the trained multiclass model.

    The affected-area calculation is a computer-vision heuristic.
    It should be treated as visual evidence, not as an independent
    disease diagnosis.
"""

from __future__ import annotations

import base64
import logging
import time

import torch
import torch.nn.functional as F

from app.models import model_loader
from app.services.feature_extraction import extract_features
from app.services.image_processing import (
    encode_png,
    preprocess_image,
    segment_leaf,
)
from app.utils.exceptions import InvalidImageError


logger = logging.getLogger(__name__)


# ============================================================
# EXPECTED MODEL CLASSES
# ============================================================

EXPECTED_CLASSES = {
    "healthy",
    "red_spider_mite",
    "rust_level_1",
    "rust_level_2",
    "rust_level_3",
    "rust_level_4",
}


EXPECTED_LABEL_MAPPING = {
    0: "healthy",
    1: "red_spider_mite",
    2: "rust_level_1",
    3: "rust_level_2",
    4: "rust_level_3",
    5: "rust_level_4",
}


# ============================================================
# CLASS INTERPRETATION
# ============================================================

CLASS_INFO = {
    "healthy": {
        "display_name": "Healthy",
        "disease_type": "healthy",
        "severity": None,
        "recommendation": (
            "No major visual disease class was detected. "
            "Continue regular monitoring of the plant."
        ),
    },

    "red_spider_mite": {
        "display_name": "Red Spider Mite",
        "disease_type": "red_spider_mite",
        "severity": "Detected",
        "recommendation": (
            "Inspect nearby leaves and plants for similar "
            "symptoms and consider consulting an agricultural "
            "expert."
        ),
    },

    "rust_level_1": {
        "display_name": "Coffee Leaf Rust",
        "disease_type": "coffee_leaf_rust",
        "severity": "Level 1",
        "recommendation": (
            "Inspect nearby leaves and plants for similar "
            "symptoms and consider consulting an agricultural "
            "expert."
        ),
    },

    "rust_level_2": {
        "display_name": "Coffee Leaf Rust",
        "disease_type": "coffee_leaf_rust",
        "severity": "Level 2",
        "recommendation": (
            "Inspect nearby leaves and plants for similar "
            "symptoms and consider consulting an agricultural "
            "expert."
        ),
    },

    "rust_level_3": {
        "display_name": "Coffee Leaf Rust",
        "disease_type": "coffee_leaf_rust",
        "severity": "Level 3",
        "recommendation": (
            "Inspect nearby leaves and plants for similar "
            "symptoms and consider consulting an agricultural "
            "expert."
        ),
    },

    "rust_level_4": {
        "display_name": "Coffee Leaf Rust",
        "disease_type": "coffee_leaf_rust",
        "severity": "Level 4",
        "recommendation": (
            "Inspect nearby leaves and plants for similar "
            "symptoms and consider consulting an agricultural "
            "expert."
        ),
    },
}


# ============================================================
# BASE64 ENCODING
# ============================================================

def _to_base64(
    image_bytes: bytes,
) -> str:
    """
    Convert PNG bytes to Base64 for frontend display.
    """

    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def _normalize_labels(
    labels: dict,
) -> dict[int, str]:
    """
    Convert labels loaded from labels.json into:

        {
            0: "healthy",
            1: "red_spider_mite",
            ...
        }

    JSON object keys are always strings, while PyTorch class
    indices are integers. This function makes the mapping
    consistent for model inference.
    """

    if not isinstance(labels, dict):
        raise RuntimeError(
            "Model labels must be a JSON object/dictionary."
        )

    normalized: dict[int, str] = {}

    for key, value in labels.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid model label index: {key!r}"
            ) from exc

        label = str(value).strip().lower()

        normalized[index] = label

    return normalized


# ============================================================
# LABEL VALIDATION
# ============================================================

def _validate_labels(
    labels: dict[int, str],
) -> None:
    """
    Validate that the loaded model labels match the expected
    six-class RoCoLe classification setup.
    """

    if not isinstance(labels, dict):
        raise RuntimeError(
            "Model labels are invalid."
        )

    if labels != EXPECTED_LABEL_MAPPING:
        raise RuntimeError(
            "Model labels do not match the expected "
            "6-class RoCoLe mapping. "
            f"Expected={EXPECTED_LABEL_MAPPING}, "
            f"Received={labels}"
        )


# ============================================================
# VISUAL EVIDENCE
# ============================================================

def _build_visual_evidence(
    raw_class: str,
    affected_percentage: float,
) -> list[str]:
    """
    Generate cautious visual evidence statements.

    The color-based affected-area detector is a visual
    heuristic. It does not independently prove a disease.
    """

    evidence: list[str] = []

    if raw_class.startswith("rust_level_"):

        evidence.append(
            "Brown/rust-colored regions were detected "
            "in the leaf's visually affected area."
        )

    elif raw_class == "red_spider_mite":

        evidence.append(
            "Visible leaf discoloration or affected regions "
            "were detected by the image analysis pipeline."
        )

    elif raw_class == "healthy":

        evidence.append(
            "No major affected color regions were detected "
            "by the visual analysis pipeline."
        )

    if affected_percentage > 0:

        evidence.append(
            "Estimated visually affected leaf area: "
            f"{affected_percentage:.2f}%."
        )

    else:

        evidence.append(
            "No visually affected leaf area was detected "
            "by the color-based analysis."
        )

    return evidence


# ============================================================
# CLASS INTERPRETATION
# ============================================================

def _get_class_info(
    raw_class: str,
) -> dict:
    """
    Convert the raw model class into user-facing information.
    """

    info = CLASS_INFO.get(raw_class)

    if info is not None:
        return info

    logger.warning(
        "Unknown model label received: %s",
        raw_class,
    )

    return {
        "display_name": raw_class.replace(
            "_",
            " ",
        ).title(),

        "disease_type": raw_class,

        "severity": "Unknown",

        "recommendation": (
            "Inspect the leaf and consider consulting "
            "an agricultural expert."
        ),
    }


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(
    image_bytes: bytes,
) -> dict:
    """
    Execute the complete multiclass prediction pipeline.

    Returns:
        Dictionary containing:

        - prediction
        - raw_class
        - confidence
        - disease_type
        - severity
        - recommendation
        - visual_evidence
        - probability breakdown
        - affected area
        - extracted features
        - processed images
        - processing time
    """

    start = time.perf_counter()

    # ========================================================
    # 1. VALIDATE INPUT
    # ========================================================

    if not image_bytes:
        raise InvalidImageError(
            "No image was provided."
        )

    # ========================================================
    # 2. PREPROCESS IMAGE
    # ========================================================

    try:

        tensor = preprocess_image(
            image_bytes
        )

    except InvalidImageError:
        raise

    except Exception as exc:

        logger.exception(
            "Image preprocessing failed."
        )

        raise InvalidImageError(
            f"Unable to preprocess image: {exc}"
        ) from exc

    # ========================================================
    # 3. LOAD MODEL
    # ========================================================

    try:

        model = model_loader.get_model()

        device = model_loader.get_device()

        raw_labels = model_loader.load_labels()

    except Exception as exc:

        logger.exception(
            "Failed to load model or labels."
        )

        raise RuntimeError(
            f"Unable to load prediction model: {exc}"
        ) from exc

    # ========================================================
    # 4. NORMALIZE + VALIDATE LABEL MAPPING
    # ========================================================

    try:

        labels = _normalize_labels(
            raw_labels
        )

        _validate_labels(
            labels
        )

    except Exception as exc:

        logger.exception(
            "Model label validation failed."
        )

        raise RuntimeError(
            f"Invalid model label configuration: {exc}"
        ) from exc

    # ========================================================
    # 5. MOVE INPUT TO DEVICE
    # ========================================================

    try:

        tensor = tensor.to(
            device,
            non_blocking=True,
        )

    except Exception as exc:

        logger.exception(
            "Failed to move input tensor to device."
        )

        raise RuntimeError(
            f"Unable to move image to device: {exc}"
        ) from exc

    logger.info(
        "Prediction started "
        "(device=%s, shape=%s)",
        device,
        tuple(tensor.shape),
    )

    # ========================================================
    # 6. MODEL INFERENCE
    # ========================================================

    try:

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )[0]

    except Exception as exc:

        logger.exception(
            "Model inference failed."
        )

        raise RuntimeError(
            f"Model inference failed: {exc}"
        ) from exc

    # ========================================================
    # 7. VERIFY MODEL OUTPUT
    # ========================================================

    if probabilities.ndim != 1:

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(probabilities.shape)}"
        )

    if len(probabilities) != len(labels):

        raise RuntimeError(
            "Model output class count does not match "
            "the labels configuration. "
            f"Model={len(probabilities)}, "
            f"Labels={len(labels)}"
        )

    # ========================================================
    # 8. TOP PREDICTION
    # ========================================================

    top_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    if top_index not in labels:

        raise RuntimeError(
            "Model returned unknown class index: "
            f"{top_index}"
        )

    raw_class = (
        labels[top_index]
        .strip()
        .lower()
    )

    if raw_class not in EXPECTED_CLASSES:

        raise RuntimeError(
            "Model returned unexpected class: "
            f"{raw_class}"
        )

    confidence = round(
        float(
            probabilities[
                top_index
            ].item()
        )
        * 100.0,
        1,
    )

    # ========================================================
    # 9. PROBABILITY BREAKDOWN
    # ========================================================

    probability_breakdown = {
        labels[index]: round(
            float(
                probability.item()
            )
            * 100.0,
            2,
        )
        for index, probability
        in enumerate(probabilities)
    }

    # ========================================================
    # 10. CLASS INTERPRETATION
    # ========================================================

    info = _get_class_info(
        raw_class
    )

    prediction_name = info[
        "display_name"
    ]

    disease_type = info[
        "disease_type"
    ]

    severity = info[
        "severity"
    ]

    recommendation = info[
        "recommendation"
    ]

    # ========================================================
    # 11. LEAF SEGMENTATION
    # ========================================================

    try:

        (
            original,
            segmented,
            affected,
            leaf_mask,
            affected_mask,
            area_stats,
        ) = segment_leaf(
            image_bytes
        )

    except InvalidImageError:
        raise

    except Exception as exc:

        logger.exception(
            "Leaf segmentation failed."
        )

        raise RuntimeError(
            f"Leaf segmentation failed: {exc}"
        ) from exc

    # ========================================================
    # 12. AFFECTED AREA
    # ========================================================

    affected_pixels = int(
        area_stats.get(
            "affected_pixels",
            0,
        )
    )

    leaf_pixels = int(
        area_stats.get(
            "leaf_pixels",
            0,
        )
    )

    affected_percentage = float(
        area_stats.get(
            "percentage",
            0.0,
        )
    )

    # Safety clamp.
    affected_percentage = max(
        0.0,
        min(
            affected_percentage,
            100.0,
        ),
    )

    # Make sure affected pixels can never exceed
    # detected leaf pixels.
    affected_pixels = max(
        0,
        min(
            affected_pixels,
            leaf_pixels,
        ),
    )

    # ========================================================
    # 13. VISUAL EVIDENCE
    # ========================================================

    visual_evidence = _build_visual_evidence(
        raw_class,
        affected_percentage,
    )

    # ========================================================
    # 14. FEATURE EXTRACTION
    # ========================================================

    try:

        features = extract_features(
            segmented,
            leaf_mask,
        )

    except Exception as exc:

        logger.exception(
            "Feature extraction failed."
        )

        # Feature extraction is supplementary to the model
        # prediction. Do not break the entire prediction
        # response if feature extraction fails.
        features = {
            "error": (
                f"Feature extraction failed: {exc}"
            )
        }

    # ========================================================
    # 15. ENCODE PROCESSED IMAGES
    # ========================================================

    try:

        segmented_png = encode_png(
            segmented
        )

        affected_png = encode_png(
            affected
        )

        segmented_b64 = _to_base64(
            segmented_png
        )

        affected_b64 = _to_base64(
            affected_png
        )

    except Exception as exc:

        logger.exception(
            "Failed to encode processed images."
        )

        raise RuntimeError(
            f"Failed to encode processed images: {exc}"
        ) from exc

    # ========================================================
    # 16. PROCESSING TIME
    # ========================================================

    elapsed_ms = round(
        (
            time.perf_counter()
            - start
        )
        * 1000
    )

    logger.info(
        "Prediction completed: "
        "%s [%s] (%.1f%%) in %d ms",
        prediction_name,
        raw_class,
        confidence,
        elapsed_ms,
    )

    # ========================================================
    # 17. FINAL RESPONSE
    # ========================================================

    return {

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        "prediction": prediction_name,

        "raw_class": raw_class,

        "confidence": confidence,

        "disease_type": disease_type,

        "severity": severity,

        # ----------------------------------------------------
        # Human-readable information
        # ----------------------------------------------------

        "visual_evidence": visual_evidence,

        "recommendation": recommendation,

        # ----------------------------------------------------
        # Processing information
        # ----------------------------------------------------

        "processing_time_ms": elapsed_ms,

        # ----------------------------------------------------
        # Model probabilities
        # ----------------------------------------------------

        "probabilities": probability_breakdown,

        # ----------------------------------------------------
        # Affected area
        # ----------------------------------------------------

        "affected_area": {

            "pixels": affected_pixels,

            "leaf_pixels": leaf_pixels,

            "percentage": round(
                affected_percentage,
                2,
            ),
        },

        # ----------------------------------------------------
        # Extracted CV features
        # ----------------------------------------------------

        "features": features,

        # ----------------------------------------------------
        # Processed images
        # ----------------------------------------------------

        "images": {

            "segmented": segmented_b64,

            "affected": affected_b64,
        },
    }