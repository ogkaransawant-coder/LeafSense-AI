"""
Loads the trained EfficientNet-B0 multiclass model once
and reuses it for all predictions.

RoCoLe 6-class model:

0 - healthy
1 - red_spider_mite
2 - rust_level_1
3 - rust_level_2
4 - rust_level_3
5 - rust_level_4
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from app.models.model import create_model
from app.utils.exceptions import ModelNotAvailableError


logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

MODELS_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = (
    MODELS_DIR / "coffee_leaf_multiclass_model.pth"
)

LABELS_PATH = (
    MODELS_DIR / "labels.json"
)


# Used by image_processing.py
IMAGE_SIZE = 224


# ============================================================
# EXPECTED CLASSES
# ============================================================

EXPECTED_LABELS = {
    0: "healthy",
    1: "red_spider_mite",
    2: "rust_level_1",
    3: "rust_level_2",
    4: "rust_level_3",
    5: "rust_level_4",
}


# ============================================================
# GLOBAL MODEL STATE
# ============================================================

_model: Optional[nn.Module] = None

_labels: Optional[Dict[int, str]] = None

_device: Optional[torch.device] = None

_lock = threading.Lock()


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    """
    Return the device used for inference.

    Uses CUDA when available, otherwise CPU.
    """

    global _device

    if _device is None:

        _device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            "Using device: %s",
            _device
        )

    return _device


# ============================================================
# LABELS
# ============================================================

def load_labels() -> Dict[int, str]:
    """
    Load class labels from labels.json.

    Expected:

        0 -> healthy
        1 -> red_spider_mite
        2 -> rust_level_1
        3 -> rust_level_2
        4 -> rust_level_3
        5 -> rust_level_4
    """

    global _labels

    if _labels is not None:
        return _labels


    if not LABELS_PATH.exists():

        raise ModelNotAvailableError(
            f"Missing labels file: {LABELS_PATH}"
        )


    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            raw = json.load(f)


    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise ModelNotAvailableError(
            f"Could not read labels file: {LABELS_PATH}"
        ) from exc


    try:

        _labels = {
            int(k): str(v)
            for k, v in raw.items()
        }

    except (
        ValueError,
        TypeError,
    ) as exc:

        raise ModelNotAvailableError(
            "Invalid labels.json format."
        ) from exc


    # --------------------------------------------------------
    # Validate number of classes
    # --------------------------------------------------------

    if len(_labels) != 6:

        raise ModelNotAvailableError(
            "Expected 6 classes in labels.json, "
            f"but found {len(_labels)}: {_labels}"
        )


    # --------------------------------------------------------
    # Validate class mapping
    # --------------------------------------------------------

    normalized_labels = {
        index: label.strip().lower()
        for index, label in _labels.items()
    }

    normalized_expected = {
        index: label.lower()
        for index, label in EXPECTED_LABELS.items()
    }


    if normalized_labels != normalized_expected:

        raise ModelNotAvailableError(
            "labels.json does not match the expected "
            "RoCoLe multiclass mapping.\n"
            f"Expected: {EXPECTED_LABELS}\n"
            f"Found: {_labels}"
        )


    logger.info(
        "Loaded labels: %s",
        list(_labels.values())
    )


    return _labels


# ============================================================
# MODEL
# ============================================================

def get_model() -> nn.Module:
    """
    Load the multiclass EfficientNet-B0 model once.

    Subsequent calls reuse the already-loaded model.
    """

    global _model


    if _model is not None:
        return _model


    with _lock:

        # Double-check after acquiring the lock
        if _model is not None:
            return _model


        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device = get_device()


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = load_labels()


        # ----------------------------------------------------
        # Create six-class model
        # ----------------------------------------------------

        model = create_model(
            num_classes=len(labels)
        )


        # ----------------------------------------------------
        # Check model weights
        # ----------------------------------------------------

        if not WEIGHTS_PATH.exists():

            raise ModelNotAvailableError(
                f"Missing model weights: {WEIGHTS_PATH}"
            )


        logger.info(
            "Loading model weights from: %s",
            WEIGHTS_PATH
        )


        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        try:

            checkpoint = torch.load(
                WEIGHTS_PATH,
                map_location=device
            )

        except Exception as exc:

            raise ModelNotAvailableError(
                f"Could not load model weights: "
                f"{WEIGHTS_PATH}"
            ) from exc


        # ----------------------------------------------------
        # Support different checkpoint formats
        # ----------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ):

            if "model_state_dict" in checkpoint:

                checkpoint = (
                    checkpoint["model_state_dict"]
                )

            elif "state_dict" in checkpoint:

                checkpoint = (
                    checkpoint["state_dict"]
                )


        if not isinstance(
            checkpoint,
            dict
        ):

            raise ModelNotAvailableError(
                "Invalid model checkpoint format."
            )


        # ----------------------------------------------------
        # Remove DataParallel prefix if present
        # ----------------------------------------------------

        checkpoint = {
            (
                key[7:]
                if key.startswith("module.")
                else key
            ): value

            for key, value in checkpoint.items()
        }


        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        try:

            model.load_state_dict(
                checkpoint
            )

        except RuntimeError as exc:

            raise ModelNotAvailableError(
                "Model weights are incompatible with "
                "the current model architecture. "
                "Make sure the 6-class "
                "coffee_leaf_multiclass_model.pth "
                "is being used."
            ) from exc


        # ----------------------------------------------------
        # Move model to device
        # ----------------------------------------------------

        model.to(device)

        model.eval()


        # ----------------------------------------------------
        # Cache model
        # ----------------------------------------------------

        _model = model


        logger.info(
            "6-class EfficientNet-B0 model loaded successfully."
        )

        logger.info(
            "Classes: %s",
            list(labels.values())
        )


        return _model


# ============================================================
# PRELOAD
# ============================================================

def try_preload() -> bool:
    """
    Try loading the model during application startup.

    Returns:
        True  -> model loaded successfully
        False -> model loading failed
    """

    try:

        get_model()

        return True

    except Exception as exc:

        logger.warning(
            "Model preload failed: %s",
            exc
        )

        return False


# ============================================================
# MODEL STATUS
# ============================================================

def is_model_loaded() -> bool:
    """
    Return True if the model is already loaded.
    """

    return _model is not None