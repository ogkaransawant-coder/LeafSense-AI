"""
/analyze route.

Flow:
    upload
        -> validate
        -> save upload
        -> EfficientNet prediction
        -> OpenCV analysis
        -> Gemini visual analysis
        -> return combined JSON

Important architecture:
    EfficientNet-B0 = official disease classifier
    OpenCV           = segmentation + affected-area estimation
    Gemini Vision    = visual explanation / observations

Gemini does NOT replace the EfficientNet prediction and does NOT
calculate the official affected-area percentage.

Status codes:
    200 - Success
    400 - Invalid or unsupported image
    500 - Model unavailable or internal server error
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.services.gemini_analysis import analyze_leaf_image
from app.services.prediction import run_prediction
from app.utils.exceptions import (
    InvalidImageError,
    ModelNotAvailableError,
)


# =========================================================
# Logger
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# Router
# =========================================================

router = APIRouter(
    tags=["analysis"]
)


# =========================================================
# Upload configuration
# =========================================================

# Store uploaded files for debugging / audit.
UPLOAD_DIR = (
    Path(__file__).resolve().parent.parent.parent / "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Supported image formats
# =========================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# =========================================================
# Maximum upload size
# =========================================================

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# =========================================================
# Gemini MIME helper
# =========================================================

def _get_gemini_mime_type(
    content_type: Optional[str],
) -> str:
    """
    Return a safe MIME type for Gemini.

    The upload validation above already ensures that only
    JPEG, PNG, and WEBP files reach this function.
    """

    if content_type in ALLOWED_CONTENT_TYPES:
        return content_type

    return "image/jpeg"


# =========================================================
# Extract affected-area percentage
# =========================================================

def _extract_affected_percentage(
    result: Dict[str, Any],
) -> Optional[float]:
    """
    Extract the affected-area percentage from the existing
    OpenCV/prediction result.

    This value is ONLY passed to Gemini as context.

    Gemini does NOT calculate or modify this value.
    """

    affected_area = result.get(
        "affected_area"
    )

    if not isinstance(
        affected_area,
        dict,
    ):
        return None

    percentage = affected_area.get(
        "percentage"
    )

    if percentage is None:
        return None

    try:
        return float(percentage)

    except (
        TypeError,
        ValueError,
    ):
        logger.warning(
            "Could not convert affected-area percentage "
            "to float: %r",
            percentage,
        )

        return None


# =========================================================
# Extract visual evidence
# =========================================================

def _extract_visual_evidence(
    result: Dict[str, Any],
) -> list[str]:
    """
    Extract existing visual evidence from the prediction result.

    The exact prediction service structure can vary slightly,
    so this function safely handles missing/non-list values.
    """

    visual_evidence = result.get(
        "visual_evidence"
    )

    if not isinstance(
        visual_evidence,
        list,
    ):
        return []

    cleaned: list[str] = []

    for item in visual_evidence:

        if item is None:
            continue

        text = str(item).strip()

        if text:
            cleaned.append(text)

    return cleaned[:10]


# =========================================================
# POST /analyze
# =========================================================

@router.post("/analyze")
async def analyze_leaf(
    file: UploadFile = File(...),
):
    """
    Analyze a coffee leaf image.

    Pipeline:

        1. Validate uploaded image.
        2. Check file size.
        3. Save image for debugging/audit.
        4. Run EfficientNet + OpenCV prediction pipeline.
        5. Send the original image to Gemini Vision.
        6. Add Gemini analysis to the prediction response.
        7. Return the combined result.

    EfficientNet remains responsible for:

        - healthy
        - red_spider_mite
        - rust_level_1
        - rust_level_2
        - rust_level_3
        - rust_level_4

    OpenCV remains responsible for:

        - leaf segmentation
        - affected-region estimation
        - affected-area percentage

    Gemini provides:

        - visual summary
        - visible symptoms
        - color observations
        - lesion observations
        - consistency note
        - monitoring points

    Gemini failure does NOT make the main image-analysis
    pipeline fail.
    """

    # =====================================================
    # 1. Validate filename
    # =====================================================

    filename = (
        Path(file.filename).name
        if file.filename
        else "unknown"
    )

    logger.info(
        "Received analysis request: filename=%s, content_type=%s",
        filename,
        file.content_type,
    )

    # =====================================================
    # 2. Validate content type
    # =====================================================

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        logger.warning(
            "Unsupported image type: filename=%s, content_type=%s",
            filename,
            file.content_type,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type "
                f"'{file.content_type}'. "
                "Upload a JPG, PNG, or WEBP image."
            ),
        )

    # =====================================================
    # 3. Read uploaded file
    # =====================================================

    try:

        contents = await file.read()

    except Exception as exc:

        logger.exception(
            "Failed to read uploaded file: %s",
            filename,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to read uploaded image.",
            },
        )

    # =====================================================
    # 4. Check empty file
    # =====================================================

    if not contents:

        logger.warning(
            "Empty uploaded file: %s",
            filename,
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # =====================================================
    # 5. Validate file size
    # =====================================================

    file_size = len(contents)

    logger.info(
        "Uploaded file size: %d bytes",
        file_size,
    )

    if file_size > MAX_FILE_SIZE_BYTES:

        logger.warning(
            "File too large: filename=%s, size=%d bytes",
            filename,
            file_size,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "File too large. "
                "Maximum allowed size is 10 MB."
            ),
        )

    # =====================================================
    # 6. Save uploaded image
    # =====================================================

    try:

        destination = (
            UPLOAD_DIR
            / f"{uuid.uuid4().hex}_{filename}"
        )

        destination.write_bytes(
            contents
        )

        logger.info(
            "Saved uploaded image to: %s",
            destination,
        )

    except OSError as exc:

        # Saving is only for debugging/audit.
        # Prediction can continue if storage fails.

        logger.warning(
            "Could not save uploaded image '%s': %s",
            filename,
            exc,
        )

    # =====================================================
    # 7. Run EfficientNet + OpenCV prediction
    # =====================================================

    try:

        logger.info(
            "Starting prediction for: %s",
            filename,
        )

        # run_prediction() is synchronous and performs
        # CPU/GPU/image-processing work.
        #
        # Running it in a thread prevents the async FastAPI
        # event loop from being blocked.

        result = await run_in_threadpool(
            run_prediction,
            image_bytes=contents,
        )

        logger.info(
            "EfficientNet/OpenCV analysis completed: %s",
            filename,
        )

    # =====================================================
    # 8. Invalid image
    # =====================================================

    except InvalidImageError as exc:

        logger.warning(
            "Invalid image rejected: filename=%s error=%s",
            filename,
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # =====================================================
    # 9. Model unavailable
    # =====================================================

    except ModelNotAvailableError as exc:

        logger.error(
            "Model unavailable while analyzing '%s': %s",
            filename,
            exc,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": "ModelNotAvailableError",
            },
        )

    # =====================================================
    # 10. Unexpected prediction error
    # =====================================================

    except Exception as exc:

        logger.exception(
            "Unexpected error during prediction: "
            "filename=%s",
            filename,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": (
                    "Internal server error during "
                    "image analysis."
                ),
                "type": type(exc).__name__,
                "detail": str(exc),
            },
        )

    # =====================================================
    # 11. Gemini Vision analysis
    # =====================================================

    try:

        logger.info(
            "Starting Gemini visual analysis: %s",
            filename,
        )

        # -------------------------------------------------
        # Existing EfficientNet prediction
        # -------------------------------------------------

        prediction = result.get(
            "prediction"
        )

        confidence = result.get(
            "confidence"
        )

        # -------------------------------------------------
        # Existing OpenCV affected-area percentage
        # -------------------------------------------------

        affected_percentage = (
            _extract_affected_percentage(
                result
            )
        )

        # -------------------------------------------------
        # Existing visual evidence
        # -------------------------------------------------

        visual_evidence = (
            _extract_visual_evidence(
                result
            )
        )

        # -------------------------------------------------
        # Correct MIME type for Gemini
        # -------------------------------------------------

        gemini_mime_type = (
            _get_gemini_mime_type(
                file.content_type
            )
        )

        # -------------------------------------------------
        # Run Gemini in a thread.
        #
        # The Gemini SDK call is synchronous, so don't
        # block FastAPI's async event loop.
        # -------------------------------------------------

        gemini_result = await run_in_threadpool(
            analyze_leaf_image,
            image_bytes=contents,
            mime_type=gemini_mime_type,
            prediction=prediction,
            confidence=confidence,
            affected_percentage=affected_percentage,
            visual_evidence=visual_evidence,
        )

        # -------------------------------------------------
        # Add Gemini result to the existing response.
        # -------------------------------------------------

        result["gemini_analysis"] = gemini_result

        logger.info(
            "Gemini visual analysis completed: "
            "filename=%s available=%s",
            filename,
            gemini_result.get(
                "available",
                False,
            ),
        )

    # =====================================================
    # 12. Gemini unexpected error
    # =====================================================

    except Exception as exc:

        # IMPORTANT:
        #
        # Gemini is an additional analysis layer.
        #
        # If Gemini fails, DO NOT fail the entire
        # EfficientNet/OpenCV image analysis.
        #
        # The user should still receive:
        #
        #   prediction
        #   confidence
        #   affected area
        #   features
        #   processed images
        #
        # with Gemini marked unavailable.

        logger.exception(
            "Unexpected Gemini analysis error: "
            "filename=%s",
            filename,
        )

        result["gemini_analysis"] = {
            "available": False,
            "summary": (
                "Gemini visual analysis is currently "
                "unavailable. The main image analysis "
                "was completed successfully."
            ),
            "visual_observations": [],
            "visible_symptoms": [],
            "color_observations": [],
            "lesion_observations": [],
            "consistency_note": "",
            "monitoring": [],
            "confidence_note": (
                "Gemini was unable to analyze the image."
            ),
            "model": None,
            "error": str(exc),
        }

    # =====================================================
    # 13. Return combined result
    # =====================================================

    logger.info(
        "Leaf analysis completed successfully: %s",
        filename,
    )

    return JSONResponse(
        status_code=200,
        content=result,
    )

    # =====================================================
    # 14. Close uploaded file
    # =====================================================

    # NOTE:
    #
    # FastAPI normally manages the UploadFile lifecycle.
    # The finally block below is intentionally omitted because
    # all processing is already completed before returning.