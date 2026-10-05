"""
/evaluate routes.

POST /evaluate
    Upload a labeled validation dataset (.zip) and return real
    Accuracy, Precision, Recall, F1-score and Confusion Matrix.

GET /evaluate/latest
    Return the most recently cached evaluation result.

All metrics are computed from actual model predictions.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.evaluation import (
    get_latest_evaluation,
    run_evaluation,
)

from app.utils.exceptions import (
    EvaluationDatasetError,
    ModelNotAvailableError,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Router
# ---------------------------------------------------------

router = APIRouter(
    prefix="/evaluate",
    tags=["Evaluation"],
)


# ---------------------------------------------------------
# Allowed upload types
# ---------------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",
}


# ---------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------

@router.post("")
async def evaluate_model(
    file: UploadFile = File(...)
):
    """
    Upload a labeled validation ZIP and run a full
    multiclass model evaluation.

    Expected ZIP structure:

        validation.zip
        ├── healthy/
        ├── red_spider_mite/
        ├── rust_level_1/
        ├── rust_level_2/
        ├── rust_level_3/
        └── rust_level_4/

    The evaluation service performs predictions on all
    images and calculates:

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Per-class metrics
        - Confusion matrix
    """

    # -----------------------------------------------------
    # Validate file type
    # -----------------------------------------------------

    filename = (file.filename or "").lower()

    if not filename.endswith(".zip") and (
        file.content_type not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a .zip file containing one labeled "
                "folder per class."
            ),
        )

    # -----------------------------------------------------
    # Read uploaded file
    # -----------------------------------------------------

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded ZIP file is empty.",
        )

    logger.info(
        "Evaluation started: %s (%d bytes)",
        file.filename,
        len(contents),
    )

    # -----------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------

    try:
        result = run_evaluation(contents)

        logger.info(
            "Evaluation completed successfully."
        )

        return JSONResponse(
            status_code=200,
            content=result,
        )

    # -----------------------------------------------------
    # Invalid validation dataset
    # -----------------------------------------------------

    except EvaluationDatasetError as exc:
        logger.warning(
            "Invalid evaluation dataset: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # -----------------------------------------------------
    # Model unavailable
    # -----------------------------------------------------

    except ModelNotAvailableError as exc:
        logger.error(
            "Model unavailable during evaluation: %s",
            exc,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
            },
        )

    # -----------------------------------------------------
    # Unexpected error
    # -----------------------------------------------------

    except Exception as exc:  # noqa: BLE001

        logger.exception(
            "Unexpected evaluation error"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": f"Internal evaluation error: {exc}",
            },
        )


# ---------------------------------------------------------
# GET /evaluate/latest
# ---------------------------------------------------------

@router.get("/latest")
def latest_evaluation():
    """
    Return the most recently cached evaluation result.

    If no evaluation has been performed yet, the service
    should return None or an empty result.
    """

    return {
        "result": get_latest_evaluation()
    }