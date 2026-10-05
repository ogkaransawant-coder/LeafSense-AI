"""
/health route.

GET /health
    Returns the current backend and model status.

The endpoint is useful for:
    - Frontend health checks
    - Railway deployment health checks
    - Debugging model loading
    - Monitoring whether /analyze can use the model
"""

from __future__ import annotations

from fastapi import APIRouter

from app.models import model_loader


# =========================================================
# Router
# =========================================================

router = APIRouter(
    tags=["health"]
)


# =========================================================
# GET /health
# =========================================================

@router.get("/health")
def health_check():
    """
    Liveness / readiness-style health check.

    `status`
        Confirms that the FastAPI server is running.

    `model_loaded`
        Confirms whether the multiclass coffee-leaf model
        has successfully been loaded and is available for
        prediction.
    """

    model_loaded = model_loader.is_model_loaded()

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }