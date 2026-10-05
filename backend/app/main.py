"""
LeafSense AI — FastAPI backend entrypoint.

Provides:
- /health
- /analyze
- /evaluate
- /evaluate/latest
- /chat

Optimized for lightweight startup:
- No model preloading at startup.
- Model loads automatically on the first prediction request.
- Environment variables are loaded from backend/.env.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# ENVIRONMENT
# ============================================================

# Load backend/.env before importing routes.
#
# This is important because chat.py and gemini_analysis.py
# read Gemini configuration from environment variables.
#
# Expected file:
#     backend/.env
#
# Example:
#     GEMINI_API_KEY=your_api_key_here

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from app.models import model_loader
from app.routes import analyze, chat, evaluate, health
from app.utils.logger import configure_logging


# ============================================================
# LOGGING
# ============================================================

configure_logging()
logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="LeafSense AI API",
    description=(
        "AI-powered coffee leaf disease detection, "
        "visual analysis, severity estimation, "
        "disease explanation, and AI chat assistant."
    ),
    version="0.3.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:3000",
        "http://127.0.0.1:3000",

        # GitHub Codespaces frontend
        "https://studious-spoon-qv4qqxqgjrw736wqw-3000.app.github.dev",

        # Render frontend
        "https://leafsense-ai-spz7.onrender.com",
    ],

    # Allow GitHub Codespaces preview ports.
    allow_origin_regex=r"https://.*-3000\.app\.github\.dev",

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(evaluate.router)
app.include_router(chat.router)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup() -> None:
    """
    Keep application startup lightweight.

    The model is NOT loaded during startup.
    It will be loaded automatically when the first
    prediction request reaches /analyze.
    """

    logger.info(
        "LeafSense AI backend started successfully."
    )

    logger.info(
        "Environment file: %s",
        ENV_FILE,
    )

    logger.info(
        "Gemini API key configured: %s",
        bool(__import__("os").getenv("GEMINI_API_KEY")),
    )

    logger.info(
        "Model will be loaded on the first prediction request."
    )

    logger.info(
        "AI chat endpoint is available at /chat."
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API information endpoint.
    """

    return {
        "service": "LeafSense AI API",
        "status": "running",
        "model_loaded": model_loader.is_model_loaded(),
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "chat": "/chat",
            "evaluate": "/evaluate",
            "latest_evaluation": "/evaluate/latest",
            "docs": "/docs",
        },
    }