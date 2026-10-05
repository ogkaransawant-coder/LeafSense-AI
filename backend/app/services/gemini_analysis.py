"""
CoffeeLeaf AI - Gemini Vision Analysis Service

Purpose
-------
This service sends the uploaded coffee-leaf image to Google Gemini
for visual analysis.

IMPORTANT ARCHITECTURE RULE
---------------------------
EfficientNet-B0 remains the official disease classifier.

Gemini is used only for:
    - visual observations
    - symptom observations
    - color/pattern observations
    - explanation of the existing prediction
    - monitoring suggestions

Gemini must NOT:
    - replace the EfficientNet prediction
    - change the EfficientNet class
    - calculate the official affected-area percentage
    - invent measurements
    - provide pesticide dosage
    - claim certainty beyond the supplied evidence

The official affected-area percentage continues to come from the
existing OpenCV image-analysis pipeline.

Environment variables
---------------------
GEMINI_API_KEY
GEMINI_ANALYSIS_MODEL
GEMINI_ANALYSIS_FALLBACK_MODEL
GEMINI_TIMEOUT_MS
GEMINI_ANALYSIS_MAX_OUTPUT_TOKENS
GEMINI_ANALYSIS_THINKING_LEVEL
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

DEFAULT_ANALYSIS_MODEL = "gemini-3.5-flash-lite"

DEFAULT_ANALYSIS_FALLBACK_MODEL = "gemini-3.6-flash"


def _get_safe_model(
    environment_name: str,
    default_model: str,
) -> str:
    """
    Read a Gemini model name from the environment.

    Removes an optional 'models/' prefix.

    Also prevents an old unsupported Gemini 2.5 configuration
    from being accidentally reused.
    """

    configured = os.getenv(
        environment_name,
        default_model,
    ).strip()

    # --------------------------------------------------------
    # Protect against the old configuration used previously.
    # --------------------------------------------------------

    if configured in {
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "models/gemini-2.5-flash-lite",
    }:
        logger.warning(
            "%s was configured with an old Gemini model: %s. "
            "Using default model: %s",
            environment_name,
            configured,
            default_model,
        )

        return default_model

    # --------------------------------------------------------
    # Remove optional models/ prefix.
    # --------------------------------------------------------

    if configured.startswith("models/"):
        configured = configured[len("models/") :]

    return configured or default_model


GEMINI_ANALYSIS_MODEL = _get_safe_model(
    "GEMINI_ANALYSIS_MODEL",
    DEFAULT_ANALYSIS_MODEL,
)


GEMINI_ANALYSIS_FALLBACK_MODEL = _get_safe_model(
    "GEMINI_ANALYSIS_FALLBACK_MODEL",
    DEFAULT_ANALYSIS_FALLBACK_MODEL,
)


# ============================================================
# REQUEST SETTINGS
# ============================================================

GEMINI_TIMEOUT_MS = int(
    os.getenv(
        "GEMINI_TIMEOUT_MS",
        "20000",
    )
)


GEMINI_ANALYSIS_MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "GEMINI_ANALYSIS_MAX_OUTPUT_TOKENS",
        "500",
    )
)


GEMINI_ANALYSIS_THINKING_LEVEL = os.getenv(
    "GEMINI_ANALYSIS_THINKING_LEVEL",
    "low",
).strip().lower()


# ============================================================
# SUPPORTED CLASSES
# ============================================================

SUPPORTED_CLASSES = {
    "healthy",
    "red_spider_mite",
    "rust_level_1",
    "rust_level_2",
    "rust_level_3",
    "rust_level_4",
}


# ============================================================
# DEFAULT RESULT
# ============================================================

def _empty_analysis(
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a safe empty Gemini analysis.

    This allows the main image-analysis pipeline to continue
    even when Gemini is unavailable.
    """

    return {
        "available": False,
        "summary": (
            "Gemini visual analysis is currently unavailable. "
            "The EfficientNet classification and OpenCV analysis "
            "remain available."
        ),
        "visual_observations": [],
        "visible_symptoms": [],
        "color_observations": [],
        "lesion_observations": [],
        "consistency_note": "",
        "monitoring": [],
        "confidence_note": (
            "Gemini analysis was not available for this image."
        ),
        "model": None,
        "error": reason,
    }


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_gemini_client():
    """
    Create the Google Gemini client.

    The API key is read only from the backend environment.
    """

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    try:
        from google import genai
        from google.genai import types

    except ImportError as exc:
        raise RuntimeError(
            "The google-genai package is not installed. "
            "Run: pip install -U google-genai"
        ) from exc

    http_options = types.HttpOptions(
        timeout=GEMINI_TIMEOUT_MS,
    )

    client = genai.Client(
        api_key=api_key,
        http_options=http_options,
    )

    return client


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def _normalize_class(
    raw_class: Optional[str],
) -> str:
    """
    Normalize the EfficientNet class label.
    """

    if not raw_class:
        return "unknown"

    normalized = str(raw_class).strip().lower()

    if normalized in SUPPORTED_CLASSES:
        return normalized

    return normalized


# ============================================================
# PROMPT
# ============================================================

def _build_analysis_prompt(
    prediction: Optional[str],
    confidence: Optional[float],
    affected_percentage: Optional[float],
    visual_evidence: Optional[List[str]],
) -> str:
    """
    Build the Gemini Vision prompt.

    The prompt deliberately separates:

        EfficientNet = classification source of truth
        Gemini       = visual explanation
        OpenCV       = affected-area measurement
    """

    normalized_prediction = _normalize_class(
        prediction
    )

    confidence_text = (
        f"{float(confidence):.2f}%"
        if confidence is not None
        else "not provided"
    )

    affected_text = (
        f"{float(affected_percentage):.2f}%"
        if affected_percentage is not None
        else "not provided"
    )

    evidence_text = "None provided."

    if visual_evidence:
        cleaned_evidence = []

        for item in visual_evidence[:10]:
            if item:
                cleaned_evidence.append(
                    str(item).strip()
                )

        if cleaned_evidence:
            evidence_text = "\n".join(
                f"- {item}"
                for item in cleaned_evidence
            )

    return f"""
You are the visual analysis assistant for CoffeeLeaf AI.

You are looking at a coffee leaf image.

IMPORTANT SYSTEM ARCHITECTURE:

1. EfficientNet-B0 is the official disease classifier.
2. The supplied EfficientNet prediction must NOT be changed.
3. Gemini is NOT the disease classifier.
4. OpenCV calculates the official affected-area percentage.
5. Gemini must NOT recalculate or invent the affected-area percentage.
6. Gemini should describe only visual evidence that can reasonably
   be observed in the supplied image.
7. Do not invent symptoms that cannot be visually supported.
8. Do not invent laboratory measurements.
9. Do not provide pesticide dosage.
10. Do not claim that the image alone proves a diagnosis.
11. If something is unclear, explicitly say that it is unclear.
12. Do not override the EfficientNet prediction.

CURRENT EFFICIENTNET RESULT
---------------------------
Predicted class:
{normalized_prediction}

EfficientNet confidence:
{confidence_text}

OpenCV affected-area estimate:
{affected_text}

Existing visual evidence from the application:
{evidence_text}

YOUR TASK
---------
Analyze the uploaded coffee-leaf image visually and return ONLY
valid JSON.

The JSON must contain exactly these fields:

{{
  "summary": "A short visual summary of what is visible in the leaf image.",
  "visual_observations": [
    "Short observation 1",
    "Short observation 2"
  ],
  "visible_symptoms": [
    "Only symptoms that are visibly supported by the image"
  ],
  "color_observations": [
    "Visible color or discoloration observations"
  ],
  "lesion_observations": [
    "Visible lesion, spot, patch, or surface-pattern observations"
  ],
  "consistency_note": "Explain whether the visible appearance appears broadly consistent with the supplied EfficientNet prediction, without changing the prediction.",
  "monitoring": [
    "Practical things the user can visually monitor"
  ],
  "confidence_note": "A short limitation note explaining that image-based visual observations are not a laboratory diagnosis."
}}

RULES FOR THE JSON:

- Use concise sentences.
- Do not use Markdown.
- Do not include ```json.
- Do not include text before or after the JSON.
- Use empty arrays when no reliable observation is visible.
- Do not invent details.
- Do not change the EfficientNet class.
- Do not give a new disease classification.
- Do not estimate affected percentage.
- Do not provide chemical or pesticide dosage.
"""


# ============================================================
# GEMINI RESPONSE EXTRACTION
# ============================================================

def _extract_response_text(
    response: Any,
) -> str:
    """
    Safely extract text from a Gemini response.
    """

    try:
        text = response.text
    except Exception:
        text = None

    if text:
        return str(text).strip()

    return ""


# ============================================================
# JSON CLEANING
# ============================================================

def _clean_json_text(
    text: str,
) -> str:
    """
    Remove accidental Markdown code fences from Gemini output.
    """

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]

    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


# ============================================================
# JSON VALIDATION
# ============================================================

def _parse_analysis_response(
    text: str,
) -> Dict[str, Any]:
    """
    Parse Gemini's JSON response.

    If Gemini returns malformed JSON, raise ValueError so the
    caller can safely fall back instead of returning corrupted data.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned = _clean_json_text(text)

    try:
        parsed = json.loads(cleaned)

    except json.JSONDecodeError as exc:

        logger.warning(
            "Gemini analysis returned invalid JSON: %s",
            exc,
        )

        raise ValueError(
            "Gemini returned invalid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Gemini analysis response must be a JSON object."
        )

    return parsed


# ============================================================
# SAFE LIST
# ============================================================

def _safe_string_list(
    value: Any,
) -> List[str]:
    """
    Convert a Gemini JSON field into a safe list of strings.
    """

    if not isinstance(value, list):
        return []

    result: List[str] = []

    for item in value:

        if item is None:
            continue

        text = str(item).strip()

        if text:
            result.append(text)

    return result[:10]


# ============================================================
# NORMALIZE ANALYSIS
# ============================================================

def _normalize_analysis(
    data: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """
    Normalize Gemini's response into the stable structure used
    by the rest of CoffeeLeaf AI.
    """

    summary = data.get(
        "summary",
        "",
    )

    if not isinstance(summary, str):
        summary = str(summary)

    consistency_note = data.get(
        "consistency_note",
        "",
    )

    if not isinstance(consistency_note, str):
        consistency_note = str(consistency_note)

    confidence_note = data.get(
        "confidence_note",
        "",
    )

    if not isinstance(confidence_note, str):
        confidence_note = str(confidence_note)

    return {
        "available": True,
        "summary": summary.strip(),
        "visual_observations": _safe_string_list(
            data.get("visual_observations")
        ),
        "visible_symptoms": _safe_string_list(
            data.get("visible_symptoms")
        ),
        "color_observations": _safe_string_list(
            data.get("color_observations")
        ),
        "lesion_observations": _safe_string_list(
            data.get("lesion_observations")
        ),
        "consistency_note": consistency_note.strip(),
        "monitoring": _safe_string_list(
            data.get("monitoring")
        ),
        "confidence_note": confidence_note.strip(),
        "model": model_name,
        "error": None,
    }


# ============================================================
# GENERATE WITH GEMINI
# ============================================================

def _generate_analysis(
    client: Any,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> Dict[str, Any]:
    """
    Send the image + prompt to Gemini.

    The Gemini Python SDK is imported here so that the rest of
    the backend can still start even if the package is missing.
    """

    from google.genai import types

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    config_kwargs: Dict[str, Any] = {
        "system_instruction": (
            "You are CoffeeLeaf AI's visual analysis assistant. "
            "Follow the supplied EfficientNet prediction and "
            "describe visual evidence only. Never replace the "
            "classifier prediction."
        ),
        "temperature": 0.2,
        "max_output_tokens": GEMINI_ANALYSIS_MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
    }

    # --------------------------------------------------------
    # Thinking configuration.
    #
    # Some Gemini models support thinking configuration.
    # If a model/version does not accept it, retry the request
    # without thinking_config.
    # --------------------------------------------------------

    thinking_level = GEMINI_ANALYSIS_THINKING_LEVEL

    if thinking_level:
        try:
            config_kwargs["thinking_config"] = (
                types.ThinkingConfig(
                    thinking_level=thinking_level,
                )
            )
        except Exception:
            logger.debug(
                "ThinkingConfig is unavailable for model %s.",
                model_name,
            )

    try:

        config = types.GenerateContentConfig(
            **config_kwargs
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                image_part,
            ],
            config=config,
        )

    except Exception as first_error:

        # ----------------------------------------------------
        # Retry once without thinking configuration.
        #
        # This protects the image-analysis service from SDK/model
        # differences without adding a long retry loop.
        # ----------------------------------------------------

        if "thinking_config" not in config_kwargs:
            raise

        logger.warning(
            "Gemini analysis request failed with thinking "
            "configuration. Retrying without it. "
            "model=%s error=%s",
            model_name,
            first_error,
        )

        config_kwargs.pop(
            "thinking_config",
            None,
        )

        config = types.GenerateContentConfig(
            **config_kwargs
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                image_part,
            ],
            config=config,
        )

    text = _extract_response_text(
        response
    )

    parsed = _parse_analysis_response(
        text
    )

    return _normalize_analysis(
        parsed,
        model_name,
    )


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def analyze_leaf_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    affected_percentage: Optional[float] = None,
    visual_evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze a coffee-leaf image using Gemini Vision.

    Parameters
    ----------
    image_bytes:
        Original uploaded image bytes.

    mime_type:
        Image MIME type, for example:
            image/jpeg
            image/png
            image/webp

    prediction:
        Existing EfficientNet prediction.

    confidence:
        Existing EfficientNet confidence.

    affected_percentage:
        Existing OpenCV affected-area percentage.

        IMPORTANT:
        This value is provided to Gemini as context only.
        Gemini does NOT calculate or modify it.

    visual_evidence:
        Existing evidence generated by the CoffeeLeaf AI
        pipeline.

    Returns
    -------
    dict
        Stable Gemini analysis object.

    Failure behavior
    ----------------
    Gemini failures do NOT raise an exception here.

    Instead, a safe result with:

        available = False

    is returned.

    This is intentional because Gemini is an additional AI
    analysis layer. The core EfficientNet/OpenCV pipeline should
    continue working even when Gemini is temporarily unavailable.
    """

    # --------------------------------------------------------
    # Basic image validation.
    # --------------------------------------------------------

    if not image_bytes:
        logger.warning(
            "Gemini analysis skipped because image bytes are empty."
        )

        return _empty_analysis(
            "No image data was provided."
        )

    # --------------------------------------------------------
    # Normalize MIME type.
    # --------------------------------------------------------

    normalized_mime = (
        str(mime_type or "image/jpeg")
        .strip()
        .lower()
    )

    allowed_mime_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if normalized_mime not in allowed_mime_types:
        logger.warning(
            "Unsupported MIME type for Gemini analysis: %s",
            normalized_mime,
        )

        return _empty_analysis(
            f"Unsupported image type: {normalized_mime}"
        )

    # --------------------------------------------------------
    # Build prompt.
    # --------------------------------------------------------

    prompt = _build_analysis_prompt(
        prediction=prediction,
        confidence=confidence,
        affected_percentage=affected_percentage,
        visual_evidence=visual_evidence,
    )

    # --------------------------------------------------------
    # Create Gemini client.
    # --------------------------------------------------------

    try:

        client = _get_gemini_client()

    except Exception as exc:

        logger.error(
            "Unable to create Gemini analysis client: %s",
            exc,
        )

        return _empty_analysis(
            str(exc)
        )

    # --------------------------------------------------------
    # Try primary model.
    # --------------------------------------------------------

    try:

        logger.info(
            "Starting Gemini visual analysis. model=%s",
            GEMINI_ANALYSIS_MODEL,
        )

        result = _generate_analysis(
            client=client,
            model_name=GEMINI_ANALYSIS_MODEL,
            image_bytes=image_bytes,
            mime_type=normalized_mime,
            prompt=prompt,
        )

        logger.info(
            "Gemini visual analysis completed successfully. "
            "model=%s",
            GEMINI_ANALYSIS_MODEL,
        )

        return result

    except Exception as primary_error:

        logger.warning(
            "Primary Gemini visual analysis failed. "
            "model=%s error=%s",
            GEMINI_ANALYSIS_MODEL,
            primary_error,
        )

    # --------------------------------------------------------
    # Try fallback model.
    # --------------------------------------------------------

    if (
        GEMINI_ANALYSIS_FALLBACK_MODEL
        and GEMINI_ANALYSIS_FALLBACK_MODEL
        != GEMINI_ANALYSIS_MODEL
    ):

        try:

            logger.info(
                "Trying fallback Gemini visual analysis. "
                "model=%s",
                GEMINI_ANALYSIS_FALLBACK_MODEL,
            )

            result = _generate_analysis(
                client=client,
                model_name=GEMINI_ANALYSIS_FALLBACK_MODEL,
                image_bytes=image_bytes,
                mime_type=normalized_mime,
                prompt=prompt,
            )

            logger.info(
                "Fallback Gemini visual analysis completed. "
                "model=%s",
                GEMINI_ANALYSIS_FALLBACK_MODEL,
            )

            return result

        except Exception as fallback_error:

            logger.error(
                "Fallback Gemini visual analysis failed. "
                "model=%s error=%s",
                GEMINI_ANALYSIS_FALLBACK_MODEL,
                fallback_error,
            )

            return _empty_analysis(
                "Gemini visual analysis failed."
            )

    # --------------------------------------------------------
    # No fallback configured.
    # --------------------------------------------------------

    return _empty_analysis(
        "Gemini visual analysis failed."
    )


# ============================================================
# OPTIONAL HEALTH CHECK
# ============================================================

def is_gemini_analysis_configured() -> bool:
    """
    Return True when a Gemini API key is configured.

    This does NOT make a network request.
    """

    return bool(
        os.getenv(
            "GEMINI_API_KEY",
            "",
        ).strip()
    )