"""
CoffeeLeaf AI - Chat Route

This route handles conversational questions about an existing
coffee-leaf diagnosis.

Important architecture rules:

1. EfficientNet is the classification source of truth.
2. Gemini does NOT re-classify the leaf.
3. Gemini image analysis is supporting visual evidence only.
4. OpenCV/application affected-area percentage remains authoritative.
5. Disease-specific information comes from disease_knowledge.py.
6. If Gemini is unavailable, the application returns a local fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.disease_knowledge import get_disease_knowledge


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

DEFAULT_PRIMARY_MODEL = "gemini-3.5-flash-lite"
DEFAULT_FALLBACK_MODEL = "gemini-3.6-flash"

FORBIDDEN_MODELS = {
    "gemini-2.5-flash",
    "models/gemini-2.5-flash",
}


def _get_safe_model(
    environment_name: str,
    default_model: str,
) -> str:
    """
    Read a Gemini model from environment variables.

    Old/unsupported configuration values are replaced with
    the current configured default.
    """

    configured = os.getenv(
        environment_name,
        default_model,
    ).strip()

    if configured.startswith("models/"):
        configured = configured[len("models/"):]

    if configured in FORBIDDEN_MODELS:
        logger.warning(
            "%s was configured with forbidden model '%s'. "
            "Using '%s' instead.",
            environment_name,
            configured,
            default_model,
        )
        return default_model

    return configured or default_model


GEMINI_MODEL = _get_safe_model(
    "GEMINI_MODEL",
    DEFAULT_PRIMARY_MODEL,
)

GEMINI_FALLBACK_MODEL = _get_safe_model(
    "GEMINI_FALLBACK_MODEL",
    DEFAULT_FALLBACK_MODEL,
)


# ============================================================
# GEMINI REQUEST SETTINGS
# ============================================================

GEMINI_TIMEOUT_MS = int(
    os.getenv(
        "GEMINI_TIMEOUT_MS",
        "20000",
    )
)

GEMINI_MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "GEMINI_MAX_OUTPUT_TOKENS",
        "700",
    )
)

GEMINI_THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "low",
).strip().lower()


# ============================================================
# REQUEST LIMITS
# ============================================================

MAX_QUESTION_LENGTH = 2000
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_LENGTH = 1200

MAX_VISUAL_OBSERVATIONS = 12
MAX_VISUAL_OBSERVATION_LENGTH = 500


# ============================================================
# SUPPORTED ROCOLE CLASSES
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
# REQUEST MODELS
# ============================================================


class GeminiAnalysisContext(BaseModel):
    """
    Optional visual analysis produced by the Gemini image-analysis
    service.

    This information is supporting evidence only.

    It must never override the EfficientNet classification.
    """

    available: bool = False

    summary: Optional[str] = None

    visual_observations: Optional[List[str]] = None

    visible_symptoms: Optional[List[str]] = None

    color_observations: Optional[List[str]] = None

    lesion_observations: Optional[List[str]] = None

    consistency_note: Optional[str] = None

    monitoring: Optional[List[str]] = None

    confidence_note: Optional[str] = None

    model: Optional[str] = None

    error: Optional[str] = None


class DiagnosisContext(BaseModel):
    """
    Diagnosis produced by the image-analysis pipeline.
    """

    prediction: Optional[str] = None

    raw_class: str = Field(
        ...,
        description=(
            "Raw six-class model label, for example "
            "'rust_level_2'."
        ),
    )

    confidence: Optional[float] = None

    disease_type: Optional[str] = None

    severity: Optional[str] = None

    affected_area: Optional[Dict[str, Any]] = None

    visual_evidence: Optional[List[str]] = None

    recommendation: Optional[str] = None

    gemini_analysis: Optional[GeminiAnalysisContext] = None


class ChatHistoryMessage(BaseModel):
    """
    Previous chat message.
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_HISTORY_MESSAGE_LENGTH,
    )


class ChatRequest(BaseModel):
    """
    Request body for POST /chat.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
    )

    diagnosis: DiagnosisContext

    history: List[ChatHistoryMessage] = Field(
        default_factory=list,
    )


# ============================================================
# RESPONSE MODEL
# ============================================================


class ChatResponse(BaseModel):
    """
    Response returned to the frontend.
    """

    answer: str

    raw_class: str

    prediction: Optional[str] = None

    confidence: Optional[float] = None

    severity: Optional[str] = None

    source: str = "gemini"

    model: Optional[str] = None


# ============================================================
# CLASS VALIDATION
# ============================================================


def _normalize_class(raw_class: str) -> str:
    """
    Normalize and validate the six supported RoCoLe classes.
    """

    normalized = str(raw_class).strip().lower()

    if normalized not in SUPPORTED_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported diagnosis class: '{raw_class}'. "
                "Expected one of the six supported RoCoLe classes."
            ),
        )

    return normalized


# ============================================================
# QUESTION CLEANING
# ============================================================


def _clean_question(question: str) -> str:
    """
    Clean and validate the user's question.
    """

    cleaned = str(question).strip()

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Question is too long. "
                f"Maximum length is {MAX_QUESTION_LENGTH} characters."
            ),
        )

    return cleaned


# ============================================================
# AFFECTED AREA
# ============================================================


def _get_affected_percentage(
    affected_area: Optional[Dict[str, Any]],
) -> Optional[float]:
    """
    Safely extract the application's affected-area percentage.
    """

    if not affected_area:
        return None

    value = affected_area.get("percentage")

    if value is None:
        return None

    try:
        percentage = float(value)
    except (TypeError, ValueError):
        return None

    return max(
        0.0,
        min(
            percentage,
            100.0,
        ),
    )


# ============================================================
# STRING HELPERS
# ============================================================


def _clean_string_list(
    values: Optional[List[str]],
    max_items: int = MAX_VISUAL_OBSERVATIONS,
) -> List[str]:
    """
    Safely clean a list of strings.
    """

    if not isinstance(values, list):
        return []

    cleaned: List[str] = []

    for item in values:
        if not isinstance(item, str):
            continue

        value = item.strip()

        if not value:
            continue

        cleaned.append(
            value[:MAX_VISUAL_OBSERVATION_LENGTH]
        )

        if len(cleaned) >= max_items:
            break

    return cleaned


# ============================================================
# GEMINI IMAGE-ANALYSIS CONTEXT
# ============================================================


def _build_gemini_analysis_context(
    analysis: Optional[GeminiAnalysisContext],
) -> str:
    """
    Convert Gemini image-analysis output into safe structured text.
    """

    if analysis is None:
        return (
            "GEMINI IMAGE ANALYSIS\n"
            "Status: Not available."
        )

    if not analysis.available:
        return (
            "GEMINI IMAGE ANALYSIS\n"
            "Status: Not available for this analysis."
        )

    visual_observations = _clean_string_list(
        analysis.visual_observations
    )

    visible_symptoms = _clean_string_list(
        analysis.visible_symptoms
    )

    color_observations = _clean_string_list(
        analysis.color_observations
    )

    lesion_observations = _clean_string_list(
        analysis.lesion_observations
    )

    monitoring = _clean_string_list(
        analysis.monitoring
    )

    visual_text = (
        "\n".join(
            f"- {item}"
            for item in visual_observations
        )
        if visual_observations
        else "- None provided."
    )

    symptoms_text = (
        "\n".join(
            f"- {item}"
            for item in visible_symptoms
        )
        if visible_symptoms
        else "- None provided."
    )

    colors_text = (
        "\n".join(
            f"- {item}"
            for item in color_observations
        )
        if color_observations
        else "- None provided."
    )

    lesions_text = (
        "\n".join(
            f"- {item}"
            for item in lesion_observations
        )
        if lesion_observations
        else "- None provided."
    )

    monitoring_text = (
        "\n".join(
            f"- {item}"
            for item in monitoring
        )
        if monitoring
        else "- None provided."
    )

    return f"""
GEMINI IMAGE ANALYSIS

Status:
Available

Analysis model:
{analysis.model or "Not provided"}

Summary:
{analysis.summary or "Not provided"}

Visual observations:
{visual_text}

Visible symptoms:
{symptoms_text}

Color observations:
{colors_text}

Lesion observations:
{lesions_text}

Consistency note:
{analysis.consistency_note or "Not provided"}

Monitoring observations:
{monitoring_text}

Confidence note:
{analysis.confidence_note or "Not provided"}
""".strip()


# ============================================================
# DISEASE KNOWLEDGE
# ============================================================


def _load_knowledge(
    raw_class: str,
) -> Dict[str, Any]:
    """
    Load disease knowledge for the current model class.
    """

    try:
        knowledge = get_disease_knowledge(raw_class)

    except KeyError as exc:
        logger.warning(
            "Disease knowledge not found for class=%s",
            raw_class,
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"No disease knowledge is available "
                f"for class '{raw_class}'."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Disease knowledge lookup failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load disease knowledge.",
        ) from exc

    if not isinstance(knowledge, dict):
        raise HTTPException(
            status_code=500,
            detail="Disease knowledge has an invalid format.",
        )

    return knowledge


# ============================================================
# DIAGNOSIS CONTEXT
# ============================================================


def _build_diagnosis_context(
    diagnosis: DiagnosisContext,
    knowledge: Dict[str, Any],
) -> str:
    """
    Build structured diagnosis context for Gemini.

    EfficientNet remains the classification source of truth.
    """

    affected_percentage = _get_affected_percentage(
        diagnosis.affected_area
    )

    confidence_text = (
        f"{diagnosis.confidence:.1f}%"
        if diagnosis.confidence is not None
        else "Not provided"
    )

    affected_text = (
        f"{affected_percentage:.2f}%"
        if affected_percentage is not None
        else "Not provided"
    )

    symptoms = knowledge.get(
        "symptoms",
        [],
    )

    monitoring_points = knowledge.get(
        "what_to_monitor",
        [],
    )

    if not isinstance(symptoms, list):
        symptoms = []

    if not isinstance(monitoring_points, list):
        monitoring_points = []

    visual_evidence = diagnosis.visual_evidence or []

    visual_evidence = _clean_string_list(
        visual_evidence,
        max_items=12,
    )

    if not visual_evidence:
        visual_evidence = [
            "No additional visual evidence was provided."
        ]

    symptoms_text = "\n".join(
        f"- {item}"
        for item in symptoms[:12]
    )

    monitoring_text = "\n".join(
        f"- {item}"
        for item in monitoring_points[:12]
    )

    evidence_text = "\n".join(
        f"- {item}"
        for item in visual_evidence
    )

    gemini_analysis_text = _build_gemini_analysis_context(
        diagnosis.gemini_analysis
    )

    recommendation = (
        diagnosis.recommendation
        or "Not provided"
    )

    return f"""
COFFEELEAF AI DIAGNOSIS

CLASSIFICATION SOURCE OF TRUTH:
EfficientNet image-classification model

Raw model class:
{diagnosis.raw_class}

Prediction:
{diagnosis.prediction or knowledge.get("title", "Unknown")}

Confidence:
{confidence_text}

Disease type:
{diagnosis.disease_type or "Not provided"}

Severity:
{diagnosis.severity or "Not provided"}

Estimated affected leaf area:
{affected_text}

OpenCV / application visual evidence:
{evidence_text}

Application recommendation:
{recommendation}

{gemini_analysis_text}

DISEASE KNOWLEDGE

Title:
{knowledge.get("title", "")}

Short description:
{knowledge.get("short_description", "")}

What is it?
{knowledge.get("what_is_it", "")}

Known symptoms:
{symptoms_text}

What to monitor:
{monitoring_text}

Monitoring guidance:
{knowledge.get("monitoring", "")}

Prevention:
{knowledge.get("prevention", "")}

When to seek expert help:
{knowledge.get("when_to_seek_help", "")}

Disclaimer:
{knowledge.get("disclaimer", "")}
""".strip()


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================


def _build_system_instructions(
    diagnosis_context: str,
) -> str:
    """
    Build the system instructions given to Gemini.
    """

    return f"""
You are CoffeeLeaf AI, a conversational assistant for
coffee-leaf image analysis.

The application has already classified the coffee leaf using
its EfficientNet image-classification model.

Your task is ONLY to explain the existing diagnosis and answer
the user's question.

CLASSIFICATION RULES

1. EfficientNet is the classification source of truth.

2. Do NOT re-classify the leaf.

3. Do NOT change the supplied model prediction.

4. Do NOT replace the EfficientNet prediction with Gemini
   image-analysis observations.

5. Gemini image analysis is supporting visual evidence only.
   It is NOT an independent disease classifier.

6. The OpenCV/application affected-area percentage is the
   application's supplied affected-area estimate.
   Do not calculate a new percentage.

7. Treat the model prediction as an AI-assisted assessment,
   not a guaranteed agricultural diagnosis.

8. Explain the supplied diagnosis in simple language.

9. Use the supplied disease knowledge for disease-specific
   information.

10. Use supplied OpenCV evidence and Gemini image-analysis
    observations when explaining visible characteristics.

11. Do not invent symptoms, observations, measurements,
    treatments, pesticides, chemical doses, or application rates.

12. If the user asks for pesticide or chemical dosage, explain
    that treatment and dosage should be confirmed with a
    qualified agricultural expert and the applicable product
    label/local guidance.

13. The affected-area percentage is an estimate from the
    application's visual-analysis pipeline. It is NOT
    automatically the same thing as disease severity.

14. If confidence is low, explain that the result should be
    interpreted cautiously.

15. If the user asks why the model predicted the disease,
    use only the supplied visual evidence. Do not invent hidden
    neural-network reasoning.

16. If Gemini image analysis disagrees with EfficientNet,
    do NOT change the classification. Explain that EfficientNet
    remains the application's primary prediction.

17. If the prediction is Healthy, clearly explain that the model
    did not detect strong visual evidence of the supported
    disease classes.

RESPONSE QUALITY

18. Answer the user's actual question first.

19. Be concise but useful.

20. Always finish your sentences.

21. Do not output unfinished Markdown.

22. If using bold Markdown, every opening ** must have a matching
    closing **.

23. Prefer plain Markdown headings and short paragraphs.

24. Do not output internal reasoning or hidden model thoughts.

25. Do not claim to know hidden neural-network reasoning.

26. If the user asks "what is the disease?", answer directly
    using the supplied EfficientNet prediction.

27. If the user asks about confidence, use the supplied
    confidence value.

28. If the user asks about affected area, use the supplied
    OpenCV/application percentage.

29. If the user asks a general question about the current
    diagnosis, explain the current diagnosis instead of
    generating a new classification.

CURRENT DIAGNOSIS

{diagnosis_context}
""".strip()


# ============================================================
# MARKDOWN VALIDATION
# ============================================================


def _has_unbalanced_bold_markdown(
    text: str,
) -> bool:
    """
    Detect unbalanced Markdown bold markers.
    """

    if not text:
        return False

    return text.count("**") % 2 != 0


def _looks_truncated(
    text: str,
) -> bool:
    """
    Detect common Gemini truncation patterns.
    """

    if not isinstance(text, str):
        return True

    cleaned = " ".join(
        text.strip().split()
    )

    if not cleaned:
        return True

    if len(cleaned) < 20:
        return True

    normalized = cleaned.lower()

    incomplete_exact = {
        "header/summary:",
        "header/summary",
        "diagnosis summary:",
        "diagnosis summary",
        "the ai model classified",
        "the ai model classified the",
        "the coffeeleaf ai classification model has",
        "the efficientnet image-classification model has",
    }

    if normalized in incomplete_exact:
        return True

    incomplete_endings = (
        " as",
        " as **",
        " classified as",
        " assessed as",
        " assessed the leaf as",
        " classified the leaf as",
        " the leaf as",
        " has assessed the leaf",
        " has classified the leaf",
        " image-classification model has assessed",
        " image-classification model has classified",
        " coffeeleaf ai classification model has assessed",
        " coffeeleaf ai classification model has classified",
        " the",
        " a",
        " an",
        " with",
        " because",
        " and",
        " or",
        " that",
        " which",
        " is",
        " are",
        " was",
        " were",
    )

    if normalized.endswith(incomplete_endings):
        return True

    if _has_unbalanced_bold_markdown(text):
        return True

    incomplete_phrases = (
        "the leaf was classified as",
        "the leaf has been classified as",
        "the model classified the leaf as",
        "the model assessed the leaf as",
        "the ai model classified the leaf as",
        "the ai model assessed the leaf as",
    )

    for phrase in incomplete_phrases:
        if normalized.endswith(phrase):
            return True

    return False


def _validate_gemini_answer(
    answer: str,
) -> bool:
    """
    Validate Gemini's answer before returning it.
    """

    if not isinstance(answer, str):
        return False

    answer = answer.strip()

    if not answer:
        return False

    if _looks_truncated(answer):
        return False

    normalized = " ".join(
        answer.lower().split()
    )

    bad_fragments = (
        "assessed the leaf as **",
        "classified the leaf as **",
        "classification model has assessed the leaf as",
        "classification model has classified the leaf as",
    )

    if any(
        fragment in normalized
        for fragment in bad_fragments
    ):
        return False

    if _has_unbalanced_bold_markdown(answer):
        return False

    return True


# ============================================================
# CONVERSATION HISTORY
# ============================================================


def _is_bad_history_message(
    content: str,
) -> bool:
    """
    Detect malformed previous assistant responses.
    """

    if not content:
        return True

    return _looks_truncated(content)


def _build_gemini_contents(
    history: List[ChatHistoryMessage],
    question: str,
):
    """
    Convert application history into Gemini Content objects.
    """

    try:
        from google.genai import types

    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini dependency is not installed. "
                "Run: pip install -U google-genai"
            ),
        ) from exc

    contents = []

    recent_history = history[-MAX_HISTORY_MESSAGES:]

    for message in recent_history:

        content = message.content.strip()

        if not content:
            continue

        if (
            message.role == "assistant"
            and _is_bad_history_message(content)
        ):
            logger.warning(
                "Skipping malformed assistant history message."
            )
            continue

        role = (
            "model"
            if message.role == "assistant"
            else "user"
        )

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(
                        text=content[
                            :MAX_HISTORY_MESSAGE_LENGTH
                        ]
                    )
                ],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=question
                )
            ],
        )
    )

    return contents


# ============================================================
# GEMINI CLIENT
# ============================================================


def _get_gemini_client():
    """
    Create the Google Gemini client.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI chat is not configured. "
                "Set GEMINI_API_KEY on the backend."
            ),
        )

    try:
        from google import genai
        from google.genai import types

    except ImportError as exc:
        logger.exception(
            "Google GenAI SDK is not installed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini dependency is not installed. "
                "Run: pip install -U google-genai"
            ),
        ) from exc

    try:
        http_options = types.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS
        )

        return genai.Client(
            api_key=api_key,
            http_options=http_options,
        )

    except Exception as exc:
        logger.exception(
            "Could not initialize Gemini client."
        )

        raise HTTPException(
            status_code=503,
            detail="Could not initialize the Gemini AI client.",
        ) from exc


# ============================================================
# GEMINI ERROR HELPERS
# ============================================================


def _get_error_status_code(
    error: Exception,
) -> Optional[int]:
    """
    Extract an HTTP status code from a Gemini exception.
    """

    for attribute in (
        "status_code",
        "code",
    ):
        value = getattr(
            error,
            attribute,
            None,
        )

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass

    return None


def _get_error_text(
    error: Exception,
) -> str:
    """
    Safely convert an exception to text.
    """

    try:
        return str(error).strip()
    except Exception:
        return ""


def _is_timeout_error(
    error: Exception,
) -> bool:
    """
    Detect timeout-like errors.
    """

    error_text = _get_error_text(
        error
    ).lower()

    timeout_words = (
        "timeout",
        "timed out",
        "deadline exceeded",
        "read timeout",
        "connect timeout",
    )

    return any(
        word in error_text
        for word in timeout_words
    )


# ============================================================
# GEMINI MODEL LIST
# ============================================================


def _get_models_to_try() -> List[str]:
    """
    Return primary and fallback Gemini models.

    The obsolete Gemini 2.5 Flash configuration is never used.
    """

    models: List[str] = []

    primary = GEMINI_MODEL.strip()
    fallback = GEMINI_FALLBACK_MODEL.strip()

    if primary.startswith("models/"):
        primary = primary[7:]

    if fallback.startswith("models/"):
        fallback = fallback[7:]

    if primary in FORBIDDEN_MODELS:
        primary = DEFAULT_PRIMARY_MODEL

    if fallback in FORBIDDEN_MODELS:
        fallback = DEFAULT_FALLBACK_MODEL

    if primary:
        models.append(primary)

    if (
        fallback
        and fallback not in models
    ):
        models.append(fallback)

    return models


# ============================================================
# GEMINI GENERATION
# ============================================================


def _generate_gemini_response(
    client: Any,
    contents: Any,
    instructions: str,
):
    """
    Generate a Gemini response.

    Primary model is attempted once.
    Fallback model is attempted once.
    """

    try:
        from google.genai import types

    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini dependency is not installed. "
                "Run: pip install -U google-genai"
            ),
        ) from exc

    models_to_try = _get_models_to_try()

    if not models_to_try:
        raise HTTPException(
            status_code=503,
            detail="No Gemini model is configured.",
        )

    last_error: Optional[Exception] = None

    for index, model_name in enumerate(
        models_to_try
    ):
        try:
            logger.info(
                "Generating Gemini chat response "
                "using model=%s thinking=%s",
                model_name,
                GEMINI_THINKING_LEVEL,
            )

            config_kwargs: Dict[str, Any] = {
                "system_instruction": instructions,
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            }

            # Gemini 3.x supports thinking_level.
            # Keep this configurable so it can be changed through
            # the backend environment without modifying this file.
            if GEMINI_THINKING_LEVEL:
                config_kwargs["thinking_config"] = (
                    types.ThinkingConfig(
                        thinking_level=GEMINI_THINKING_LEVEL
                    )
                )

            generation_config = (
                types.GenerateContentConfig(
                    **config_kwargs
                )
            )

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )

            logger.info(
                "Gemini response generated successfully "
                "using model=%s",
                model_name,
            )

            return response, model_name

        except Exception as exc:
            last_error = exc

            status_code = _get_error_status_code(
                exc
            )

            error_text = _get_error_text(
                exc
            )

            logger.warning(
                "Gemini request failed. "
                "model=%s status=%s error=%s",
                model_name,
                status_code,
                error_text,
            )

            if index < len(models_to_try) - 1:
                logger.warning(
                    "Trying fallback Gemini model: %s",
                    models_to_try[index + 1],
                )
                continue

    if last_error is None:
        raise HTTPException(
            status_code=503,
            detail="The AI assistant is temporarily unavailable.",
        )

    status_code = _get_error_status_code(
        last_error
    )

    error_text = _get_error_text(
        last_error
    )

    logger.error(
        "All Gemini models failed. "
        "primary=%s fallback=%s "
        "last_status=%s error=%s",
        GEMINI_MODEL,
        GEMINI_FALLBACK_MODEL,
        status_code,
        error_text,
    )

    if _is_timeout_error(last_error):
        raise HTTPException(
            status_code=504,
            detail=(
                "The AI assistant took too long to respond. "
                "Please try again."
            ),
        ) from last_error

    if status_code == 429:
        raise HTTPException(
            status_code=429,
            detail=(
                "The AI assistant is temporarily busy. "
                "Please try again in a moment."
            ),
        ) from last_error

    if status_code == 404:
        raise HTTPException(
            status_code=503,
            detail=(
                "The configured Gemini model is not available "
                "for this API account. "
                f"Primary: {GEMINI_MODEL}. "
                f"Fallback: {GEMINI_FALLBACK_MODEL}."
            ),
        ) from last_error

    if status_code in {401, 403}:
        raise HTTPException(
            status_code=503,
            detail=(
                "The Gemini API key is not authorized. "
                "Check GEMINI_API_KEY."
            ),
        ) from last_error

    if status_code in {
        500,
        502,
        503,
        504,
    }:
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI assistant is temporarily unavailable. "
                "Please try again."
            ),
        ) from last_error

    raise HTTPException(
        status_code=502,
        detail=(
            "The AI assistant could not generate "
            "a response right now."
        ),
    ) from last_error


# ============================================================
# RESPONSE TEXT
# ============================================================


def _extract_response_text(
    response: Any,
) -> str:
    """
    Extract generated text from a Gemini response.
    """

    try:
        text = response.text

        if isinstance(text, str):
            text = text.strip()

            if text:
                return text

    except Exception:
        pass

    try:
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if candidates:

            chunks: List[str] = []

            for candidate in candidates:

                content = getattr(
                    candidate,
                    "content",
                    None,
                )

                parts = getattr(
                    content,
                    "parts",
                    None,
                )

                if not parts:
                    continue

                for part in parts:

                    part_text = getattr(
                        part,
                        "text",
                        None,
                    )

                    if isinstance(
                        part_text,
                        str,
                    ):
                        chunks.append(
                            part_text
                        )

            return "\n".join(
                chunks
            ).strip()

    except Exception as exc:
        logger.warning(
            "Could not extract Gemini response parts: %s",
            exc,
        )

    return ""


# ============================================================
# LOCAL FALLBACK
# ============================================================


def _build_local_fallback(
    diagnosis: DiagnosisContext,
    knowledge: Dict[str, Any],
    question: str,
) -> str:
    """
    Generate a useful response without Gemini.

    This keeps the chat functional even when Gemini is
    unavailable.
    """

    prediction = (
        diagnosis.prediction
        or diagnosis.raw_class
        or "Unknown"
    )

    confidence = diagnosis.confidence

    affected = _get_affected_percentage(
        diagnosis.affected_area
    )

    confidence_text = (
        f"{confidence:.1f}%"
        if confidence is not None
        else "not available"
    )

    affected_text = (
        f"{affected:.2f}%"
        if affected is not None
        else "not available"
    )

    symptoms = knowledge.get(
        "symptoms",
        [],
    )

    monitoring = knowledge.get(
        "what_to_monitor",
        [],
    )

    if not isinstance(symptoms, list):
        symptoms = []

    if not isinstance(monitoring, list):
        monitoring = []

    # --------------------------------------------------------
    # HEALTHY
    # --------------------------------------------------------

    if prediction.strip().lower() == "healthy":

        return "\n".join(
            [
                "### Diagnosis Summary",
                "",
                (
                    f"The model classified this coffee leaf "
                    f"as **Healthy** with a confidence of "
                    f"**{confidence_text}**."
                ),
                "",
                (
                    f"The estimated affected area is "
                    f"**{affected_text}**."
                ),
                "",
                "### What this means",
                "",
                (
                    "The model did not detect strong visual "
                    "evidence of the supported coffee-leaf "
                    "disease classes."
                ),
                "",
                "### What to monitor",
                "",
                "- Check the leaf regularly for new spots or lesions.",
                "- Watch for reddish-brown, orange, or unusual discoloration.",
                "- Monitor nearby leaves for similar changes.",
                "- Compare the plant over time rather than relying on a single image.",
                "",
                "### Important",
                "",
                (
                    "This is an AI-assisted visual assessment "
                    "and should be treated as a screening result "
                    "rather than a definitive agricultural diagnosis."
                ),
            ]
        )

    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    answer: List[str] = [
        "### Diagnosis Summary",
        "",
        (
            f"The model classified this coffee leaf as "
            f"**{prediction}** with a confidence of "
            f"**{confidence_text}**."
        ),
        "",
        (
            f"The estimated affected area is "
            f"**{affected_text}**."
        ),
        "",
        "### What this means",
        "",
        str(
            knowledge.get(
                "short_description",
                "The model detected visual patterns associated with this diagnosis.",
            )
        ),
    ]

    if symptoms:

        answer.extend(
            [
                "",
                "### Symptoms",
                "",
            ]
        )

        for symptom in symptoms[:6]:
            answer.append(
                f"- {symptom}"
            )

    if monitoring:

        answer.extend(
            [
                "",
                "### What to monitor",
                "",
            ]
        )

        for item in monitoring[:6]:
            answer.append(
                f"- {item}"
            )

    answer.extend(
        [
            "",
            "### Important",
            "",
            str(
                knowledge.get(
                    "disclaimer",
                    (
                        "This is an AI-assisted visual assessment "
                        "and should not replace expert agricultural diagnosis."
                    ),
                )
            ),
        ]
    )

    return "\n".join(answer)


# ============================================================
# CHAT ENDPOINT
# ============================================================


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat_with_diagnosis(
    request: ChatRequest,
) -> ChatResponse:
    """
    Answer user questions using the existing diagnosis,
    disease knowledge, and supplied visual-analysis evidence.
    """

    # ========================================================
    # 1. Validate question
    # ========================================================

    question = _clean_question(
        request.question
    )

    # ========================================================
    # 2. Validate diagnosis class
    # ========================================================

    raw_class = _normalize_class(
        request.diagnosis.raw_class
    )

    # ========================================================
    # 3. Load disease knowledge
    # ========================================================

    knowledge = _load_knowledge(
        raw_class
    )

    # ========================================================
    # 4. Build diagnosis context
    # ========================================================

    diagnosis_context = _build_diagnosis_context(
        diagnosis=request.diagnosis,
        knowledge=knowledge,
    )

    # ========================================================
    # 5. Build Gemini instructions
    # ========================================================

    instructions = _build_system_instructions(
        diagnosis_context
    )

    # ========================================================
    # 6. Build conversation
    # ========================================================

    contents = _build_gemini_contents(
        history=request.history,
        question=question,
    )

    # ========================================================
    # 7. Get Gemini client
    # ========================================================

    try:
        client = _get_gemini_client()

    except HTTPException as exc:

        logger.warning(
            "Gemini is unavailable. "
            "Using local fallback. status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )

        return ChatResponse(
            answer=_build_local_fallback(
                diagnosis=request.diagnosis,
                knowledge=knowledge,
                question=question,
            ),
            raw_class=raw_class,
            prediction=request.diagnosis.prediction,
            confidence=request.diagnosis.confidence,
            severity=request.diagnosis.severity,
            source="local_fallback",
            model=None,
        )

    # ========================================================
    # 8. Generate Gemini response
    # ========================================================

    try:

        response, model_used = await run_in_threadpool(
            _generate_gemini_response,
            client,
            contents,
            instructions,
        )

    except HTTPException as exc:

        logger.warning(
            "Gemini generation failed. "
            "Returning local fallback. "
            "status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )

        return ChatResponse(
            answer=_build_local_fallback(
                diagnosis=request.diagnosis,
                knowledge=knowledge,
                question=question,
            ),
            raw_class=raw_class,
            prediction=request.diagnosis.prediction,
            confidence=request.diagnosis.confidence,
            severity=request.diagnosis.severity,
            source="local_fallback",
            model=None,
        )

    except Exception:

        logger.exception(
            "Unexpected Gemini error. "
            "Returning local fallback."
        )

        return ChatResponse(
            answer=_build_local_fallback(
                diagnosis=request.diagnosis,
                knowledge=knowledge,
                question=question,
            ),
            raw_class=raw_class,
            prediction=request.diagnosis.prediction,
            confidence=request.diagnosis.confidence,
            severity=request.diagnosis.severity,
            source="local_fallback",
            model=None,
        )

    # ========================================================
    # 9. Extract Gemini answer
    # ========================================================

    answer = _extract_response_text(
        response
    )

    # ========================================================
    # 10. Validate Gemini answer
    # ========================================================

    if not _validate_gemini_answer(
        answer
    ):

        logger.warning(
            "Gemini returned an incomplete or malformed response. "
            "Using local fallback. model=%s response=%r",
            model_used,
            answer[:500],
        )

        return ChatResponse(
            answer=_build_local_fallback(
                diagnosis=request.diagnosis,
                knowledge=knowledge,
                question=question,
            ),
            raw_class=raw_class,
            prediction=request.diagnosis.prediction,
            confidence=request.diagnosis.confidence,
            severity=request.diagnosis.severity,
            source="local_fallback",
            model=model_used,
        )

    # ========================================================
    # 11. Final cleanup
    # ========================================================

    answer = answer.strip()

    if not answer:

        logger.warning(
            "Gemini returned an empty answer. "
            "Using local fallback."
        )

        return ChatResponse(
            answer=_build_local_fallback(
                diagnosis=request.diagnosis,
                knowledge=knowledge,
                question=question,
            ),
            raw_class=raw_class,
            prediction=request.diagnosis.prediction,
            confidence=request.diagnosis.confidence,
            severity=request.diagnosis.severity,
            source="local_fallback",
            model=model_used,
        )

    # ========================================================
    # 12. Return valid Gemini response
    # ========================================================

    return ChatResponse(
        answer=answer,
        raw_class=raw_class,
        prediction=request.diagnosis.prediction,
        confidence=request.diagnosis.confidence,
        severity=request.diagnosis.severity,
        source="gemini",
        model=model_used,
    )