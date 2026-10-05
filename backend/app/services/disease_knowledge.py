"""
Disease knowledge base for CoffeeLeaf AI.

This module contains structured, user-facing information for the
six classes supported by the RoCoLe multiclass model.

The ML model is responsible for classification.
This module is responsible only for explaining the classification
result to the user.

Supported classes:
    0 - healthy
    1 - red_spider_mite
    2 - rust_level_1
    3 - rust_level_2
    4 - rust_level_3
    5 - rust_level_4
"""

from __future__ import annotations

from typing import Any, Dict


# ============================================================
# DISEASE KNOWLEDGE
# ============================================================

DISEASE_KNOWLEDGE: Dict[str, Dict[str, Any]] = {

    # ========================================================
    # HEALTHY
    # ========================================================

    "healthy": {
        "title": "Healthy Coffee Leaf",

        "short_description": (
            "The uploaded leaf was classified as healthy by the "
            "CoffeeLeaf AI image classification model."
        ),

        "what_is_it": (
            "A healthy coffee leaf generally does not show major "
            "visible symptoms associated with the disease classes "
            "supported by this system."
        ),

        "symptoms": [
            "Generally green leaf coloration",
            "No major rust-like lesions",
            "No substantial visible affected region",
            "No prominent discoloration associated with the supported disease classes",
        ],

        "what_to_monitor": [
            "Changes in leaf color",
            "New spots or lesions",
            "Rust-colored regions",
            "Unusual speckling or discoloration",
            "Changes appearing on nearby leaves",
        ],

        "monitoring": (
            "Continue regularly inspecting the coffee plant and "
            "nearby leaves for changes in color, spots, lesions, "
            "or other unusual symptoms."
        ),

        "prevention": (
            "Maintain regular plant monitoring, good field hygiene, "
            "and inspect surrounding plants so that unusual symptoms "
            "can be detected early."
        ),

        "when_to_seek_help": (
            "If new or rapidly increasing symptoms appear, consider "
            "consulting a qualified agricultural expert."
        ),

        "disclaimer": (
            "The healthy classification is an AI-based assessment "
            "of the uploaded image and does not guarantee that the "
            "entire plant is free from disease."
        ),
    },


    # ========================================================
    # RED SPIDER MITE
    # ========================================================

    "red_spider_mite": {
        "title": "Red Spider Mite",

        "short_description": (
            "The uploaded leaf was classified as Red Spider Mite "
            "by the CoffeeLeaf AI image classification model."
        ),

        "what_is_it": (
            "Red spider mites are small plant-feeding pests that "
            "can damage leaves while feeding on plant tissue."
        ),

        "symptoms": [
            "Leaf discoloration",
            "Fine speckling or stippling on leaves",
            "Loss of normal green coloration",
            "Progressive visible leaf damage",
            "Affected areas may increase over time",
        ],

        "what_to_monitor": [
            "Fine speckling on nearby leaves",
            "Increasing discoloration",
            "Similar symptoms on surrounding plants",
            "Changes in the affected region over time",
        ],

        "monitoring": (
            "Inspect nearby coffee leaves and plants for similar "
            "symptoms. Pay attention to whether discoloration or "
            "speckling is appearing on additional leaves."
        ),

        "prevention": (
            "Regularly inspect plants and monitor for early signs "
            "of pest activity. Maintaining good crop monitoring can "
            "help identify affected plants early."
        ),

        "when_to_seek_help": (
            "If symptoms are spreading or becoming more severe, "
            "consider consulting a qualified agricultural expert "
            "for identification and appropriate management."
        ),

        "disclaimer": (
            "The Red Spider Mite classification is an AI-based "
            "image assessment and should be confirmed through "
            "appropriate field inspection or expert assessment."
        ),
    },


    # ========================================================
    # COFFEE LEAF RUST - LEVEL 1
    # ========================================================

    "rust_level_1": {
        "title": "Coffee Leaf Rust — Level 1",

        "short_description": (
            "The model classified the uploaded leaf as Coffee Leaf "
            "Rust Level 1, representing an early visible level in "
            "the RoCoLe classification used by this application."
        ),

        "what_is_it": (
            "Coffee Leaf Rust is a fungal disease affecting coffee "
            "leaves. The model uses the RoCoLe rust-level classes "
            "to distinguish different observed levels of rust "
            "symptoms."
        ),

        "symptoms": [
            "Small or limited rust-colored regions",
            "Early visible discoloration",
            "Localized affected areas on the leaf",
        ],

        "what_to_monitor": [
            "Increase in rust-colored regions",
            "New affected areas on the same leaf",
            "Similar symptoms on nearby leaves",
            "Changes in affected-area percentage",
        ],

        "monitoring": (
            "Monitor the affected leaf and surrounding leaves for "
            "changes or expansion of visible symptoms."
        ),

        "prevention": (
            "Regularly inspect coffee plants and surrounding leaves "
            "for early signs of rust symptoms. If symptoms progress, "
            "seek guidance from a qualified agricultural expert."
        ),

        "when_to_seek_help": (
            "If the affected region increases or similar symptoms "
            "appear on multiple leaves, consider consulting a "
            "qualified agricultural expert."
        ),

        "disclaimer": (
            "The Level 1 classification is an AI-based image "
            "assessment. The displayed rust level should not be "
            "treated as a definitive field diagnosis."
        ),
    },


    # ========================================================
    # COFFEE LEAF RUST - LEVEL 2
    # ========================================================

    "rust_level_2": {
        "title": "Coffee Leaf Rust — Level 2",

        "short_description": (
            "The model classified the uploaded leaf as Coffee Leaf "
            "Rust Level 2 in the RoCoLe multiclass classification."
        ),

        "what_is_it": (
            "Coffee Leaf Rust is a fungal disease affecting coffee "
            "leaves. Level 2 represents one of the intermediate "
            "rust-level classes used by the trained RoCoLe model."
        ),

        "symptoms": [
            "More noticeable rust-colored regions",
            "Visible leaf discoloration",
            "Increasing localized leaf damage",
            "Larger visible affected areas compared with early symptoms",
        ],

        "what_to_monitor": [
            "Expansion of rust-colored regions",
            "New lesions or affected areas",
            "Symptoms on nearby leaves",
            "Changes in the affected-area percentage",
        ],

        "monitoring": (
            "Closely monitor the affected leaf and nearby coffee "
            "plants for additional affected regions or progression "
            "of visible symptoms."
        ),

        "prevention": (
            "Continue regular inspection of coffee plants and "
            "surrounding leaves. If symptoms continue to spread, "
            "consider seeking professional agricultural guidance."
        ),

        "when_to_seek_help": (
            "If symptoms are increasing or appearing across "
            "multiple plants, consider consulting a qualified "
            "agricultural expert."
        ),

        "disclaimer": (
            "The Level 2 classification is an AI-based image "
            "assessment and should not be considered a definitive "
            "field diagnosis."
        ),
    },


    # ========================================================
    # COFFEE LEAF RUST - LEVEL 3
    # ========================================================

    "rust_level_3": {
        "title": "Coffee Leaf Rust — Level 3",

        "short_description": (
            "The model classified the uploaded leaf as Coffee Leaf "
            "Rust Level 3 in the RoCoLe multiclass classification."
        ),

        "what_is_it": (
            "Coffee Leaf Rust is a fungal disease affecting coffee "
            "leaves. Level 3 is one of the higher rust-level classes "
            "represented in the trained RoCoLe dataset."
        ),

        "symptoms": [
            "More extensive rust-colored regions",
            "Noticeable leaf discoloration",
            "Larger affected areas",
            "More visible signs of leaf damage",
        ],

        "what_to_monitor": [
            "Further expansion of affected regions",
            "New symptoms on surrounding leaves",
            "Changes in leaf condition",
            "Spread of similar symptoms to nearby plants",
        ],

        "monitoring": (
            "Closely inspect the affected plant and surrounding "
            "plants for progression and additional affected leaves."
        ),

        "prevention": (
            "Maintain regular monitoring of coffee plants and "
            "consider professional agricultural guidance when "
            "symptoms are increasing or spreading."
        ),

        "when_to_seek_help": (
            "If symptoms appear to be spreading across multiple "
            "leaves or plants, consider consulting a qualified "
            "agricultural expert."
        ),

        "disclaimer": (
            "The Level 3 classification is an AI-based image "
            "assessment and does not independently establish a "
            "definitive field diagnosis."
        ),
    },


    # ========================================================
    # COFFEE LEAF RUST - LEVEL 4
    # ========================================================

    "rust_level_4": {
        "title": "Coffee Leaf Rust — Level 4",

        "short_description": (
            "The model classified the uploaded leaf as Coffee Leaf "
            "Rust Level 4 in the RoCoLe multiclass classification."
        ),

        "what_is_it": (
            "Coffee Leaf Rust is a fungal disease affecting coffee "
            "leaves. Level 4 is the highest rust-level class "
            "represented in the six-class RoCoLe model used by "
            "CoffeeLeaf AI."
        ),

        "symptoms": [
            "Extensive rust-colored affected regions",
            "Substantial visible discoloration",
            "Large affected areas of the leaf",
            "More extensive visible leaf damage",
        ],

        "what_to_monitor": [
            "Further expansion of affected areas",
            "Symptoms appearing on nearby leaves",
            "Spread to surrounding plants",
            "Changes in overall plant condition",
        ],

        "monitoring": (
            "Inspect nearby leaves and plants for similar symptoms "
            "and monitor the affected plant for progression."
        ),

        "prevention": (
            "Maintain regular crop monitoring and seek qualified "
            "agricultural guidance when extensive symptoms are "
            "observed."
        ),

        "when_to_seek_help": (
            "For extensive or spreading symptoms, consider "
            "consulting a qualified agricultural expert for "
            "appropriate assessment and management."
        ),

        "disclaimer": (
            "The Level 4 classification is an AI-based image "
            "assessment and should not be treated as a definitive "
            "field diagnosis."
        ),
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_disease_knowledge(raw_class: str) -> Dict[str, Any]:
    """
    Return the knowledge entry for a model class.

    Args:
        raw_class:
            Raw model label, for example:
            'rust_level_2'.

    Returns:
        Dictionary containing the explanation data.

    Raises:
        KeyError:
            If the class is not supported.
    """

    normalized_class = str(raw_class).strip().lower()

    if normalized_class not in DISEASE_KNOWLEDGE:
        raise KeyError(
            f"Unsupported disease class: {raw_class}"
        )

    return DISEASE_KNOWLEDGE[normalized_class]


def get_disease_title(raw_class: str) -> str:
    """
    Return the user-facing title for a model class.
    """

    knowledge = get_disease_knowledge(raw_class)

    return str(
        knowledge.get(
            "title",
            raw_class.replace("_", " ").title()
        )
    )


def get_short_description(raw_class: str) -> str:
    """
    Return a short user-facing description.
    """

    knowledge = get_disease_knowledge(raw_class)

    return str(
        knowledge.get(
            "short_description",
            ""
        )
    )


def get_symptoms(raw_class: str) -> list[str]:
    """
    Return the known symptom list for a model class.
    """

    knowledge = get_disease_knowledge(raw_class)

    symptoms = knowledge.get("symptoms", [])

    return [
        str(symptom)
        for symptom in symptoms
    ]


def get_monitoring_points(raw_class: str) -> list[str]:
    """
    Return monitoring points for a model class.
    """

    knowledge = get_disease_knowledge(raw_class)

    points = knowledge.get(
        "what_to_monitor",
        []
    )

    return [
        str(point)
        for point in points
    ]


def build_disease_explanation(
    raw_class: str,
    confidence: float | None = None,
    affected_percentage: float | None = None,
) -> Dict[str, Any]:
    """
    Build the complete explanation returned by the API.

    The explanation is based on the model's predicted class.

    Args:
        raw_class:
            Model class.

        confidence:
            Model confidence percentage.

        affected_percentage:
            Estimated visually affected leaf area.

    Returns:
        Structured explanation dictionary.
    """

    knowledge = get_disease_knowledge(raw_class)

    explanation: Dict[str, Any] = {
        "title": knowledge["title"],
        "short_description": knowledge["short_description"],
        "what_is_it": knowledge["what_is_it"],
        "symptoms": knowledge["symptoms"],
        "what_to_monitor": knowledge["what_to_monitor"],
        "monitoring": knowledge["monitoring"],
        "prevention": knowledge["prevention"],
        "when_to_seek_help": knowledge["when_to_seek_help"],
        "disclaimer": knowledge["disclaimer"],
    }

    # --------------------------------------------------------
    # Optional model-specific context
    # --------------------------------------------------------

    if confidence is not None:
        explanation["model_confidence"] = round(
            float(confidence),
            1
        )

    if affected_percentage is not None:
        explanation["affected_area_percentage"] = round(
            max(
                0.0,
                min(
                    float(affected_percentage),
                    100.0
                )
            ),
            2
        )

    return explanation