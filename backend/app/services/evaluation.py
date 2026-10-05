"""
Model evaluation service.

Computes REAL metrics by running the currently-loaded
multiclass model over the validation dataset or an uploaded ZIP.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.models import model_loader
from app.services.image_processing import preprocess_image
from app.utils.exceptions import (
    EvaluationDatasetError,
    InvalidImageError,
)

logger = logging.getLogger(__name__)


# =====================================================================
# PATHS
# =====================================================================

MODELS_DIR = (
    Path(__file__).resolve().parent.parent / "models"
)

RESULTS_PATH = (
    MODELS_DIR / "evaluation_results.json"
)

CONFUSION_MATRIX_PATH = (
    MODELS_DIR / "evaluation_confusion_matrix.png"
)


# backend/validation_dataset
VALIDATION_DATASET = (
    Path(__file__).resolve().parents[2]
    / "validation_dataset"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_ZIP_SIZE_BYTES = 500 * 1024 * 1024

MAX_IMAGES = 5000


# =====================================================================
# EXPECTED MULTICLASS LABELS
# =====================================================================

EXPECTED_CLASSES = [
    "healthy",
    "red_spider_mite",
    "rust_level_1",
    "rust_level_2",
    "rust_level_3",
    "rust_level_4",
]


# =====================================================================
# HELPERS
# =====================================================================

def _find_class_dirs(
    root: Path,
    labels: Dict[int, str],
) -> Dict[str, Path]:
    """
    Find directories corresponding to the model's classes.

    Expected folder names:

        healthy
        red_spider_mite
        rust_level_1
        rust_level_2
        rust_level_3
        rust_level_4

    The search also handles ZIP files that contain one
    additional root directory.
    """

    valid_names = {
        str(name).strip().lower()
        for name in labels.values()
    }

    def scan(base: Path) -> Dict[str, Path]:

        found = {}

        if not base.is_dir():
            return found

        for child in base.iterdir():

            if not child.is_dir():
                continue

            folder_name = (
                child.name
                .strip()
                .lower()
            )

            if folder_name in valid_names:

                # Store using the exact model label
                for label in labels.values():

                    if (
                        str(label)
                        .strip()
                        .lower()
                        == folder_name
                    ):
                        found[str(label)] = child
                        break

        return found

    # First search the root itself
    found = scan(root)

    if found:
        return found

    # Then search one level deeper
    for sub in root.iterdir():

        if not sub.is_dir():
            continue

        found = scan(sub)

        if found:
            return found

    return {}


def _extract_zip(
    zip_bytes: bytes,
    dest: Path,
):
    """
    Extract uploaded ZIP safely enough for the
    current evaluation workflow.
    """

    try:

        with zipfile.ZipFile(
            io.BytesIO(zip_bytes)
        ) as zf:

            zf.extractall(dest)

    except zipfile.BadZipFile as exc:

        raise EvaluationDatasetError(
            "Uploaded file is not a valid ZIP."
        ) from exc


def _create_zip_from_folder(
    folder: Path,
) -> bytes:
    """
    Create an in-memory ZIP from a validation folder.
    """

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zf:

        for file in folder.rglob("*"):

            if file.is_file():

                zf.write(
                    file,
                    file.relative_to(folder),
                )

    buffer.seek(0)

    return buffer.read()


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
) -> bytes:
    """
    Create confusion matrix PNG.
    """

    fig, ax = plt.subplots(
        figsize=(8, 7),
        dpi=150,
    )

    im = ax.imshow(
        matrix,
        cmap="Greens",
    )

    ax.set_xticks(
        range(len(class_names))
    )

    ax.set_yticks(
        range(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        class_names,
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Coffee Leaf Multiclass Confusion Matrix"
    )

    vmax = (
        matrix.max()
        if matrix.size
        else 1
    )

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            ax.text(
                j,
                i,
                str(int(value)),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > vmax / 2
                    else "black"
                ),
            )

    fig.colorbar(
        im,
        fraction=0.046,
        pad=0.04,
    )

    fig.tight_layout()

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
    )

    plt.close(fig)

    return buffer.getvalue()


# =====================================================================
# IMAGE PREPROCESSING FOR EVALUATION
# =====================================================================

def _prepare_image_tensor(
    image_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert an image into the tensor format expected
    by EfficientNet-B0.

    preprocess_image() returns:

        H x W x C

    NumPy array.

    The model requires:

        1 x C x H x W
    """

    image = preprocess_image(
        str(image_path)
    )

    if not isinstance(
        image,
        np.ndarray,
    ):
        raise InvalidImageError(
            "Image preprocessing did not return a NumPy array."
        )

    if image.ndim != 3:
        raise InvalidImageError(
            f"Unexpected image shape: {image.shape}"
        )

    # HWC -> CHW
    tensor = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1,
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Float32
    tensor = tensor.float()

    # Move to GPU/CPU
    tensor = tensor.to(
        device,
        non_blocking=True,
    )

    return tensor


# =====================================================================
# CORE EVALUATION
# =====================================================================

def run_evaluation(
    zip_bytes: bytes,
    skip_size_check: bool = False,
) -> dict:

    if not zip_bytes:

        raise EvaluationDatasetError(
            "Uploaded file is empty."
        )

    if (
        not skip_size_check
        and len(zip_bytes) > MAX_ZIP_SIZE_BYTES
    ):

        raise EvaluationDatasetError(
            f"Dataset exceeds "
            f"{MAX_ZIP_SIZE_BYTES // (1024 * 1024)} MB."
        )


    # ---------------------------------------------------------------
    # Load current model
    # ---------------------------------------------------------------

    model = model_loader.get_model()

    device = model_loader.get_device()

    labels = model_loader.load_labels()


    # ---------------------------------------------------------------
    # Validate labels
    # ---------------------------------------------------------------

    if not labels:

        raise EvaluationDatasetError(
            "No model labels were found."
        )


    class_names = [
        labels[index]
        for index in sorted(labels)
    ]


    # Check that this is our multiclass model
    if len(class_names) != 6:

        raise EvaluationDatasetError(
            "Expected a 6-class CoffeeLeaf model, "
            f"but found {len(class_names)} classes: "
            f"{class_names}"
        )


    # Ensure expected labels exist
    normalized_classes = {
        str(name).strip().lower()
        for name in class_names
    }

    missing_classes = (
        set(EXPECTED_CLASSES)
        - normalized_classes
    )

    if missing_classes:

        raise EvaluationDatasetError(
            "Model labels are missing expected classes: "
            f"{sorted(missing_classes)}"
        )


    name_to_index = {
        str(name).strip().lower(): index
        for index, name in labels.items()
    }


    start = time.perf_counter()


    # ---------------------------------------------------------------
    # Temporary extraction
    # ---------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="coffee_eval_"
    ) as tmp:

        tmp_path = Path(tmp)


        _extract_zip(
            zip_bytes,
            tmp_path,
        )


        # -----------------------------------------------------------
        # Find class folders
        # -----------------------------------------------------------

        class_dirs = _find_class_dirs(
            tmp_path,
            labels,
        )


        if not class_dirs:

            expected = ", ".join(
                class_names
            )

            raise EvaluationDatasetError(
                "Couldn't find multiclass folders. "
                f"Expected folders: {expected}"
            )


        # -----------------------------------------------------------
        # Evaluation arrays
        # -----------------------------------------------------------

        y_true = []

        y_pred = []


        per_class_counts = {
            name: 0
            for name in class_names
        }


        skipped = 0


        # -----------------------------------------------------------
        # Evaluate every image
        # -----------------------------------------------------------

        model.eval()


        for class_name, folder in class_dirs.items():

            true_index = name_to_index[
                class_name.strip().lower()
            ]


            for img_path in sorted(
                folder.rglob("*")
            ):

                if (
                    img_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue


                if len(y_true) >= MAX_IMAGES:
                    break


                try:

                    tensor = _prepare_image_tensor(
                        img_path,
                        device,
                    )


                except (
                    InvalidImageError,
                    OSError,
                    ValueError,
                ) as exc:

                    logger.warning(
                        "Skipping image %s: %s",
                        img_path,
                        exc,
                    )

                    skipped += 1

                    continue


                try:

                    with torch.no_grad():

                        logits = model(
                            tensor
                        )

                        prediction = int(
                            torch.argmax(
                                logits,
                                dim=1,
                            ).item()
                        )


                except Exception as exc:

                    logger.warning(
                        "Model failed on %s: %s",
                        img_path,
                        exc,
                    )

                    skipped += 1

                    continue


                y_true.append(
                    true_index
                )

                y_pred.append(
                    prediction
                )

                per_class_counts[
                    class_name
                ] += 1


        # -----------------------------------------------------------
        # Make sure evaluation happened
        # -----------------------------------------------------------

        if not y_true:

            raise EvaluationDatasetError(
                "No readable images found."
            )


        # -----------------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------------

        indices = sorted(
            labels.keys()
        )


        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=indices,
        )


        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=indices,
                zero_division=0,
            )
        )


        accuracy = (
            float(
                np.trace(matrix)
                / matrix.sum()
            )
            if matrix.sum()
            else 0.0
        )


        # Macro averages:
        # every class contributes equally.
        macro_precision = float(
            np.mean(precision)
        )

        macro_recall = float(
            np.mean(recall)
        )

        macro_f1 = float(
            np.mean(f1)
        )


        # Weighted averages:
        # accounts for number of samples per class.
        weighted_precision = float(
            np.average(
                precision,
                weights=support,
            )
        )

        weighted_recall = float(
            np.average(
                recall,
                weights=support,
            )
        )

        weighted_f1 = float(
            np.average(
                f1,
                weights=support,
            )
        )


        # -----------------------------------------------------------
        # Confusion matrix image
        # -----------------------------------------------------------

        confusion_png = (
            _plot_confusion_matrix(
                matrix,
                class_names,
            )
        )


        confusion_b64 = (
            base64.b64encode(
                confusion_png
            ).decode()
        )


        # -----------------------------------------------------------
        # Processing time
        # -----------------------------------------------------------

        elapsed = round(
            (
                time.perf_counter()
                - start
            ) * 1000
        )


        # -----------------------------------------------------------
        # Result object
        # -----------------------------------------------------------

        result = {

            "model": (
                "EfficientNet-B0 "
                "(coffee_leaf_multiclass_model.pth)"
            ),

            "class_names": class_names,

            "accuracy": round(
                accuracy,
                4,
            ),

            # Keep these keys for your frontend
            # Macro metrics are used here.
            "precision": round(
                macro_precision,
                4,
            ),

            "recall": round(
                macro_recall,
                4,
            ),

            "f1": round(
                macro_f1,
                4,
            ),

            "macro_precision": round(
                macro_precision,
                4,
            ),

            "macro_recall": round(
                macro_recall,
                4,
            ),

            "macro_f1": round(
                macro_f1,
                4,
            ),

            "weighted_precision": round(
                weighted_precision,
                4,
            ),

            "weighted_recall": round(
                weighted_recall,
                4,
            ),

            "weighted_f1": round(
                weighted_f1,
                4,
            ),

            "per_class": [

                {
                    "class_name": class_names[i],

                    "precision": round(
                        float(
                            precision[i]
                        ),
                        4,
                    ),

                    "recall": round(
                        float(
                            recall[i]
                        ),
                        4,
                    ),

                    "f1": round(
                        float(
                            f1[i]
                        ),
                        4,
                    ),

                    "support": int(
                        support[i]
                    ),
                }

                for i in range(
                    len(class_names)
                )
            ],

            "confusion_matrix": (
                matrix.tolist()
            ),

            "confusion_matrix_image": (
                confusion_b64
            ),

            "dataset": {

                "total_images_evaluated": (
                    len(y_true)
                ),

                "per_class_counts": (
                    per_class_counts
                ),

                "skipped_unreadable": (
                    skipped
                ),
            },

            "processing_time_ms": elapsed,

            "evaluated_at": time.time(),
        }


        # -----------------------------------------------------------
        # Cache evaluation results
        # -----------------------------------------------------------

        try:

            MODELS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )


            RESULTS_PATH.write_text(
                json.dumps(
                    result,
                    indent=2,
                ),
                encoding="utf-8",
            )


            CONFUSION_MATRIX_PATH.write_bytes(
                confusion_png
            )


        except OSError:

            logger.warning(
                "Couldn't cache evaluation files."
            )


        logger.info(
            "Multiclass evaluation complete: "
            "%.2f%% accuracy on %d images.",
            accuracy * 100,
            len(y_true),
        )


        return result


# =====================================================================
# AUTOMATIC EVALUATION
# =====================================================================

def run_default_evaluation():
    """
    Runs automatically using:

        backend/validation_dataset

    IMPORTANT:
    validation_dataset must contain the six multiclass
    folders for this function to evaluate the new model.
    """

    if RESULTS_PATH.exists():

        logger.info(
            "Using cached evaluation results."
        )

        return get_latest_evaluation()


    if not VALIDATION_DATASET.exists():

        logger.warning(
            "Validation dataset not found: %s",
            VALIDATION_DATASET,
        )

        return None


    logger.info(
        "Running bundled validation dataset evaluation..."
    )


    zip_bytes = _create_zip_from_folder(
        VALIDATION_DATASET
    )


    return run_evaluation(
        zip_bytes,
        skip_size_check=True,
    )


# =====================================================================
# CACHED RESULTS
# =====================================================================

def get_latest_evaluation():

    if not RESULTS_PATH.exists():
        return None


    try:

        return json.loads(
            RESULTS_PATH.read_text(
                encoding="utf-8"
            )
        )


    except (
        OSError,
        json.JSONDecodeError,
    ):

        return None