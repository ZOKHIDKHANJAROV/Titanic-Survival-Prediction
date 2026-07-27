from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


REPORT_DIR = Path("reports")


def save_confusion_matrix(
    model,
    X,
    y,
):

    REPORT_DIR.mkdir(
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X,
        y,
        ax=ax,
        cmap="Blues",
    )

    plt.tight_layout()

    path = REPORT_DIR / "confusion_matrix.png"

    plt.savefig(path)

    plt.close(fig)

    return path