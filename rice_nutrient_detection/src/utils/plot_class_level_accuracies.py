from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_metrics() -> Dict[str, Dict[str, float]]:
    """Return class-level accuracies (%) and overall for each model.

    Provided by the user:
    - Rule-Based: N=98, P=72, K=68, Overall=78.50
    - Random Forest: N=88, P=85, K=90, Overall=88.20
    - SVM: N=86, P=80, K=92, Overall=86.00
    - XGBoost (XGB): N=90, P=87, K=93, Overall=89.50
    - EfficientNet B0: N=96, P=92, K=95, Overall=94.00
    """
    return {
        "Rule-Based": {"Nitrogen": 98.0, "Phosphorus": 72.0, "Potassium": 68.0, "Overall": 78.50},
        "Random Forest": {"Nitrogen": 88.0, "Phosphorus": 85.0, "Potassium": 90.0, "Overall": 88.20},
        "SVM": {"Nitrogen": 86.0, "Phosphorus": 80.0, "Potassium": 92.0, "Overall": 86.00},
        "XGBoost (XGB)": {"Nitrogen": 90.0, "Phosphorus": 87.0, "Potassium": 93.0, "Overall": 89.50},
        "EfficientNet B0": {"Nitrogen": 96.0, "Phosphorus": 92.0, "Potassium": 95.0, "Overall": 94.00},
    }


def save_csv(metrics: Dict[str, Dict[str, float]], out_path: Path) -> None:
    models = list(metrics.keys())
    classes = ["Nitrogen", "Phosphorus", "Potassium", "Overall"]

    header = ",".join(["Model"] + classes)
    rows: List[str] = []
    for model in models:
        values = [str(metrics[model].get(cls, "")) for cls in classes]
        rows.append(",".join([model] + values))
    out_path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")


def plot_grouped_bars(metrics: Dict[str, Dict[str, float]], out_png: Path) -> None:
    sns.set_style("whitegrid")

    models = list(metrics.keys())
    classes = ["Nitrogen", "Phosphorus", "Potassium"]

    x = np.arange(len(models))
    width = 0.22

    vals_n = np.array([metrics[m]["Nitrogen"] for m in models])
    vals_p = np.array([metrics[m]["Phosphorus"] for m in models])
    vals_k = np.array([metrics[m]["Potassium"] for m in models])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_n = ax.bar(x - width, vals_n, width, label="Nitrogen", color="#4c78a8")
    bars_p = ax.bar(x, vals_p, width, label="Phosphorus", color="#f58518")
    bars_k = ax.bar(x + width, vals_k, width, label="Potassium", color="#54a24b")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Class-level Accuracies by Model")
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()

    # Annotate bars with values
    for bars in [bars_n, bars_p, bars_k]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.0f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    out_dir = project_root / "rice_nutrient_detection" / "outputs" / "class_accuracy"
    ensure_dir(out_dir)

    metrics = get_metrics()

    # Save CSV snapshot
    save_csv(metrics, out_dir / "class_level_accuracies.csv")

    # Save grouped bar chart
    plot_grouped_bars(metrics, out_dir / "class_level_accuracies.png")

    print(f"Class-level accuracy diagram and CSV saved to: {out_dir}")


if __name__ == "__main__":
    main()
