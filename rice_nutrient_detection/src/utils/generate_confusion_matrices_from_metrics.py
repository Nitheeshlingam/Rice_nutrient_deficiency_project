import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_output_dir(output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)


def normalize_rows(cm: np.ndarray) -> np.ndarray:
	with np.errstate(divide='ignore', invalid='ignore'):
		row_sums = cm.sum(axis=1, keepdims=True)
		return np.divide(cm, row_sums, where=row_sums != 0)


def fabricate_cm_from_recalls(
	num_classes: int,
	samples_per_class: int,
	per_class_recall_pct: List[float],
	misclass_patterns: Dict[int, Dict[int, float]] | None = None,
) -> np.ndarray:
	"""Fabricate a confusion matrix from per-class recalls and misclassification patterns.

	- per_class_recall_pct: list of recalls (in %) for each class i; cm[i,i] = round(recall% * samples_per_class)
	- misclass_patterns: mapping true_class -> {pred_class: proportion_of_errors}
	  Proportions for a given row must sum to 1.0 (we normalize if not exact).
	"""
	cm = np.zeros((num_classes, num_classes), dtype=int)

	for i in range(num_classes):
		recall = max(0.0, min(100.0, per_class_recall_pct[i])) / 100.0
		tp = int(round(recall * samples_per_class))
		cm[i, i] = min(tp, samples_per_class)

		errors = samples_per_class - cm[i, i]
		if errors <= 0:
			continue

		pattern = (misclass_patterns or {}).get(i, {})
		# Normalize pattern weights
		total_w = sum(w for j, w in pattern.items() if j != i)
		# If no pattern or zero weights, distribute uniformly among other classes
		if total_w <= 0:
			others = [j for j in range(num_classes) if j != i]
			share = errors // len(others)
			rem = errors % len(others)
			for idx, j in enumerate(others):
				cm[i, j] += share + (1 if idx < rem else 0)
			continue

		# Proportional distribution per pattern
		allocated = 0
		others = [j for j in range(num_classes) if j != i]
		for j in others:
			w = pattern.get(j, 0.0)
			count = int(math.floor(errors * (w / total_w))) if total_w > 0 else 0
			cm[i, j] += count
			allocated += count

		# Distribute any remainder by descending weights to keep sums exact
		remaining = errors - allocated
		if remaining > 0:
			ordered = sorted(others, key=lambda j: pattern.get(j, 0.0), reverse=True)
			k = 0
			while remaining > 0 and k < len(ordered):
				cm[i, ordered[k]] += 1
				remaining -= 1
				k = (k + 1) if (k + 1) < len(ordered) else 0

	return cm


def compute_overall_accuracy(cm: np.ndarray) -> float:
	return float(np.trace(cm)) / float(np.sum(cm)) if np.sum(cm) else 0.0


def plot_confusion_matrix(
	cm: np.ndarray,
	labels: List[str],
	title: str,
	save_path: Path,
	normalize: bool = True,
) -> None:
	plt.figure(figsize=(6, 5))

	if normalize:
		data = normalize_rows(cm) * 100.0
		fmt = ".1f"
		cbar_label = "Percentage (%)"
	else:
		data = cm
		fmt = "d"
		cbar_label = "Count"

	ax = sns.heatmap(
		data,
		annot=True,
		fmt=fmt,
		cmap="Blues",
		xticklabels=labels,
		yticklabels=labels,
		cbar_kws={"label": cbar_label},
		linewidths=0.5,
		linecolor="#eeeeee",
		square=True,
	)

	ax.set_xlabel("Predicted label")
	ax.set_ylabel("True label")
	ax.set_title(title)
	plt.tight_layout()

	save_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(save_path, dpi=200)
	plt.close()


def save_matrix_csv(cm: np.ndarray, labels: List[str], save_path: Path) -> None:
	header = ",".join([" "] + labels)
	rows = []
	for i, label in enumerate(labels):
		rows.append(",".join([label] + [str(v) for v in cm[i, :].tolist()]))
	csv_text = "\n".join([header] + rows) + "\n"
	save_path.parent.mkdir(parents=True, exist_ok=True)
	save_path.write_text(csv_text, encoding="utf-8")


def main() -> Tuple[Path, Dict[str, np.ndarray]]:
	labels = ["Nitrogen", "Phosphorus", "Potassium"]
	N, P, K = 0, 1, 2

	# Per-class recalls (%) and misclassification patterns per model
	# Patterns specify how the errors are distributed to other classes.
	# Example: for RandomForest, minor misclassifications between P and K.
	per_model_recalls: Dict[str, List[float]] = {
		"Rule-Based": [100.0, 72.0, 68.0],
		"RandomForest": [88.0, 85.0, 90.0],
		"SVM": [89.0, 82.0, 87.0],
		"XGBoost": [91.0, 87.0, 89.0],
		"EfficientNetB0": [96.0, 92.0, 95.0],
	}

	misclass_patterns: Dict[str, Dict[int, Dict[int, float]]] = {
		"Rule-Based": {
			# Perfect N: no errors to distribute
			P: {N: 0.2, K: 0.8},  # P mostly confused as K
			K: {P: 0.8, N: 0.2},  # K mostly confused as P
		},
		"RandomForest": {
			# Minor P <-> K confusion
			N: {P: 0.4, K: 0.6},
			P: {K: 0.7, N: 0.3},
			K: {P: 0.7, N: 0.3},
		},
		"SVM": {
			# Slightly lower on P
			N: {P: 0.6, K: 0.4},
			P: {N: 0.5, K: 0.5},
			K: {P: 0.6, N: 0.4},
		},
		"XGBoost": {
			# Strong overall, balanced minor errors
			N: {P: 0.5, K: 0.5},
			P: {N: 0.45, K: 0.55},
			K: {P: 0.55, N: 0.45},
		},
		"EfficientNetB0": {
			# Near-perfect; any rare errors roughly balanced
			N: {P: 0.5, K: 0.5},
			P: {N: 0.5, K: 0.5},
			K: {P: 0.5, N: 0.5},
		},
	}

	epochs_info: Dict[str, str | int] = {
		"Rule-Based": "N/A",
		"RandomForest": 20,
		"SVM": 20,
		"XGBoost": 20,
		"EfficientNetB0": 30,
	}

	project_root = Path(__file__).resolve().parents[3]
	output_dir = project_root / "rice_nutrient_detection" / "outputs" / "confusion_matrices"
	ensure_output_dir(output_dir)

	matrices: Dict[str, np.ndarray] = {}
	for model_name, recalls in per_model_recalls.items():
		cm = fabricate_cm_from_recalls(
			num_classes=len(labels),
			samples_per_class=100,
			per_class_recall_pct=recalls,
			misclass_patterns=misclass_patterns.get(model_name, {}),
		)
		matrices[model_name] = cm

	# Order by overall accuracy derived from recalls (mean of diagonals/row totals)
	def avg_acc(recalls: List[float]) -> float:
		return float(sum(recalls)) / float(len(recalls))

	ordered_models = sorted(per_model_recalls.keys(), key=lambda m: avg_acc(per_model_recalls[m]), reverse=True)

	for rank, model_name in enumerate(ordered_models, start=1):
		recalls = per_model_recalls[model_name]
		cm = matrices[model_name]
		acc = compute_overall_accuracy(cm) * 100.0
		title = (
			f"{model_name} | Recalls N/P/K: {recalls[0]:.0f}/{recalls[1]:.0f}/{recalls[2]:.0f}% | "
			f"Epochs {epochs_info[model_name]}"
		)
		img_name = (
			f"{rank:02d}_{model_name}_recalls_{recalls[0]:.0f}-{recalls[1]:.0f}-{recalls[2]:.0f}" \
			f"_epochs{epochs_info[model_name]}.png"
		)
		csv_name = f"{rank:02d}_{model_name}_counts.csv"

		plot_confusion_matrix(
			cm=cm,
			labels=labels,
			title=title + f" (fabricated acc={acc:.2f}%)",
			save_path=output_dir / img_name,
			normalize=True,
		)
		save_matrix_csv(cm, labels, output_dir / csv_name)

	return output_dir, matrices


if __name__ == "__main__":
	out_dir, _ = main()
	print(f"Generated confusion matrices from per-class recalls in: {out_dir}")
