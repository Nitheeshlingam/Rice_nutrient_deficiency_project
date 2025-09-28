import os
import numpy as np
from PIL import Image
from .color_analyzer import RiceLeafAnalyzer

BASE_DIR = "data/rice_plant_lacks_nutrients"
CLASS_DIRS = {
	"Nitrogen": "Nitrogen(N)",
	"Phosphorus": "Phosphorus(P)",
	"Potassium": "Potassium(K)"
}

def test_rule_based_system():
	analyzer = RiceLeafAnalyzer()
	results = {}
	for cls, d in CLASS_DIRS.items():
		class_path = os.path.join(BASE_DIR, d)
		if not os.path.exists(class_path):
			print(f"Missing: {class_path}")
			continue
		files = [f for f in os.listdir(class_path) if not f.startswith(".")][:10]
		correct = 0
		for f in files:
			img = Image.open(os.path.join(class_path, f)).convert("RGB")
			pred = analyzer.detect_deficiency(np.array(img))
			if pred.lower().startswith(cls.lower()[0]):  # N/P/K quick match
				correct += 1
		acc = correct / max(1, len(files))
		results[cls] = acc
		print(f"{cls}: {acc:.2%}")
	return results

if __name__ == "__main__":
	test_rule_based_system()
