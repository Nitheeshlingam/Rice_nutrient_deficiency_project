# src/utils/data_preprocessor.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_DIRS = {
	"Nitrogen": "Nitrogen(N)",
	"Phosphorus": "Phosphorus(P)",
	"Potassium": "Potassium(K)"
}
CLASS_LIST = list(CLASS_DIRS.keys())

class DataPreprocessor:
	def __init__(self, base_dir, target_size=(224, 224)):
		self.base_dir = base_dir  # e.g., "data/rice_plant_lacks_nutrients"
		self.target_size = target_size
		self.classes = CLASS_LIST

	def load_images(self):
		images, labels = [], []
		for idx, cls in enumerate(self.classes):
			class_path = os.path.join(self.base_dir, CLASS_DIRS[cls])
			if not os.path.exists(class_path):
				continue
			for fname in os.listdir(class_path):
				if fname.startswith("."):
					continue
				fpath = os.path.join(class_path, fname)
				try:
					img = Image.open(fpath).convert("RGB")
					img = img.resize(self.target_size)
					images.append(np.array(img))
					labels.append(idx)
				except Exception as e:
					print(f"Skip {fpath}: {e}")
		if len(images) == 0:
			raise RuntimeError(f"No images found under base_dir='{self.base_dir}'. Expected subfolders: {list(CLASS_DIRS.values())}")
		return np.array(images), np.array(labels)

	def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
		X_temp, X_test, y_temp, y_test = train_test_split(
			X, y, test_size=test_size, random_state=random_state, stratify=y
		)
		val_ratio = val_size / (1.0 - test_size)
		X_train, X_val, y_train, y_val = train_test_split(
			X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
		)
		return (X_train, y_train), (X_val, y_val), (X_test, y_test)