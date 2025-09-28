import numpy as np
import cv2


class RiceLeafAnalyzer:
	def __init__(self):
		# HSV ranges for simple heuristic masks
		self.yellow_hsv_range = ((20, 100, 100), (35, 255, 255))  # Nitrogen
		self.purple_hsv_range = ((130, 50, 50), (160, 255, 255))  # Phosphorus
		self.brown_hsv_range = ((10, 50, 20), (25, 255, 200))     # Potassium

		# Minimal ratios required to have confidence for each class
		self.min_ratio_thresholds = {
			"Nitrogen": 0.15,
			"Phosphorus": 0.08,
			"Potassium": 0.10,
		}

	def analyze_color_features(self, image_rgb: np.ndarray) -> dict:
		"""Return simple color ratios indicative of each deficiency.

		image_rgb: HxWx3 uint8 array in RGB order
		"""
		if image_rgb is None or image_rgb.size == 0:
			raise ValueError("Empty image passed to analyze_color_features")

		# Convert to HSV for simple color masking
		hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
		total_pixels = float(image_rgb.shape[0] * image_rgb.shape[1])

		yellow_mask = cv2.inRange(hsv, self.yellow_hsv_range[0], self.yellow_hsv_range[1])
		purple_mask = cv2.inRange(hsv, self.purple_hsv_range[0], self.purple_hsv_range[1])
		brown_mask = cv2.inRange(hsv, self.brown_hsv_range[0], self.brown_hsv_range[1])

		features = {
			"yellow_ratio": float(np.count_nonzero(yellow_mask)) / total_pixels,
			"purple_ratio": float(np.count_nonzero(purple_mask)) / total_pixels,
			"brown_ratio": float(np.count_nonzero(brown_mask)) / total_pixels,
		}
		return features

	def detect_deficiency(self, image_rgb: np.ndarray) -> str:
		"""Predict deficiency class among {Nitrogen, Phosphorus, Potassium}.

		Returns the class name as one of "Nitrogen", "Phosphorus", or "Potassium".
		"""
		features = self.analyze_color_features(image_rgb)
		scores = {
			"Nitrogen": features["yellow_ratio"],
			"Phosphorus": features["purple_ratio"],
			"Potassium": features["brown_ratio"],
		}

		best_class = max(scores, key=scores.get)
		# If the best score is below threshold, still return the best guess
		# to maintain a strict 3-class output as per your dataset.
		return best_class