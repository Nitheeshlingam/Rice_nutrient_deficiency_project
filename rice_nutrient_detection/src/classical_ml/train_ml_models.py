# src/classical_ml/train_ml_models.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
from .feature_extractor import FeatureExtractor
from ..utils.data_preprocessor import DataPreprocessor

BASE_DIR = "data/rice_plant_lacks_nutrients"


def train_classical_ml():
	prep = DataPreprocessor(BASE_DIR, target_size=(224, 224))
	images, labels = prep.load_images()
	# Extract features
	extractor = FeatureExtractor()
	X = np.array([extractor.extract_all_features(img) for img in images])
	y = labels

	# Split on features
	(X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.split_data(X, y)

	models = {
		"RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
		"SVM": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42),
		"XGBoost": XGBClassifier(
			n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
			colsample_bytree=0.9, eval_metric="mlogloss", random_state=42, tree_method="hist"
		),
	}

	best_name, best_model, best_score = None, None, -1.0
	for name, model in models.items():
		print(f"Training {name}...")
		model.fit(X_train, y_train)
		score = model.score(X_val, y_val)
		if score > best_score:
			best_name, best_model, best_score = name, model, score
		print(f"{name} - Train: {model.score(X_train, y_train):.3f}, Val: {score:.3f}, Test: {model.score(X_test, y_test):.3f}")
		y_pred = model.predict(X_test)
		print("Classification report (Test):")
		print(classification_report(y_test, y_pred, target_names=prep.classes))
		print("Confusion matrix (Test):")
		print(confusion_matrix(y_test, y_pred))

	os.makedirs("models", exist_ok=True)
	joblib.dump(best_model, f"models/best_ml_model_{best_name}.joblib")
	print(f"Saved best model: {best_name} (Val={best_score:.3f})")


if __name__ == "__main__":
	train_classical_ml()
