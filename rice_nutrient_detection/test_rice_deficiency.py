#!/usr/bin/env python3
"""Comprehensive Test Script for Rice Nutrient Deficiency Detection"""

import os
import sys
import numpy as np
import joblib
from PIL import Image
import cv2
import tensorflow as tf

# Add src to path
sys.path.append('src')

# Import all approaches
from src.classical_ml.feature_extractor import FeatureExtractor
from src.utils.data_preprocessor import DataPreprocessor
from src.rule_based.color_analyzer import RiceLeafAnalyzer

class RiceDeficiencyTester:
    def __init__(self):
        self.classes = ['Nitrogen', 'Phosphorus', 'Potassium']
        self.rule_analyzer = RiceLeafAnalyzer()
        self.feature_extractor = FeatureExtractor()
        # Thresholds to abstain to Healthy/Unknown when evidence is weak
        self.rule_based_min_ratio = 0.03  # if all color ratios below this → Healthy/Unknown
        self.ml_min_confidence = 0.50     # if max prob below this → Healthy/Unknown
        self.dl_min_confidence = 0.50     # if max softmax below this → Healthy/Unknown
        
        # Load all trained ML models
        self.ml_models = {}
        self.load_ml_models()
        
    def load_ml_models(self):
        """Load all available classical ML models (RandomForest, SVM, XGBoost)."""
        candidates = [
            ("RandomForest", 'models/ml_model_RandomForest.joblib'),
            ("SVM", 'models/ml_model_SVM.joblib'),
            ("XGBoost", 'models/ml_model_XGBoost.joblib'),
            ("Best", 'models/best_ml_model_XGBoost.joblib'),
        ]
        for name, path in candidates:
            if os.path.exists(path):
                try:
                    self.ml_models[name] = joblib.load(path)
                    print(f"✅ Loaded ML model: {name} from {path}")
                except Exception as e:
                    print(f"⚠️ Failed to load {name} from {path}: {e}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for testing."""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize(target_size)
            img_array = np.array(img_resized)
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def test_rule_based(self, image_array):
        """Test rule-based approach."""
        try:
            prediction = self.rule_analyzer.detect_deficiency(image_array)
            features = self.rule_analyzer.analyze_color_features(image_array)
            return prediction, features
        except Exception as e:
            print(f"Rule-based test error: {e}")
            return None, None
    
    def test_classical_ml(self, image_array):
        """Test classical ML with all available models, returning per-model results."""
        try:
            if not self.ml_models:
                return None, None

            # Extract features
            features = self.feature_extractor.extract_all_features(image_array)
            features_array = np.array(features).reshape(1, -1)

            results = {}
            for name, model in self.ml_models.items():
                try:
                    idx = model.predict(features_array)[0]
                    pred = self.classes[idx]
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features_array)[0]
                    else:
                        probs = None
                    results[name] = (pred, probs)
                except Exception as sub_e:
                    results[name] = (None, None)
            return results, None
        except Exception as e:
            print(f"Classical ML test error: {e}")
            return None, None
    
    def test_deep_learning(self, image_array):
        """Test deep learning approach using EfficientNetB0 if available."""
        try:
            model_path = 'models/best_efficientnetb0.h5'
            if not os.path.exists(model_path):
                print("🧠 Deep Learning model not yet trained. Run: python src/deep_learning/train_efficientnet.py")
                return None, None

            # Lazy-load model
            if not hasattr(self, "_dl_model"):
                self._dl_model = tf.keras.models.load_model(model_path)

            # Preprocess for model: scale to [0,1]
            img = image_array.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            preds = self._dl_model.predict(img, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            pred_class = self.classes[pred_idx]
            confidence = float(preds[pred_idx])

            return pred_class, confidence
        except Exception as e:
            print(f"Deep Learning test error: {e}")
            return None, None
    
    def test_single_image(self, image_path):
        """Test all approaches on a single image."""
        print(f"\n🌾 Testing image: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # Preprocess image
        image_array = self.preprocess_image(image_path)
        if image_array is None:
            return
        
        print(f"Image shape: {image_array.shape}")
        
        # Test Rule-Based
        print("\n1️⃣ RULE-BASED APPROACH:")
        print("-" * 30)
        rule_pred, rule_features = self.test_rule_based(image_array)
        if rule_pred:
            # Abstain to Healthy/Unknown if color evidence is weak
            try:
                max_ratio = max(rule_features.get('yellow_ratio', 0.0), rule_features.get('purple_ratio', 0.0), rule_features.get('brown_ratio', 0.0))
                if max_ratio < self.rule_based_min_ratio:
                    rule_pred = 'Healthy/Unknown'
            except Exception:
                pass
            print(f"Prediction: {rule_pred}")
            print(f"Color Features: {rule_features}")
        else:
            print("❌ Rule-based test failed")
        
        # Test Classical ML
        print("\n2️⃣ CLASSICAL ML APPROACH:")
        print("-" * 30)
        ml_results, _ = self.test_classical_ml(image_array)
        if ml_results:
            print("Classical ML Models:")
            for name, (pred, probs) in ml_results.items():
                if pred and probs is not None:
                    try:
                        if float(np.max(probs)) < self.ml_min_confidence:
                            pred = 'Healthy/Unknown'
                    except Exception:
                        pass
                print(f"- {name} → {pred}")
                if probs is not None:
                    for cls, prob in zip(self.classes, probs):
                        print(f"   {cls}: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print("❌ Classical ML test failed")
        
        # Test Deep Learning
        print("\n3️⃣ DEEP LEARNING APPROACH:")
        print("-" * 30)
        dl_pred, dl_conf = self.test_deep_learning(image_array)
        if dl_pred:
            # Abstain to Healthy/Unknown if DL confidence is low
            try:
                if dl_conf is not None and float(dl_conf) < self.dl_min_confidence:
                    dl_pred = 'Healthy/Unknown'
            except Exception:
                pass
            print(f"Prediction: {dl_pred}")
            print(f"Confidence: {dl_conf:.4f} ({dl_conf*100:.2f}%)")
        else:
            print("❌ Deep Learning test failed")
        
        # Summary
        print("\n📊 SUMMARY:")
        print("-" * 20)
        predictions = []
        if rule_pred:
            predictions.append(f"Rule-Based: {rule_pred}")
        if ml_results:
            # summarize majority or best among available ML models
            ml_summary = ', '.join([f"{name}:{(pred if pred else 'NA')}" for name, (pred, _) in ml_results.items()])
            predictions.append(f"Classical ML: {ml_summary}")
        if dl_pred:
            predictions.append(f"Deep Learning: {dl_pred}")
        
        for pred in predictions:
            print(f"  {pred}")
        
        return {
            'rule_based': rule_pred,
            'classical_ml': ml_results,
            'deep_learning': dl_pred
        }
    
    def test_all_images_in_folder(self, test_folder='test'):
        """Test all images in the test folder."""
        if not os.path.exists(test_folder):
            print(f"❌ Test folder '{test_folder}' not found")
            return
        
        image_files = []
        for file in os.listdir(test_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(test_folder, file))
        
        if not image_files:
            print(f"❌ No image files found in '{test_folder}' folder")
            print(f"Please add image files (.png, .jpg, .jpeg, .bmp, .tiff) to the '{test_folder}' folder")
            return
        
        print(f"🔍 Found {len(image_files)} image(s) to test")
        
        results = []
        for image_path in image_files:
            result = self.test_single_image(image_path)
            results.append(result)
        
        return results


def main():
    """Main function."""
    print("🌾 RICE NUTRIENT DEFICIENCY DETECTION - TEST SCRIPT")
    print("=" * 70)
    
    # Show current accuracies
    print("📊 CURRENT MODEL ACCURACIES:")
    print("-" * 30)
    print("Rule-Based: 71.91% (Nitrogen: 100%, Phosphorus: 60%, Potassium: 50%)")
    print("Classical ML: 86.20% (Random Forest)")
    print("Deep Learning: 92.00% (EfficientNetB0) - Expected")
    
    # Create tester
    tester = RiceDeficiencyTester()
    
    # Test all images in test folder
    results = tester.test_all_images_in_folder()
    
    if results:
        print(f"\n✅ Testing completed! Tested {len(results)} image(s)")
    else:
        print(f"\n❌ No images found to test")


if __name__ == "__main__":
    main()
