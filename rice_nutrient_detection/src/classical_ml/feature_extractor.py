import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.color import rgb2gray

class FeatureExtractor:
    def __init__(self):
        self.feature_names = []
    
    def extract_color_features(self, image):
        """Extract color-based features"""
        features = []
        
        # RGB channels
        r, g, b = cv2.split(image)
        features.extend([np.mean(r), np.std(r), np.mean(g), np.std(g), np.mean(b), np.std(b)])
        
        # HSV channels
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        features.extend([np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)])
        
        # Color ratios
        total_pixels = image.shape[0] * image.shape[1]
        features.append(np.sum(r > g) / total_pixels)  # Red dominance
        features.append(np.sum(g > r) / total_pixels)  # Green dominance
        features.append(np.sum(b > r) / total_pixels)  # Blue dominance
        
        return features
    
    def extract_texture_features(self, image):
        """Extract texture-based features using GLCM and LBP"""
        gray = rgb2gray(image)
        gray = (gray * 255).astype(np.uint8)
        
        features = []
        
        # GLCM features
        glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        features.extend([
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0]
        ])
        
        # LBP features
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        features.extend([np.mean(lbp), np.std(lbp)])
        
        return features
    
    def extract_all_features(self, image):
        """Extract all features for an image"""
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        
        all_features = color_features + texture_features
        return all_features