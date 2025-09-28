import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

print("✅ All packages imported successfully!")
print("✅ Your environment is ready!")

# Test basic functionality
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
print(f"✅ Created test image: {img.shape}")

print("\n🎉 Setup complete! Ready to start the rice nutrient detection project!")