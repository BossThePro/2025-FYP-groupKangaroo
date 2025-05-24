import pandas as pd
from border import Border
import os
import cv2

b = Border()
df = pd.read_csv("/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/util/borderAdded.csv")
mask_dir = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/lesion_masks" 
convexity_values = []

for img_id in df['img_id']:
    base_id = img_id.replace('.png', '')
    mask_filename = f"{base_id}_mask.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    
    # Load the mask image (assuming it's a binary image)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is not None:
        convexity = b.convexity(mask_path)
    else:
        print(f"Warning: Mask not found for {img_id}")
        convexity = None
    convexity_values.append(convexity)
print(convexity_values)
df["Convexity"] = convexity_values

df.to_csv("borderAdded.csv", index = False)