from border import Border
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from inpaint_util import removeHair
import os
from img_util import readImageFile, saveImageFile

irregular = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_161_250_197_mask.png"
good = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_39_55_233_mask.png"

# Path to the images folder
data_dir = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/images"

# Get the base image filename by stripping '_mask.png' and replacing with '.png'
mask_filename = os.path.basename(irregular)
base_image_name = mask_filename.replace("_mask.png", ".png")

# List all image files in the directory
image_files = os.listdir(data_dir)

# Loop through and match
found_path = None
for root, dirs, files in os.walk(data_dir):
    if base_image_name in files:
        image = os.path.join(root, base_image_name)
        break

    



c = Border()

img_rgb, img_gray = readImageFile(image)
blackhat, thresh, img_out = removeHair(img_rgb, img_gray)
score_com = c.compactness(irregular)
score_conv = c.convexity(irregular)
sharp = c.sharpness(img_out, irregular)
norm_conv = (score_conv-1.0)/(1.5-1.0)
X = np.mean([[score_com,score_conv, sharp]])

score = c.computeScore(img_out, irregular)

print(score)
print(score_com, score_conv, sharp)