from border import Border
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from inpaint_util import removeHair
import os
from img_util import readImageFile, saveImageFile
import cv2



irregular = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_161_250_197_mask.png"
good = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_39_55_233_mask.png"
not_good = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_168_260_654_mask.png"
shit = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_54_83_405_mask.png"
# Path to the images folder

mask_value = irregular
data_dir = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/images"

# Get the base image filename by stripping '_mask.png' and replacing with '.png'
mask_filename = os.path.basename(mask_value)
base_image_name = mask_filename.replace("_mask.png", ".png")

# List all image files in the directory
image_files = os.listdir(data_dir)

# Loop through and match
found_path = None
for root, dirs, files in os.walk(data_dir):
    if base_image_name in files:
        image = os.path.join(root, base_image_name)
        break

    


_, mask = readImageFile(mask_value)
c = Border()

img_rgb, img_gray = readImageFile(image)
blackhat, thresh, img_out = removeHair(img_rgb, img_gray)
score_com = c.compactness(mask_value)
score_conv = c.convexity(mask_value)
sharp = c.sharpness_color_gradients(img_out, irregular)
sharp_norm = c.sharpness(img_out, irregular)
norm_conv = (score_conv-1.0)/(1.5-1.0)
X = np.mean([[score_com,score_conv, sharp]])


score = c.computeScore(img_out, good)
print(f"score without borderband: {score}")

score = c.soft_border_gradient_sharpness(img_out, mask, border_width=None, visualize=True)
print("Score with border band: ", np.mean([score_com, score_conv, score]))
print(score)
print("Compact: ", score_com)
print("Score convexity: ",score_conv)
print("Sharpness", sharp_norm)

img_gray_inpainted = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY) 
#deep_score = print(c.compute_lesion_border_sharpness_from_cv2(img_gray_inpainted, mask))
#deep_score = print(c.compute_sharpness_with_snakes(img_gray_inpainted,mask))

"""
import matplotlib.pyplot as plt

plt.imshow(img_out / 255.0)  # Correct: normalize for RGB display
plt.axis('off')
plt.title("Hair-Removed RGB Image")
plt.show()
"""
