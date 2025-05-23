from border import Border
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from inpaint_util import removeHair
import os
from img_util import readImageFile, saveImageFile



irregular = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_161_250_197_mask.png"
good = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_39_55_233_mask.png"
not_good = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/PAT_168_260_654_mask.png"
# Path to the images folder
data_dir = "/Users/simonbruun-simonsen/Desktop/FeatureExtraction/2025-FYP-groupKangaroo/data/images"

# Get the base image filename by stripping '_mask.png' and replacing with '.png'
mask_filename = os.path.basename(not_good)
base_image_name = mask_filename.replace("_mask.png", ".png")

# List all image files in the directory
image_files = os.listdir(data_dir)

# Loop through and match
found_path = None
for root, dirs, files in os.walk(data_dir):
    if base_image_name in files:
        image = os.path.join(root, base_image_name)
        break

    


_, mask = readImageFile(not_good)
c = Border()

img_rgb, img_gray = readImageFile(image)
blackhat, thresh, img_out = removeHair(img_rgb, img_gray)
score_com = c.compactness(irregular)
score_conv = c.convexity(irregular)
sharp = c.sharpness(img_out, irregular)
norm_conv = (score_conv-1.0)/(1.5-1.0)
X = np.mean([[score_com,score_conv, sharp]])


score = c.computeScore(img_out, not_good)
#vis_sharp = c.visualize_sharpness(img_out, good)
#sharp_ = c.sharpness_edge_snapped(img_out, irregular)
#sharp_vis = print(c.visualize_sharpness_with_contours(img_out, good))

#canny_impro = c.cannySharpnessImproved(img_out)
#canny_sharp_mask = c.soft_border_gradient_sharpness(img_out, mask)
#rint(score_com, score_conv, sharp)
"""
for width in [100]:
    print(f"Testing border_width = {width}")
    score = c.soft_border_gradient_sharpness(img_out, mask, border_width=width)
    print(f"Sharpness score: {score:.3f}")
"""
print(sharp)
import matplotlib.pyplot as plt

plt.imshow(img_out / 255.0)  # Correct: normalize for RGB display
plt.axis('off')
plt.title("Hair-Removed RGB Image")
plt.show()

