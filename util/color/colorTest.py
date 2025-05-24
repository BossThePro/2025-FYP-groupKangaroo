import superpixel as color
import pandas as pd
import numpy as np
#Loading the images
imagePath = input("Enter the file path of images used: ")
maskPath = input("Enter the file path of masks used: ")
df = pd.read_csv("../../data/metadata_clean_valid_mask_only.csv")
df["img_id"] = df["img_id"].str.replace(".png", "")
imageList = df["img_id"]
skippedImageList = []
color_features_list = []
#List of reference colors used for thresholding in the feature extraction
reference_colors = {
        "light_brown": ([200, 155, 130], 0.12),
        "middle_brown": ([160, 100, 67], 0.25),
        "dark_brown": ([126, 67, 48], 0.2),
        "white": ([230, 230, 230], 0.25),
        "black": ([31, 26, 26], 0.25),
        "blue_grey": ([75, 112, 137], 0.5)
    }
total = 0
image_ids = []

for imageName in imageList:
    total += 1
    image, masked_image, mask = color.loadImage(img_id=imageName, filePathImage=imagePath, filePathMask=maskPath)
    if image is None or masked_image is None or mask is None:
        print(f"Skipping {imageName} due to missing or corrupt image/mask. Image: {total}")
        image_ids.append(imageName)
        features = [np.nan] * 12
        color_features_list.append(features)
        continue


    features = color.extractColorFeatures(image, masked_image, mask, reference_colors=reference_colors)
    
    if features is not None:
        print(f"Color feature extraction for {imageName} has been completed successfully!. Image: {total}")
        color_features_list.append(features)
        image_ids.append(imageName)

    else:
        print(f"Skipping color feature extraction for {imageName} due to no important colors. Image: {total}")
        image_ids.append(imageName)
        features = [np.nan] * 12
        color_features_list.append(features)


color_final = np.array(color_features_list)

# Giving the non normalized values to a csv file such that we only need normalization of color values during cross-validation
df_features = pd.DataFrame(color_features_list, columns=[
    "light_brown", "middle_brown", "dark_brown", "white", "black", "blue_grey",
    "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B"
])
df_features["img_id"] = image_ids
df_features.to_csv("../../data/color/color_non_normalized_features.csv", index=False)