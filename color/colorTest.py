import superpixel as color
import pandas as pd
import numpy as np
##### CHANGE THIS FLAG DEPENDING ON TRAINING OR TESTING -> IMPORTANT OR ELSE TRAINING WILL HAVE TO BE REDONE ENTIRELY OR GIT REVERSED TO PREVIOUS STATE
is_training = True
#scaler file for keeping track of the scale based on training data -> will not be changed in case of testing
scaler_file = "../data/colorScaler.pkl" 
min_max_weight_file = "../data/colorMinMaxWeight.json"
if is_training == True:
    output_csv = "../data/training_color_final_scores.csv"
else:
    output_csv = "../data/test_color_final_scores.csv"
#Loading the images
imagePath = input("Enter the file path of images used: ")
maskPath = input("Enter the file path of masks used: ")
df = pd.read_csv("../data/training_data.csv")
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
#Using a weighted average to get one final score, these values will be refined over time through training the model:
weights = np.array([
    1,  #light brown
    1,  #middle brown
    2,  #dark brown
    1,  #white
    2,  #black
    2,  #blue-grey
    0.5, 0.5, 0.5, #mean R, G, B
    1.5, 1.5, 1.5 #std R, G, B
])
weights = weights / np.sum(weights)



for imageName in imageList:
    image, masked_image, mask = color.loadImage(img_id=imageName, filePathImage=imagePath, filePathMask=maskPath)
    if image is None or masked_image is None or mask is None:
        print(f"Skipping {imageName} due to missing or corrupt image/mask.")
        skippedImageList.append(imageName)
        continue


    features = color.extractColorFeatures(image, masked_image, mask, reference_colors=reference_colors)
    
    if features is not None:
        print(f"Color feature extraction for {imageName} has been completed successfully!")
        color_features_list.append(features)
    else:
        print(f"Skipping color feature extraction for {imageName} due to no important colors.")
        skippedImageList.append(imageName)

color_features = np.array(color_features_list)

scaled_features = color.saveFeatures(features=color_features, is_training=is_training, scaler_file=scaler_file, min_max_weight_file=min_max_weight_file)

color_final = color.finalScore(scaled_features=scaled_features, is_training=is_training, min_max_weight_file=min_max_weight_file, weights=weights)


#Defining final color values
df_scores = pd.DataFrame({
    "img_id": df["img_id"][:len(color_final)],
    "color_score": color_final
})
df_scores.to_csv(output_csv, index=False)