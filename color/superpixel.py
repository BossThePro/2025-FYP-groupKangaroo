import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from skimage.color import rgb2gray
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import cv2
###### Implementing SLIC superpixel segmentation
imgcount = 0
#Loading the images

df = pd.read_csv("../data/training_data.csv")
df["img_id"] = df["img_id"].str.replace(".png", "") 
df = df[0:100]
imageList = df["img_id"]
skippedImageList = []
print(imageList)
color_features_list = []
for img_id in imageList:
    print(f"Current count: {imgcount}\n Current image: {img_id}")
    image = io.imread(f"../../pds dataset/images/{img_id}.png")
    #Removing the alpha channel in order to get SLIC superpixel segmentation for the RGB values
    if(image.shape[2] == 4):
       image = image[:, :, :3]
    #Applying the given mask we got from the dataset on learnit1
    try:
        mask = io.imread(f"../../../Downloads/lesion_masks/{img_id}_mask.png")
    except:
        skippedImageList.append(img_id)
        imgcount += 1
        continue 
    mask = mask > 127

    if mask.ndim == 3:
        mask = rgb2gray(mask)
        mask = mask > 0.5
    #Masking the image by setting the background to 0
    masked_image = image * np.expand_dims(mask, axis=-1)  


    #Apply the SLIC superpixel segmentation
    segments = slic(masked_image, n_segments=100, compactness=10, sigma=1, start_label=1)

    #Create a visualization of how it looks after
    segmented_image = label2rgb(segments, masked_image, kind='avg')

    #Plot the final result (commented out for model since this is not needed for the model itself, but nice to see visually to see what happens)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].imshow(image)
    # ax[0].set_title("Original Image", fontsize=25)
    # ax[0].axis('off')

    # ax[1].imshow(segmented_image)
    # ax[1].set_title("SLIC Superpixels on masked image", fontsize=25)
    # ax[1].axis('off')

    # plt.tight_layout()
    # plt.show()

    ####### Implementing the color separation with thresholds and decision rules for each superpixel
    # Prepare list to store mean color of each superpixel within the lesion
    superpixel_colors = []

    for label in np.unique(segments):
        if label == 0: continue  # skip background if needed

        #Create mask for each superpixel
        region_mask = segments == label

        #Only keep it if it's mostly within the lesion mask
        if np.sum(mask[region_mask]) / np.sum(region_mask) > 0.5:
            # Average color within this superpixel
            avg_color = image[region_mask].mean(axis=0)
            superpixel_colors.append(avg_color)

    superpixel_colors = np.array(superpixel_colors)

    #Compare to table of colors which also includes the threshold described in the paper
    reference_colors = {
        "light_brown": ([200, 155, 130], 0.12),
        "middle_brown": ([160, 100, 67], 0.25),
        "dark_brown": ([126, 67, 48], 0.2),
        "white": ([230, 230, 230], 0.25),
        "black": ([31, 26, 26], 0.25),
        "blue_grey": ([75, 112, 137], 0.5)
    }
    #Keep a counter for each color and their appearance in a given image
    color_counter = defaultdict(int)

    #Check through each superpixel created using SLIC and compare to each color with color threshold
    for color in superpixel_colors:
        for name, (ref_rgb, threshold) in reference_colors.items():
            #np.linalg.norm is the euclidean distance
            dist = np.linalg.norm(color - np.array(ref_rgb))
            if dist < threshold*255:
                color_counter[name] += 1
                break


    total_superpixels = len(superpixel_colors)
    important_colors = []
    #Apply the 5% threshold rule, where we only count if at least 5% of the total superpixels have the same color
    for color_name, count in color_counter.items():
        if count / total_superpixels >= 0.05:
            important_colors.append(color_name)
    #Print a list of the important colors
    print(important_colors)


    #Use K Means to differentiate between colors

    # Flatten lesion pixels (RGB)
    lesion_pixels = image[mask].reshape(-1, 3)

    n_clusters = len(important_colors)

    if n_clusters > 0:
        #Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(lesion_pixels)
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        #Create an image for clustered colors
        clustered_image = np.zeros_like(image)
        clustered_image[mask] = centers[labels]

        #Create a label map to match the shape of the image
        label_map = np.zeros(mask.shape, dtype=np.int32)
        label_map[mask] = labels + 1  # Avoid background = 0

        #Draw the cluster boundaries
        outlined_image = mark_boundaries(image, label_map, color=(1, 0, 0), mode='thick')  # red lines

        #Plot everything (also commented out since this is not needed for final model, but nice to visualize)
        # fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        # ax[0].imshow(image)
        # ax[0].set_title("Original Image", fontsize=25)
        # ax[0].axis('off')

        # ax[1].imshow(clustered_image)
        # ax[1].set_title("K-Means Segmented Colors",fontsize=25)
        # ax[1].axis('off')

        # ax[2].imshow(outlined_image)
        # ax[2].set_title("Cluster Outlines (on original)",fontsize=25)
        # ax[2].axis('off')

        # plt.tight_layout()
        # plt.show()
        
        
        #Putting together the features to create a final color_feature we can use in the ML model


        #For SLIC + color decision rules, we define a ratio of colors based on our 6 colors:
        color_ratios = []
        for color in reference_colors:
            count = color_counter.get(color ,0)
            if total_superpixels != 0:
                color_ratios.append(count/total_superpixels)

        color_ratios = np.array(color_ratios)
        #For SLIC + K-means, we take the average rgb value of each cluster in the lesion (to see if colors vary a lot between clusters)

        #Mean of RGB Values
        cluster_centers = kmeans.cluster_centers_.flatten()
        #Mean and standard deviation of RGB in image
        cluster_mean = cluster_centers.reshape(-1,3).mean(axis=0)
        cluster_std = cluster_centers.reshape(-1,3).std(axis=0)
        #Final array of features
        features = np.concatenate((color_ratios, cluster_mean, cluster_std))
        color_features_list.append(features)
    else:
        print("No important colors found, skipping K-Means clustering.")

    imgcount += 1

#Normalizes from 0 to 1
scaler = MinMaxScaler()

color_features = np.array(color_features_list)
scaled_features = scaler.fit_transform(color_features)

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

color_final = np.dot(scaled_features, weights)
#Normalizing the final scores to (0, 1) in the weighted final average
color_final = (color_final - color_final.min()) / (color_final.max() - color_final.min())
color_min = float(color_final.min())
color_max = float(color_final.max())

#Saving the score from training into a file for use in the testing phase - needs to be done
print(color_final)