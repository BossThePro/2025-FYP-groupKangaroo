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
def loadImage(img_id, filePathImage, filePathMask):
    try: 
        image = io.imread(f"{filePathImage}/{img_id}.png")
        #Removing the alpha channel in order to get SLIC superpixel segmentation for the RGB values
        if(image.shape[2] == 4):
            image = image[:, :, :3]
        #Applying the given mask we got from the dataset on learnit1
        mask = io.imread(f"{filePathMask}/{img_id}_mask.png")
        mask = mask > 127
        if mask.ndim == 3:
            mask = rgb2gray(mask)
            mask = mask > 0.5
        #Masking the image by setting the background to 0
        masked_image = image * np.expand_dims(mask, axis=-1)  
        return image, masked_image, mask
    except:
        return None, None, None 
   
    
def extractColorFeatures(image, masked_image, mask, reference_colors):
    #Apply the SLIC superpixel segmentation
    segments = slic(masked_image, n_segments=100, compactness=10, sigma=1, start_label=1)

    #Create a visualization of how it looks after
    #segmented_image = label2rgb(segments, masked_image, kind='avg')

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
    #Return nothing for color if no important colors are found
    if len(important_colors) == 0:
        return None


    #Use K Means to differentiate between colors
    #Flatten lesion pixels (RGB)
    lesion_pixels = image[mask].reshape(-1, 3)

    n_clusters = len(important_colors)

    if n_clusters > 0:
        #Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(lesion_pixels)
        #Create an image for clustered colors
        #clustered_image = np.zeros_like(image)
        #clustered_image[mask] = centers[labels]

        #Create a label map to match the shape of the image
       # label_map = np.zeros(mask.shape, dtype=np.int32)
        #label_map[mask] = labels + 1  # Avoid background = 0

        #Draw the cluster boundaries
        #outlined_image = mark_boundaries(image, label_map, color=(1, 0, 0), mode='thick')  # red lines

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
        return features
    else:
        print("No important colors found, skipping K-Means clustering.")


#The part below is no longer used, check out colorTest for new version of handling this part

# for img_id in imageList:
#     image, masked_image, mask = loadImage(img_id, filePathImage, filePathMask)
#     if image is None or masked_image is None or mask is None:
#         print(f"Skipping image {img_id} due to missing file(s)")
#         skippedImageList.append(img_id)
#         continue
#     features = extractColorFeatures(image, masked_image, mask, reference_colors)
#     if features is None:
#         print(f"Skipping {img_id} due to no important colors found")
#         skippedImageList.append(img_id)
#         continue
#     color_features_list.append(features)


#Normalizes from 0 to 1 on training data
# def saveFeatures(features, is_training, scaler_file, min_max_weight_file):
#     if is_training:
#         scaler = MinMaxScaler()
#         scaled_features = scaler.fit_transform(features)
#         joblib.dump(scaler, scaler_file)
#     else:
#         scaler = joblib.load(scaler_file)
#         scaled_features = scaler.transform(features)
#     return scaled_features


# def finalScore(scaled_features, is_training, min_max_weight_file, weights):
#     color_final = np.dot(scaled_features, weights)
#     #Normalizing the final scores to (0, 1) in the weighted final average, and differentiating between training and test data
#     if is_training == True:
#         color_final_min = color_final.min()
#         color_final_max = color_final.max()
#         color_final = (color_final - color_final_min) / (color_final_max - color_final_min)
#         with open(f"{min_max_weight_file}", "w") as f:
#             json.dump({"min": float(color_final_min), "max": float(color_final_max)}, f)
#     else:
#         with open(f"{min_max_weight_file}", "r") as f:
#             min_max = json.load(f)
#         color_final_min = min_max["min"]
#         color_final_max = min_max["max"]
#         color_final = (color_final - color_final_min) / (color_final_max - color_final_min)
#         #If testing data hits extremas outside of the min max from the testing range, we limit the data to the [0, 1] scale in order to keep predictions intact (unintended behaviour can occur using certain models if they are outside of known range)
#         color_final = np.clip(color_final, 0, 1)
#     return color_final

