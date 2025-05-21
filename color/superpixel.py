import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
from collections import defaultdict
from sklearn.cluster import KMeans

###### Implementing SLIC superpixel segmentation
# Loading an image
image = io.imread("test_image.png")
#Removing the alpha channel in order to get SLIC superpixel segmentation for the RGB values
if(image.shape[2] == 4):
    image = image[:, :, :3]
#Applying the given mask we got from the dataset on learnit
mask = io.imread("test_mask_image.png") 
mask = mask > 127
# Masking the image by setting the background to 0
masked_image = image * np.expand_dims(mask, axis=-1)  


# Apply the SLIC superpixel segmentation
segments = slic(masked_image, n_segments=100, compactness=10, sigma=1, start_label=1)

# Create a visualization of how it looks after
segmented_image = label2rgb(segments, masked_image, kind='avg')

# Plot the final result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image", fontsize=25)
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title("SLIC Superpixels on masked image", fontsize=25)
ax[1].axis('off')

plt.tight_layout()
plt.show()

####### Implementing the color separation with thresholds and decision rules for each superpixel
# Prepare list to store mean color of each superpixel within the lesion
superpixel_colors = []

for label in np.unique(segments):
    if label == 0: continue  # skip background if needed

    # Create mask for each superpixel
    region_mask = segments == label

    # Only keep it if it's mostly within the lesion mask
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
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(lesion_pixels)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Create image for clustered colors
    clustered_image = np.zeros_like(image)
    clustered_image[mask] = centers[labels]

    # Create a label map to match the shape of the image
    label_map = np.zeros(mask.shape, dtype=np.int32)
    label_map[mask] = labels + 1  # Avoid background = 0

    # Draw cluster boundaries
    outlined_image = mark_boundaries(image, label_map, color=(1, 0, 0), mode='thick')  # red lines

    # Plot everything
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image", fontsize=25)
    ax[0].axis('off')

    ax[1].imshow(clustered_image)
    ax[1].set_title("K-Means Segmented Colors",fontsize=25)
    ax[1].axis('off')

    ax[2].imshow(outlined_image)
    ax[2].set_title("Cluster Outlines (on original)",fontsize=25)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No important colors found, skipping K-Means clustering.")

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

#Final array of features
color_features = np.concatenate((color_ratios, cluster_centers))

print(color_features)