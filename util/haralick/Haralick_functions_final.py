import cv2
import numpy as np
import mahotas as mh
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray

# Input:
# pathImage: Path to the individual image
# binary: The binary mask to the image.
def loadImageHaralick(pathImage, binary):

    # Load the picture and return a greyscale version of that picture.
    _, grey = readImageFile(pathImage)
    
    # Finding boudaries around the mask to crop image later around the lesion.
    x, y, w, h = cv2.boundingRect(binary)

    # Checking if the dimension are the same.
    if binary.shape[:2] != grey.shape[:2]:
        binary = binary.T

    
    # Image is cropped around the lesion.
    result = cv2.bitwise_and(grey,grey, mask=binary)
    result = result[y:y+h, x:x+w]

    return result

def HaralickExtraction(result):
    h_features = mh.features.haralick(result, ignore_zeros=True)
    # Here we take the mean for all the features for the different GLCM's 
    mean_features = h_features.mean(axis=0)

    return mean_features

# Input: List of mean haralicks features for every image.
def MinMaxScale(list_of_features):

    # Initialising the different list for the different features
    Asm = []
    Con = []
    Cor = []
    Var = []
    Idm = []
    Sa = []
    Sv = []
    Se = []
    Ent = []
    Div = []
    Die = []
    Inf1 = []
    Inf2 = []

    # It goes through each feature for each image and appends it to the relevant list
    for imageFeatures in list_of_features:
        for i in range(13):
            feature = imageFeatures[i]
            match i:
                case 0:
                    Asm.append(feature)
                case 1:
                    Con.append(feature)
                case 2:
                    Cor.append(feature)
                case 3:
                    Var.append(feature)
                case 4:
                    Idm.append(feature)
                case 5:
                    Sa.append(feature)
                case 6:
                    Sv.append(feature)
                case 7:
                    Se.append(feature)
                case 8:
                    Ent.append(feature)
                case 9:
                    Div.append(feature)
                case 10:
                    Die.append(feature)
                case 11:
                    Inf1.append(feature)
                case 12:
                    Inf2.append(feature)

    # Using the MinMaxScaler class to normalise each feature compared to their minimum and maximum value.
    scaler = MinMaxScaler()
    feature_lists = [Asm, Con, Cor, Var, Idm, Sa, Sv, Se, Ent, Div, Die, Inf1, Inf2]
    scaled_data = []
    min_max_values = []

    # Normalising each feature by its own min and max value
    for feat in feature_lists:
        arr = np.array(feat).reshape(-1, 1)
        scaled = scaler.fit_transform(arr).flatten()
        scaled_data.append(scaled)
        min_val = arr.min()
        max_val = arr.max()
        min_max_values.append((min_val, max_val))

    # Returning the scaled data and each feature min and max value
    return scaled_data, min_max_values

# Input:
# csv_file: The directory to the csv containing all the name of all the images you want features extracted from.
# directory_image: The directory to the folder containing all the image with hair being removed.
# directory_mask: The directory to the folder containing all the mask marking the lesions.
def GetFeatures(csv_file, directory_image, directory_mask):

    # Reading the csv file as pandas dataframe
    train = pd.read_csv(csv_file)
    training = train["img_id"]
    list_of_images = list(training)

    featureList = []
    img_name = []
    count = 0
    missing = []

    # Loops through each image found in the directory
    for image in list_of_images:

        # Finding the image
        pathImage = directory_image + "\\" + image
        image_name = image[:-4]
        image_mask = image_name + "_mask.png"

        # Finding the mask
        path_mask = directory_mask + "\\" + image_mask
        mask = cv2.imread(path_mask)

        # Conver the mask to a binary image.
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)  # Convert to single channel
        binary = (binary // 255).astype(np.uint8)  # Normalize to 0 and 1

        # Add name of image to list
        img_name.append(image)

        # Checks if the mask only consists of zeros
        if np.all(binary == 0):
            missing.append(count)
            continue
        
        # Appends each feature from the image to the featureList
        featureList.append(HaralickExtraction(loadImageHaralick(pathImage, binary)))
        count += 1

    # Uses the MinMaxScale function to get the scaled features
    scaledFeatures, FeatureMinMax = MinMaxScale(featureList)

    # If there is an image which had an invalid mask it will get nan values for all the features.
    if len(missing) > 0:
        for miss in missing:
            for i in range(13):
                scaledFeatures[i] = np.insert(scaledFeatures[i], miss, np.nan)
    
    # Creates a dictionary for the features
    featureData = {"img_id": img_name, "Angular Second Moment (Energy)": scaledFeatures[0], "Contrast": scaledFeatures[1], "Correlation": scaledFeatures[2]
               , "Sum of Squares: Variance": scaledFeatures[3], "Inverse Difference Moment":scaledFeatures[4], "Sum Average":scaledFeatures[5],"Sum Variance":scaledFeatures[6], "Sum Entropy":scaledFeatures[7], "Entropy":scaledFeatures[8], 
               "Difference Variance":scaledFeatures[9], "Difference Entropy":scaledFeatures[10], "Information Measure of Corelation 1":scaledFeatures[11], "Information Measure of Corelation 2":scaledFeatures[12]}

    # Creates a dataframe from the dictionary
    featureData = pd.DataFrame(data=featureData)

    # Saves the dataframe as a .csv in the data folder.
    featureData.to_csv("../../data/HaralickFeatures.csv", index = False)
    

    return featureData, FeatureMinMax


# An example use of this code could be
# directory_image = r"C:\blahblah\blahblah\blah\blah\Project in Data Science\images_hair_removed"
# directory_mask = r"C:\blahh\blahblah\blah\blahblah\Project in Data Science\lesion_masks"
# path_csv = r"C:\blah\blahblah\Project in Data Science\2025-FYP-groupKangaroo\data\test_data.csv"

# --- With the execution of this function should do all the work an output a csv file in the data folder. ---
# scaledFeatures, MinMax = GetFeatures(path_csv, directory_image, directory_mask)
