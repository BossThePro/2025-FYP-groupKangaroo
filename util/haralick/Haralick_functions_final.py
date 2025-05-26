import cv2
import numpy as np
import mahotas as mh
from util.img_util import readImageFile
from sklearn.preprocessing import MinMaxScaler

def loadImageHaralick(pathImage, binary):

    # Load the picture and return a greyscale version of that picture.
    _, grey = readImageFile(pathImage)
    
    # Finding boudaries around the mask to crop image later around the lesion.
    x, y, w, h = cv2.boundingRect(binary)

    # Checking if the dimension are the same.
    if binary.shape[:2] != grey.shape[:2]:
        binary = binary.T

    
    #assert mask is None, "There exist no mask for this lesion"    

    #assert np.all(mask == 0), "The mask only consist of black pixels"
    
    # Image is cropped around the lesion.
    result = cv2.bitwise_and(grey,grey, mask=binary)
    result = result[y:y+h, x:x+w]

    return result

def HaralickExtraction(result):

    # Extract the haralick features for all GLCM's
    h_features = mh.features.haralick(result, ignore_zeros=True)

    # Here we take the mean for all the features for the different GLCM's
    mean_features = h_features.mean(axis=0)

    return mean_features

# Input: List of mean haralicks features for every image.
def MinMaxScale(list_of_features):

    # A list for every feature
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

    # Storing each individual feauture for each image in to their relative feature list.
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

    scaler = MinMaxScaler()

    # Creates list of the features.
    feature_lists = [Asm, Con, Cor, Var, Idm, Sa, Sv, Se, Ent, Div, Die, Inf1, Inf2]

    # List for when the feature has been scaled
    scaled_data = []

    # Storing the minimum and maximum used for scaling for each feature.
    min_max_values = []

    # Looping through each feature list
    for feat in feature_lists:

        # Reshaping the one-dimensionel list to a two-dimensional because the MinMaxScaler neeeds a two dimensionel array to work.
        arr = np.array(feat).reshape(-1, 1)

        # Transform the array back to one dimension.
        scaled = scaler.fit_transform(arr).flatten()

        # The scaled haralick features get added to the 
        scaled_data.append(scaled)

        # Finding the minimum and maximum value
        min_val = arr.min()
        max_val = arr.max()

        #  In a turple add the minimum and maximum to the list
        min_max_values.append((min_val, max_val))

    # Returns the scaled data in 
    return scaled_data, min_max_values

# Input: list of all the names you want to iterate through. Directory to the images where hair has been remove so that it can find the images. Directory to the mask so they can by applied. 
def GetFeatures(list_of_images, directory_image, directory_mask):
    featureList = []
    img_name = []
    count = 0
    missing = []
    for image in list_of_images:

        pathImage = directory_image + "\\" + image
        image_name = image[:-4]
        image_mask = image_name + "_mask.png"

        
        path_mask = directory_mask + "\\" + image_mask
        mask = cv2.imread(path_mask)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)  # Convert to single channel
        binary = (binary // 255).astype(np.uint8)  # Normalize to 0 and 1

        img_name.append(image)

        if np.all(binary == 0):
            missing.append(count)
            continue

        featureList.append(HaralickExtraction(loadImageHaralick(pathImage, binary)))
        count += 1


    scaledFeatures, FeatureMinMax = MinMaxScale(featureList)
    if len(missing) > 0:
        for miss in missing:
            for i in range(13):
                scaledFeatures[i] = np.insert(scaledFeatures[i], miss, np.nan)
    
    return scaledFeatures, FeatureMinMax, img_name