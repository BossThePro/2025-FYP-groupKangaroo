import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def removeHair(img_org, img_gray, kernel_size=25, threshold=10, radius=3):
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out



test_filepath = "data/ISIC_0001769.jpg"



im = cv2.imread(test_filepath)
im2 = im[0:1500,:,:]
im_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 10, 20)


# Display the result
plt.imshow(edges, cmap="gray")
plt.title("Skin Lesion Border using Canny Edge Detection")
plt.show()
