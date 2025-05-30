# Requirements
import cv2 as cv
import numpy as np
import math
from PIL import Image
import PIL.ImageOps   

# Functions
def convert_from_cv2_to_pil(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))


def convert_from_pil_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

def reverseImage(img):
    # Color reverser start
    img = convert_from_cv2_to_pil(img)
    img_reverse = PIL.ImageOps.invert(img)
    img_reverse = convert_from_pil_to_cv2(img_reverse)
    # Color reverser end
    
    # Make a grayscale image as well
    gray_reverse = cv.cvtColor(img_reverse, cv.COLOR_BGR2GRAY)
    return img_reverse, gray_reverse

def hairFeatures(img):
    # ---------------------------------------------------------------------- #
    # Run some preparations for both dark and white hairs. Get the kernel size and define the kernel.
    # ---------------------------------------------------------------------- #

    # Parameter calculation pipeline (assumes image is between 500x500 and 1000x1000)
    # Finding the dimensions of the image and finding the square root of their products.
    dimensions = list(np.shape(img))[:2]
    size = int(math.sqrt(dimensions[0]*dimensions[1]))
    # print("Image_size:", size)

    # Normalizing the size and clamping between 0 - 1
    size = (size-500)/500
    if size < 0:
        size = 0
    elif size > 1:
        size = 1
    # print("Normal_size:", size)

    # Calculating kernel size based on smoothstep value. Round to nearest integer
    kernel_size = round(size * 15 + 10)
    # print("Kernel_size:", kernel_size)

    # Kernel for the morphological filtering
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Calculate the grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Calculate the reversed image used for the white hairs
    imgreverse, grayreverse = reverseImage(img)
    
    # ---------------------------------------------------------------------- #
    # Run the blackhat on the dark hairs, and calculate the score.
    # ---------------------------------------------------------------------- #

    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhatBlack = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)

    # Sum up the blackHat and normalize to find the blackhat score (I'm literally just guessing with the *10, as the highest blackhat score I've seen is about 0.09)
    blackhatScoreBlack = (blackhatBlack.sum()/(dimensions[0]*dimensions[1]*255))*10
    # I'll clamp it just in case though
    if blackhatScoreBlack < 0:
        blackhatScoreBlack = 0
    elif blackhatScoreBlack > 1:
        blackhatScoreBlack = 1
    
    # This reversal is done to turn the HIGH amounts of hair into a LOW threshold (reverse is turned off for testing purposes)
    # reverseBlackhatScore = blackhatScoreBlack * (-1) + 1
    # print("Blackhat_score_black:", blackhatScoreBlack)

    # Calculating smoothstep based on blackhat score (reverse is turned off for testing purposes)
    # reverseSmoothstepBlackhat = -2*reverseBlackhatScore**3 + 3*reverseBlackhatScore**2
    smoothstepBlackhat = -2*blackhatScoreBlack**3 + 3*blackhatScoreBlack**2
    # print("Smoothstep_blackhat:", smoothstepBlackhat)

    # Calculating inpainting threshold based on blackhat smoothstep and round to nearest integer (range is 5 - 25)
    thresholdBlack = round(smoothstepBlackhat * 20 + 5)
    # print("Threshold_black:", thresholdBlack)
    
    # ---------------------------------------------------------------------- #
    # Run the blackhat on the white hairs, and calculate the score.
    # ---------------------------------------------------------------------- #
    
    # Perform the blackHat filtering on the grayscale image to find the hair countours
    blackhatWhite = cv.morphologyEx(grayreverse, cv.MORPH_BLACKHAT, kernel)

    # Sum up the blackHat and normalize to find the blackhat score (I'm literally just guessing with the *10, as the highest blackhat score I've seen is about 0.09)
    blackhatScoreWhite = (blackhatWhite.sum()/(dimensions[0]*dimensions[1]*255))*10
    # I'll clamp it just in case though
    if blackhatScoreWhite < 0:
        blackhatScoreWhite = 0
    elif blackhatScoreWhite > 1:
        blackhatScoreWhite = 1
    
    # This reversal is done to turn the HIGH amounts of hair into a LOW threshold (reverse is turned off for testing purposes)
    # reverseBlackhatScore = blackhatScoreWhite * (-1) + 1
    # print("Blackhat_score_white:", blackhatScoreWhite)

    # Calculating smoothstep based on blackhat score
    smoothstepBlackhat = -2*blackhatScoreWhite**3 + 3*blackhatScoreWhite**2
    # reverseSmoothstepBlackhat = -2*reverseBlackhatScore**3 + 3*reverseBlackhatScore**2
    # print("Smoothstep_blackhat:", smoothstepBlackhat)

    # Calculating inpainting threshold based on blackhat smoothstep and round to nearest integer
    thresholdWhite = round(smoothstepBlackhat * 20 + 5)
    # print("Threshold_white:", thresholdWhite)
    
    # ---------------------------------------------------------------------- #
    # Now return the 6 values: blackhatBlack, blackhatScoreBlack, thresholdBlack, blackhatWhite, blackhatScoreWhite, thresholdWhite
    # ---------------------------------------------------------------------- #
    
    return blackhatBlack, blackhatScoreBlack, thresholdBlack, blackhatWhite, blackhatScoreWhite, thresholdWhite

def colorCheck(img):
    # Run the Canny edge detector on grayscale to find edges
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 70, 200, None, 3)
    # Find the color of the edge pixels
    edge_colors = img[np.where(edges > 0)]
    # Calculate the average color value and normalize it
    dimensions = list(np.shape(edge_colors))[:2]
    colorvalue = edge_colors.sum()/(dimensions[0]*dimensions[1]*255)
    colorvalue = np.clip(colorvalue, 0, 255)
    return colorvalue

def removeHairNew(img, radius=3):
    # First get all of the features and scores.
    blackhatBlack, blackhatScoreBlack, thresholdBlack, blackhatWhite, blackhatScoreWhite, thresholdWhite = hairFeatures(img)
    
    # cv.imshow("Black", blackhatBlack)
    # cv.imshow("White", blackhatWhite)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # If both scores are high, run an additional color check
    if blackhatScoreBlack > 0.85 and blackhatScoreWhite > 0.85:
        colorvalue = colorCheck(img)
        
        if colorvalue < 60:
            # intensify the hair countours in preparation for the inpainting algorithm
            _, thresh = cv.threshold(blackhatBlack, thresholdBlack, 255, cv.THRESH_BINARY)

            # inpaint the original image depending on the mask
            img_out = cv.inpaint(img, thresh, radius, cv.INPAINT_TELEA)
        else:
            # intensify the hair countours in preparation for the inpainting algorithm
            _, thresh = cv.threshold(blackhatWhite, thresholdWhite, 255, cv.THRESH_BINARY)

            # inpaint the original image depending on the mask
            img_out = cv.inpaint(img, thresh, radius, cv.INPAINT_TELEA)
    elif blackhatScoreBlack >= blackhatScoreWhite:
        # intensify the hair countours in preparation for the inpainting algorithm
        _, thresh = cv.threshold(blackhatBlack, thresholdBlack, 255, cv.THRESH_BINARY)

        # inpaint the original image depending on the mask
        img_out = cv.inpaint(img, thresh, radius, cv.INPAINT_TELEA)
    elif blackhatScoreBlack < blackhatScoreWhite:
        # intensify the hair countours in preparation for the inpainting algorithm
        _, thresh = cv.threshold(blackhatWhite, thresholdWhite, 255, cv.THRESH_BINARY)

        # inpaint the original image depending on the mask
        img_out = cv.inpaint(img, thresh, radius, cv.INPAINT_TELEA)

    return img_out
