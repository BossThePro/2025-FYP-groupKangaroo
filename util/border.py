import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.color import rgb2gray
from skimage import io, color, filters, measure, morphology, exposure
from skimage.morphology import binary_dilation, binary_erosion, disk
from scipy.spatial import ConvexHull
from collections import deque
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage.segmentation import slic
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
class Border:
    def compactness(self,mask_file):
        #Reads file
        mask = cv2.imread(mask_file)
        #Makes it grayscaled, just in case.
        mask_gray = rgb2gray(mask)
        #Counts all pixels brigther than 0.5
        mask_bin = mask_gray > 0.5
        
        #We sum all the true values.
        A = np.sum(mask_bin)

        #We use a morphology disk
        struct = morphology.disk(2)
        
        #mask_eroded  = morphology.binary_erosion(mask_bin, struct)
        #Returns total perimeter of all objects in BINARY image.
        L = measure.perimeter(mask_bin)
        #migth get easily influenced on the perimeter.
        compactness = (4*np.pi*A) / (L**2)

        return compactness
    def compact_CV(self, mask_file):
        mask = cv2.imread(mask_file)
        mask_gray = rgb2gray(mask)
        
        # Threshold the grayscale image to binary
        mask_bin = mask_gray > 0.5
        
        # Compute area: count all True values
        A = np.sum(mask_bin)

        # Find contours in the binary mask
        contours = measure.find_contours(mask_bin, level=0.5)
        
        # Compute perimeter by summing arc lengths of all contours
        L = sum(
            np.sum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
            for contour in contours
        )
        
        # Compactness formula: (4πA) / L²
        compactness = (4 * np.pi * A) / (L ** 2) if L != 0 else 0
        
        return compactness
#Irregulairity index, starts from 1, mostly goes up till 1.5 if no noise.     
    def convexity(self, mask_file):
        #Reads file
        mask = cv2.imread(mask_file)
        #Makes it grayscaled, just in case.
        mask = rgb2gray(mask)
        coords = np.column_stack(np.nonzero(mask))
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]
        perimeter = measure.perimeter(mask)
        
        hull_perimeter = np.sum(np.sqrt(np.sum(np.diff(np.vstack([hull_coords, hull_coords[0]]), axis = 0)**2, axis = 1)))
        convex_score = perimeter/hull_perimeter
        convex_norm = (1.5 - convex_score) / 0.5
        #Score at around 1.0 means PERFECT circle.
        #Scores at 1.5 ish its jagged. Above that we could say its influenced by noise.
        return convex_norm
    def normalize_sharpness(self, score, min_val=5, max_val=30):
        normalized = (score - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def sharpness(self, image, mask_file):
        #Just in case, we grayscale it
        mask = cv2.imread(mask_file)
        mask = rgb2gray(mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        contours = measure.find_contours(mask, 0.5)
        if not contours:
            return 0.0
        boundary = np.round(contours[0]).astype(int)


        grad_values = []
        for y, x in boundary:
            if 0 <= y < gradient_magnitude.shape[0] and 0 <= x < gradient_magnitude.shape[1]:
                grad_values.append(gradient_magnitude[y, x])
        
        if len(grad_values) == 0:
            return 0.0
        score = self.normalize_sharpness(np.mean(grad_values))
        
        return score
    

    def computeScore(self, image, mask_file):
        #Image needs to be without hair, so processed.
        sharpness_score = self.sharpness(image, mask_file)
        compactness_score = self.compactness(mask_file)
        convexity_score = self.convexity(mask_file)
        #Every score is normalized to a ratio from 0 till 1, where 1 is perfect, the compactness, sharpness and convexity
        #are absolute perfect and the irregularity_score is also perfect.
        #And 0 are worst, this can likely happen if the compactness or convexity is influenced by the shape
        #of the mask.
        irregularity_score = np.mean([sharpness_score,compactness_score,convexity_score])
        return irregularity_score
    
    def visualize_sharpness(self, image, mask_file):
        # Read and convert the mask
        mask = cv2.imread(mask_file)
        mask = rgb2gray(mask)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Find contours
        contours = measure.find_contours(mask, 0.5)
        if not contours:
            print("No contours found.")
            return

        # Visualize contours
        plt.figure(figsize=(8, 8))
        plt.imshow(image[..., ::-1])  # Convert BGR to RGB for displaying with matplotlib
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        plt.title("Lesion Contour on Original Image")
        plt.axis('off')
        plt.show()

        # Optionally, return the first contour if you need to use it later
        return contours[0]
  



    
    def soft_border_gradient_sharpness(self, image_rgb, mask, border_width=10):
        # Normalize and enhance contrast
        image_rgb = image_rgb / 255.0
        image_eq = exposure.equalize_adapthist(image_rgb)
        
        # Convert to L channel of LAB
        lab = rgb2lab(image_eq)
        l_channel = lab[:, :, 0].astype(np.float32)  # Ensure float32 for cv2
        
        # Compute Sobel gradients using OpenCV
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Create border band mask
        dilated = binary_dilation(mask, disk(border_width))
        eroded = binary_erosion(mask, disk(border_width))
        border_band = dilated ^ eroded  # XOR
        
        # Sample gradient values on the border band
        border_vals = gradient_magnitude[border_band]
        
        sharpness_score = np.mean(border_vals)
        score = self.normalize_sharpness(sharpness_score)
        
        # Visualization
        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        ax.contour(mask, colors='lime', linewidths=1)
        ax.contour(border_band, colors='red', linewidths=1)
        ax.set_title(f"Soft Border Sharpness (CV Sobel): {score:.3f}")
        ax.axis('off')
        plt.show()
        
        return score