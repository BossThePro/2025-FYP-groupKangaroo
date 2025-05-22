import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.color import rgb2gray
from skimage import measure, filters
from scipy.spatial import ConvexHull

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
        #Image needs to be without hair.
        sharpness_score = self.sharpness(image, mask_file)
        compactness_score = self.compactness(mask_file)
        convexity_score = self.convexity(mask_file)
        irregularity_score = np.mean([sharpness_score,compactness_score,convexity_score])
        return irregularity_score
