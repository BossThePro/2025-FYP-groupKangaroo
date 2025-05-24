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
from skimage.segmentation import slic, active_contour
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from skimage.filters import gaussian
class Border:
    def compactness(self,mask_file):
        #Reads file
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        #Makes it grayscaled, just in case.
        if mask is None:
            return 0.0
        
        #Counts all pixels brigther than 0.5
        mask_bin = (mask > 127).astype(bool)
        
        #We sum all the true values.
        A = np.sum(mask_bin)

        #We use a morphology disk
        struct = morphology.disk(2)
        
        #mask_eroded  = morphology.binary_erosion(mask_bin, struct)
        #Returns total perimeter of all objects in BINARY image.
        L = measure.perimeter(mask_bin, neighborhood=4)
        #migth get easily influenced on the perimeter.
        compactness = (4*np.pi*A) / (L**2)

        return compactness
    """
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
        """
#Irregulairity index, starts from 1, mostly goes up till 1.5 if no noise.     
    def convexity(self, mask_file):
        # Load and binarize mask
        mask = cv2.imread(mask_file)
        if mask is None:
            return 0.0  # Handle failed load
        
        mask = (rgb2gray(mask) > 0.5).astype(np.uint8)
        if mask.sum() < 50:  # Skip tiny/noisy masks
            return 0.0

        # Get convex hull perimeter
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) < 3:  # Not enough points for a hull
            return 0.0
        
        hull = ConvexHull(coords)
        hull_coords = coords[np.append(hull.vertices, hull.vertices[0])]
        hull_perimeter = np.sum(np.linalg.norm(np.diff(hull_coords, axis=0), axis=1))
        
        # Get actual perimeter
        perimeter = measure.perimeter(mask)
        if hull_perimeter == 0:  # Avoid division by zero
            return 0.0

        convex_score = perimeter / hull_perimeter
        #convex_norm = np.clip((1.5 - convex_score) / 0.5, 0, 1)  # Clip to [0, 1]
        return convex_score
    
    """
    #Used to normalize sharpness, set the max and min to 30 and 5 respectively.
    def normalize_sharpness(self, score, min_val=5, max_val=30):
        normalized = (score - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    """

    #Influenced by masks, if not correct placed around the lesions borders.
    """ Very basic, uses just the mask and gets the sharpness from the border and caclculating the gradients"""
    def sharpness(self, image, mask_file):
        """
        Computes the normalized sharpness score of a lesion's border.
        
        Args:
            image (np.ndarray): Input BGR image.
            mask_file (str): Path to the lesion mask.
            
        Returns:
            float: Normalized sharpness score [0, 1].
        """
        #Load and preprocess mask (ensure binary)
        mask = cv2.imread(mask_file)
        if mask is None:
            return 0.0  # Mask failed to load
        
        mask = rgb2gray(mask)
        mask = (mask > 0.5).astype(np.uint8) * 255  # Binary mask (0 or 255)

        #Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #Compute gradient magnitude (Sobel edges)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        #Find lesion boundary from mask
        contours = measure.find_contours(mask, 0.5)
        if not contours:
            return 0.0  # No contours found
        
        boundary = max(contours, key=len)  # Select longest contour
        boundary = np.round(boundary).astype(int)

        #Extract gradient values along the boundary
        grad_values = [
            gradient_magnitude[y, x]
            for y, x in boundary
            if 0 <= y < gradient_magnitude.shape[0] and 0 <= x < gradient_magnitude.shape[1]
        ]
        
        if len(grad_values) < 10:  # Too few points to trust
            return 0.0

        #Compute and normalize sharpness
        raw_score = np.mean(grad_values)
        return raw_score
    
    """
    def sharpness_color_gradients(self, image, mask_file):
    
        # Load mask and convert to grayscale binary mask
        mask = cv2.imread(mask_file)
        mask_gray = rgb2gray(mask)
        
        # Find contours in the mask to get the lesion boundary
        contours = measure.find_contours(mask_gray, 0.5)
        if not contours:
            return 0.0
        boundary = max(contours, key=len)
        boundary = np.round(boundary).astype(int)

        # Calculate Sobel gradient magnitude for each color channel
        grad_mags = []
        for c in range(3):
            channel = image[:, :, c]
            sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            grad_mags.append(grad_mag)

        # Average gradient magnitude across the 3 channels
        gradient_magnitude = np.mean(grad_mags, axis=0)

        # Extract gradient values at the boundary points
        grad_values = []
        for y, x in boundary:
            if 0 <= y < gradient_magnitude.shape[0] and 0 <= x < gradient_magnitude.shape[1]:
                grad_values.append(gradient_magnitude[y, x])

        if len(grad_values) == 0:
            return 0.0

        # Normalize sharpness score (use your existing normalize method)
        mean_grad = np.mean(grad_values)
        score = self.normalize_sharpness(mean_grad)
        return score
    """
    """
    def computeScore(self, image, mask_file):
        #Image needs to be without hair, so processed.
        sharpness_score = self.sharpness_color_gradients(image, mask_file)
        compactness_score = self.compactness(mask_file)
        convexity_score = self.convexity(mask_file)
        #Every score is normalized to a ratio from 0 till 1, where 1 is perfect, the compactness, sharpness and convexity
        #are absolute perfect and the irregularity_score is also perfect.
        #And 0 are worst, this can likely happen if the compactness or convexity is influenced by the shape
        #of the mask.
        irregularity_score = np.mean([sharpness_score,compactness_score,convexity_score])
        return irregularity_score
    """
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
  
    """Adaptive sharpness method using border bands. Might be useful, when the masks arent reliable and you want to try
    and take outside the masks boundary. The mask can miss the boundary between skin and lesion, which might not get the correct
    sharpness feature then. Using this method, it will increase the chance of correct calculations of sharpness"""
    def soft_border_gradient_sharpness(self, image_rgb, mask, border_width=None, visualize=False):
        """
        Calculate border sharpness using multi-scale gradient analysis in LAB color space.

        Args:
            image_rgb: Input RGB image (0-255 range)
            mask: Binary mask of the lesion
            border_width: Optional fixed border width (automatically calculated if None)
            visualize: Whether to generate visualization plots
            
        Returns:
            Raw sharpness score, need to be normalized after it has been run through all the data.
        """
        # Validate inputs
        if not isinstance(mask, np.ndarray) or mask.dtype != bool:
            mask = mask.astype(bool)
        if np.sum(mask) == 0:
            return 0.0

        # Preprocessing - convert to LAB and use L channel
        #Labdog
        lab = rgb2lab(image_rgb / 255.0)
        l_channel = lab[:, :, 0].astype(np.float32)

        # Adaptive contrast enhancement
        l_channel = exposure.equalize_adapthist(l_channel/np.max(l_channel))

        # Compute gradients using Scharr operator (more accurate than Sobel for 3x3 according to the internet)
        grad_x = cv2.Scharr(l_channel, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(l_channel, cv2.CV_64F, 0, 1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Calculate adaptive border width based on lesion characteristics
        if border_width is None:
            properties = measure.regionprops(mask.astype(int))
            if not properties:
                return 0.0
                
            major_axis = properties[0].major_axis_length
            border_width = max(2, int(major_axis * 0.05))  # 5% of major axis length

        # Multi-scale border analysis with weighted sampling
        sharpness_values = []
        weights = [0.2, 0.6, 0.2]  # Emphasize middle scale
        scales = [0.5, 1.0, 2.0]    # Relative scale factors

        for scale, weight in zip(scales, weights):
            width = max(1, int(border_width * scale))
            
            # Create border band
            outer = binary_dilation(mask, disk(width))
            inner = binary_erosion(mask, disk(width))
            border_band = np.logical_and(outer, ~inner)
            
            if np.sum(border_band) > 0:
                # Use percentile to be robust to outliers
                band_values = gradient_magnitude[border_band]
                sharpness_values.append(weight * np.percentile(band_values, 85))

        if not sharpness_values:
            return 0.0

        # Combine multi-scale measurements
        sharpness_score = np.sum(sharpness_values)

        # Normalize based on image-wide gradient distribution
        norm_factor = np.percentile(gradient_magnitude, 95)
        if norm_factor > 0:
            sharpness_score /= norm_factor

        # Visualization if requested
        if visualize:
            self._visualize_border_sharpness(
                image_rgb, mask, gradient_magnitude, 
                border_width, sharpness_score
            )

        return sharpness_score
    


    def _visualize_border_sharpness(self, image_rgb, mask, gradient_mag, width, score):
        """Helper method for visualization"""
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Original image with contour
        ax[0].imshow(image_rgb)
        ax[0].contour(mask, colors='yellow', linewidths=1.5)
        ax[0].set_title('Original Image')

        # Gradient magnitude with optimal border
        optimal_border = binary_dilation(mask, disk(width)) ^ binary_erosion(mask, disk(width))
        gradient_display = np.log1p(gradient_mag)  # Log scale for better visibility
        ax[1].imshow(gradient_display, cmap='viridis')
        ax[1].contour(optimal_border, colors='red', linewidths=1)
        ax[1].set_title(f'Gradient Magnitude\nBorder Width: {width}px')

        # Overlay of gradient on image
        ax[2].imshow(image_rgb)
        gradient_normalized = gradient_mag / np.max(gradient_mag)
        ax[2].imshow(gradient_normalized, cmap='hot', alpha=0.4)
        ax[2].contour(mask, colors='lime', linewidths=2)
        ax[2].set_title(f'Sharpness Score: {score:.3f}')

        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.show()