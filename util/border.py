import cv2
import numpy as np
import matplotlib.pyplot as plt

# skimage
from skimage import io, color, filters, measure, morphology, exposure
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk, convex_hull_image
from skimage.segmentation import slic, active_contour
from skimage.util import random_noise

# scipy
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d, binary_fill_holes
from scipy.spatial import ConvexHull

# sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# collections
from collections import deque

class Border:
    def intensity_based_fractal_dimensions(self, image_path, epsilons=[2, 4, 8, 16, 32], visualize=False):
        """
        Compute the intensity-based fractal dimension of a grayscale image.
        Optionally visualize grid overlays for each epsilon.

        Parameters:
            image_path (str): Path to the image.
            epsilons (list): List of box sizes (ε) to use for the grid.
            visualize (bool): Whether to show grid overlays for each ε.

        Returns:
            float: Estimated fractal dimension.
        """
        # Load and convert image to grayscale
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        h, w = img_gray.shape

        log_eps = []
        log_I = []

        for eps in epsilons:
            I_eps = 0
            if visualize:
                # Load color image for drawing (optional)
                img_color = cv2.imread(image_path)
            
            for y in range(0, h, eps):
                for x in range(0, w, eps):
                    cell = img_gray[y:y+eps, x:x+eps]
                    if cell.size == 0:
                        continue
                    delta_I = int(np.max(cell)) - int(np.min(cell))
                    I_eps += (delta_I + 1)

                    # Draw grid cell (optional)
                    if visualize and img_color is not None:
                        cv2.rectangle(img_color, (x, y), (x+eps, y+eps), color=(0, 255, 0), thickness=1)

            if I_eps > 0:
                log_eps.append(np.log(1.0 / eps))
                log_I.append(np.log(I_eps))

                # Show the grid overlay for this ε
                if visualize and img_color is not None:
                    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img_rgb)
                    plt.title(f"Grid Overlay (ε = {eps})")
                    plt.axis('off')
                    plt.show()

        # Linear regression on log-log plot
        X = np.array(log_eps).reshape(-1, 1)
        y = np.array(log_I)
        model = LinearRegression().fit(X, y)
        fd = model.coef_[0]

        return fd

    """
    def intensity_based_fractal_dimension(self, image_path, epsilons=[2, 4, 8, 16, 32]):
        """"""
        Compute the intensity-based fractal dimension of a grayscale image.
        Parameters:
            image_path (str): Path to the image.
            epsilons (list): List of box sizes (ε) to use for the grid.
        Returns:
            float: Estimated fractal dimension.
        """ """
        # Load and convert image to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        h, w = img.shape

        log_eps = []
        log_I = []

        for eps in epsilons:
            I_eps = 0
            for y in range(0, h, eps):
                for x in range(0, w, eps):
                    cell = img[y:y+eps, x:x+eps]
                    if cell.size == 0:
                        continue
                    delta_I = int(np.max(cell)) - int(np.min(cell))
                    I_eps += (delta_I + 1)
            if I_eps > 0:
                log_eps.append(np.log(1.0 / eps))
                log_I.append(np.log(I_eps))

        # Linear regression on log-log plot
        X = np.array(log_eps).reshape(-1, 1)
        y = np.array(log_I)
        model = LinearRegression().fit(X, y)
        fd = model.coef_[0]

        return fd
    """
    def compute_fractal_BII(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading {mask_path}")
            return 0.0
        
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        # Keep only largest connected component (remove islands)
        labels = measure.label(mask_bin, connectivity=2)
        if labels.max() == 0:
            return 0.0
        largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
        largest_component = (labels == largest_label)
        
        # Fill holes inside the largest component
        filled_mask = binary_fill_holes(largest_component).astype(np.uint8)
        
        # Extract contours
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        
        # Create binary image of contours only
        boundary_img = np.zeros_like(filled_mask, dtype=np.uint8)
        cv2.drawContours(boundary_img, contours, -1, 1, 1)
        
        # Compute fractal dimension of the contour image
        fd = self.fractal_dimension(boundary_img)
        return fd
    """    
    def peaks(self, mask_path, sigma=2):
        
        Compute a raw irregularity score from a binary lesion mask.

        Parameters:
            mask_path (str): Path to binary mask.
            sigma (float): Gaussian smoothing parameter for radius smoothing.

        Returns:
            float: Normalized radial irregularity score.
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask at: {mask_path}")

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            
            raise ValueError(f"No contour found in mask: {mask_path}")
            
        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            raise ValueError("Zero area contour.")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
   
        # Get radii from centroid
        radii = []
        angles = []
        for point in contour:
            x, y = point[0]
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            radii.append(r)
            angles.append(theta)

        # Sort by angle for continuity
        angles = np.array(angles)
        radii = np.array(radii)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        sorted_indices = np.argsort(angles)
        sorted_radii = radii[sorted_indices]

        # Smooth the radius sequence
        smoothed_radii = gaussian_filter1d(sorted_radii, sigma=sigma)

        # Compute normalized absolute deviation
        mean_radius = np.mean(smoothed_radii)
        irregularity_score = np.mean(np.abs(smoothed_radii - mean_radius)) / mean_radius

        return irregularity_score
 """

    def compute_solidity(self, mask_path):
        """
        Computes the solidity of a lesion from a mask image file.

        Args:
            mask_path (str): Path to the binary mask image (white lesion on black background).

        Returns:
            float: Solidity score (0 to 1). Returns 0.0 if invalid.
        """
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: could not read mask at {mask_path}")
            return 0.0

        # Convert to binary (values: 0 or 1)
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # Compute region properties
        props = measure.regionprops(mask_bin.astype(np.uint8))
        if not props:
            return 0.0

        # Get largest region (in case of noise or multiple)
        largest_region = max(props, key=lambda x: x.area)

        # Return solidity
        return largest_region.solidity
    
    """ 
    Didnt make sense to include, unless we could identify the correect lesion.
    def extract_largest_region(self, mask):
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        if not regions:
            return np.zeros_like(mask)

        # Find the largest region
        largest_region = max(regions, key=lambda r: r.area)
        largest_mask = labeled_mask == largest_region.label
        return largest_mask
 """  
    def compactness(self, mask_file):
        # Reads the file as grayscale
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0.0  # Failed to load
        
        # Ensure binary mask: threshold at 127
        mask_bin = (mask > 127).astype(bool)
        
        # Check for empty mask
        A = np.sum(mask_bin)
        if A == 0:
            return 0.0  # No area to compute

        # Compute perimeter
        L = measure.perimeter(mask_bin, neighborhood=4)
        if L == 0:
            return 0.0  # Avoid division by zero

        # Compute compactness
        compactness = (L ** 2) / (4 * np.pi * A)
        #compactness = (4 * np.pi * A) / (L ** 2)
        
        return compactness
    
#Irregulairity index, starts from 1, mostly goes up till 1.5 if no noise.     
    def convexity(self, mask_path, debug=False):
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Mask is None for path: {mask_path}")
                raise ValueError(f"Invalid mask path or file: {mask_path}")

            if debug:
                cv2.imshow("Input Mask", mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            white_pixels = np.sum(binary > 0)
            print(f"White pixels: {white_pixels}")
            if white_pixels < 100:
                print("Too few white pixels, returning 1.0")
                return 1.0

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(f"Number of contours found: {len(contours)}")
            if not contours:
                print("No contours found, returning 1.0")
                return 1.0

            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            print(f"Largest contour area: {contour_area}")
            if contour_area < 1:
                print("Largest contour area too small, returning 1.0")
                return 1.0

            perimeter = cv2.arcLength(largest_contour, True)
            hull = cv2.convexHull(largest_contour)
            hull_perimeter = cv2.arcLength(hull, True)
            print(f"Perimeter: {perimeter}, Hull perimeter: {hull_perimeter}")

            if hull_perimeter < 1e-6 or perimeter < 1e-6:
                print("Perimeter(s) too small, returning 1.0")
                return 1.0

            convexity_score = max(1.0, perimeter / hull_perimeter)
            print(f"Convexity score: {convexity_score}")

            if debug:
                debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(debug_img, [hull], -1, (0, 0, 255), 2)
                cv2.imshow("Contour (Green) vs Convex Hull (Red)", debug_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return float(convexity_score)

        except Exception as e:
            print(f"Error processing {mask_path}: {str(e)}")
            return 1.00

    

