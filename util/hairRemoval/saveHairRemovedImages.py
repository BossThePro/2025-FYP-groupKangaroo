import hairRemovalNew as hair
import pandas as pd
import cv2
from PIL import Image
import os

df = pd.read_csv("../data/metadata_clean_valid_mask_only.csv")
imageList = df["img_id"]

output_dir = "../../../Documents/images_hair_removed/"
os.makedirs(output_dir, exist_ok=True)
count = 0
for img in imageList:
    count += 1
    image_path = f"../../../Documents/images/{img}"
    img_array = cv2.imread(image_path)  # Read the image
    
    if img_array is None:
        print(f"Warning: could not load {image_path}")
        continue

    new_image = hair.removeHairNew(img_array)  # Now it's an array, not a string

    #Converting BGR from openCV images into RGB before saving
    new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    img_base = os.path.splitext(img)[0]
    save_path = os.path.join(output_dir, f"{img_base}.png")
    Image.fromarray(new_image_rgb).save(save_path)
    print(f"Removed hair from image {img}. Image: {count}")
