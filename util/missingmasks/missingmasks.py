# We have found out that some masks are missing from the dataset, this piece of code aims to tell us which images are missing masks for the possibility of ignoring these images as they will not work
import pandas as pd
from skimage import io
df = pd.read_csv("../../data/metadata_clean.csv")
imageList = df["img_id"]
maskImageList = []
imgcount = 0
for img_id in imageList:
    base_id = img_id.replace(".png", "")
    print(f"Current count: {imgcount}\n Current image: {img_id}")
    try:
        mask = io.imread(f"../../../../Documents/lesion_masks/{base_id}_mask.png")
        maskImageList.append(True)
    except:
        maskImageList.append(False)
        imgcount += 1
        continue 
    imgcount+=1
df["has_mask"] = maskImageList
df.to_csv("../../data/metadata_clean.csv", index=False)
df_with_masks = df[df["has_mask"]]
df_with_masks.to_csv("../../data/metadata_clean_masks_only.csv", index=False)

print(df.shape)
print(df_with_masks.shape)