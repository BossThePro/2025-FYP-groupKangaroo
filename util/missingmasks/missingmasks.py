# We have found out that some masks are missing from the dataset, this piece of code aims to tell us which images are missing masks for the possibility of ignoring these images as they will not work
import pandas as pd
from skimage import io
df = pd.read_csv("../../data/metadata.csv")
df["img_id"] = df["img_id"].str.replace(".png", "") 
imageList = df["img_id"]
skippedImageList = []
imgcount = 0
for img_id in imageList:
    print(f"Current count: {imgcount}\n Current image: {img_id}")
    try:
        mask = io.imread(f"../../../../Downloads/lesion_masks/{img_id}_mask.png")
    except:
        skippedImageList.append(img_id)
        imgcount += 1
        continue 
    imgcount+=1
dict = {"img_id": skippedImageList}
skippedImage = pd.DataFrame(dict)
skippedImage.to_csv("../../data/missing_masks.csv")
