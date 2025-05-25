#Gives a training_color_non_normalized for each column in training data
import pandas as pd

#Load the CSV files
color_features_df = pd.read_csv('../../data/color/color_non_normalized_features.csv')
training_data_df = pd.read_csv('../../data/trainingTesting/training_data.csv')
training_data_df["img_id"] = training_data_df["img_id"].str.replace(".png", "")

#Assuming both files have a column named 'img_id'
#Filter color features based on training img_ids
training_img_ids = training_data_df['img_id']
#Merge based on 'img_id', preserving the initial order from training_data
training_color_features_df = pd.merge(
    training_data_df[['img_id']],  # only need 'img_id' to preserve order
    color_features_df,
    on='img_id',
    how='left'
)

# Save to new CSV
training_color_features_df.to_csv('../../data/color/training_color_non_normalized_features.csv', index=False)