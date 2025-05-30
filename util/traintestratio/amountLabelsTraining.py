import pandas as pd

df = pd.read_csv("../../data/trainingTesting/training_data.csv")

print(df["cancerous/non-cancerous"].value_counts())