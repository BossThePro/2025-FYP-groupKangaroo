#Using pandas and provided data from the research paper to find the highest mean from scores between 50-50 to 85-15 (cut off to try to avoid too few samples of each cancer type in training, even if we bootstrap)

import pandas as pd
df = pd.read_csv("test_training_ratio.csv")
df = df[0:27]
model_cols = ["Logistic regression", "Random forest", "KNN", "SVC"]
df["Train-Test Mean"] = df.groupby("Train-test split ratio")[model_cols].transform("mean").mean(axis = 1)
print(df.loc[df["Train-Test Mean"].idxmax()])









