from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

#Importing seed from seedGenerator:
with open("seedForModel.txt", "r") as f:
    seed = int(f.read().strip())
print(seed)
#Loading in the data
df_training = pd.read_csv("../../data/trainingTesting/training_data.csv")
df_color_features = pd.read_csv("../../data/color/training_color_non_normalized_features.csv")
df_asymmetry_features = pd.read_csv("../../data/asymmetry/ass_scores_2Features.csv")
df_haralick_features = pd.read_csv("../../data/Haralick/HaralickFeatures.csv")
df_border_features = pd.read_csv("../../data/Border/Border_features.csv")

#Preprocessing of data
df_asymmetry_features["img_id"] = df_asymmetry_features["img_id"].str.replace(".png", "")
df_border_features["img_id"] = df_border_features["img_id"].str.replace(".png", "")
df_haralick_features["img_id"] = df_haralick_features["img_id"].str.replace(".png", "")
df_merged = df_color_features.merge(df_asymmetry_features, on="img_id")
df_merged2 = df_merged.merge(df_haralick_features, on="img_id")
df_color_features = df_merged2.merge(df_border_features, on="img_id")

labels = df_training["cancerous/non-cancerous"].values
exclude_cols = ['img_id', 'cancerous/non-cancerous']
feature_columns = [col for col in df_color_features.columns if col not in exclude_cols]
X_all = df_color_features[feature_columns].values.astype(float)

#Cleaning data from NaN values
not_nan_mask = ~np.isnan(X_all).any(axis=1)
X_all = X_all[not_nan_mask]
labels = labels[not_nan_mask]

#Splitting features into each type for normalization etc.
n_color = 12
n_asym = 2
n_haralick = 13
n_border = 4

X_color = X_all[:, :n_color]
X_asym = X_all[:, n_color:n_color + n_asym]
X_haralick = X_all[:, n_color + n_asym:n_color + n_asym + n_haralick]
X_border = X_all[:, n_color + n_asym + n_haralick: ]

#Scaling on the color feature
scaler_color = MinMaxScaler()
X_color_scaled = scaler_color.fit_transform(X_color)

X_final = np.hstack([X_color_scaled, X_asym, X_border, X_haralick])

# #Reconstructing the feature column name (primarily for testing in terms of a csv file to check everything works)
# color_feature_names = feature_columns[:n_color]
# asym_feature_names = feature_columns[n_color:n_color + n_asym]
# haralick_feature_names = feature_columns[n_color + n_asym:n_color + n_asym + n_haralick]
# border_feature_names = feature_columns[n_color + n_asym + n_haralick:]

# final_column_names = (
#     [f"color_{name}" for name in color_feature_names] +
#     [f"asym_{name}" for name in asym_feature_names] +
#     [f"border_{name}" for name in border_feature_names] +
#     [f"haralick_{name}" for name in haralick_feature_names]
# )

# #Creating dataframe from X_final for testing
# df_X_final = pd.DataFrame(X_final, columns=final_column_names)

# #Saving dataframe to a csv to check
# df_X_final.to_csv("X_final_features.csv", index=False)
# print("Saved X_final to X_final_features.csv")

#Training model using logisticregression
model = LogisticRegression(max_iter=1000000, random_state=seed)
model.fit(X_final, labels)


print(labels.shape)
print(X_final.shape)
#Test Data Preparation
df_test = pd.read_csv("../../data/trainingTesting/test_data.csv")
df_test_color = pd.read_csv("../../data/color/testing_color_non_normalized_features.csv")
df_test_asym = pd.read_csv("../../data/asymmetry/asym_scores_test.csv")
df_test_haralick = pd.read_csv("../../data/Haralick/HaralickFeaturesTest.csv")
df_test_border = pd.read_csv("../../data/Border/Border_features_Test.csv")

df_test_asym["img_id"] = df_test_asym["img_id"].str.replace(".png", "")
df_test_border["img_id"] = df_test_border["img_id"].str.replace(".png", "")
df_test_haralick["img_id"] = df_test_haralick["img_id"].str.replace(".png", "")
df_merged_test = df_test_color.merge(df_test_asym, on="img_id")
df_merged2_test = df_merged_test.merge(df_test_haralick, on="img_id")
df_test_color = df_merged2_test.merge(df_test_border, on="img_id")

test_labels = df_test["cancerous/non-cancerous"].values
X_test = df_test_color[feature_columns].values.astype(float)
not_nan_mask = ~np.isnan(X_test).any(axis=1)
X_test = X_test[not_nan_mask]
test_labels = test_labels[not_nan_mask]

X_test_color = X_test[:, :n_color]
X_test_asym = X_test[:, n_color:n_color + n_asym]
X_test_haralick = X_test[:, n_color + n_asym:n_color + n_asym + n_haralick]
X_test_border = X_test[:, n_color + n_asym + n_haralick:]
X_test_color_scaled = scaler_color.transform(X_test_color)
X_test_final = np.hstack([X_test_color_scaled, X_test_asym, X_test_border, X_test_haralick])


# #Use the same final_column_names from earlier
# df_X_test_final = pd.DataFrame(X_test_final, columns=final_column_names)
# #Save the test data to a csv (to check features are loaded in correctly before running model)
# df_X_test_final.to_csv("X_test_final_features.csv", index=False)
# print("Saved X_test_final to X_test_final_features.csv")
# print(X_test_final.shape)
# print(test_labels.shape)

#Predict test values based on model
y_pred_test = model.predict(X_test_final)
cancerous_index = np.where(model.classes_ == "cancerous")[0][0]
probs_test = model.predict_proba(X_test_final)[:, cancerous_index]

# Evaluation on model
pos_label = "cancerous"
acc = accuracy_score(test_labels, y_pred_test)
prec = precision_score(test_labels, y_pred_test, pos_label=pos_label, zero_division=0)
rec = recall_score(test_labels, y_pred_test, pos_label=pos_label, zero_division=0)
f1 = f1_score(test_labels, y_pred_test, pos_label=pos_label, zero_division=0)
cm = confusion_matrix(test_labels, y_pred_test, labels=[pos_label, "non-cancerous"])
TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

print("Test Set Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:")
print(f" TP: {TP} | FN: {FN}")
print(f" FP: {FP} | TN: {TN}")


#Extracting matching img_ids after removing NaNs
img_ids_test = df_test_color["img_id"].values
img_ids_test = img_ids_test[not_nan_mask]

#Creating a DataFrame for the probabilities with corresponding img_ids
df_probs = pd.DataFrame({
    "img_id": img_ids_test,
    "predicted_probability_cancerous": probs_test
})

#Save to CSV
df_probs.to_csv("test_predicted_probabilities_cancerous_with_haralick.csv", index=False)
print("Saved predicted probabilities to predicted_probabilities_with_img_ids.csv")

