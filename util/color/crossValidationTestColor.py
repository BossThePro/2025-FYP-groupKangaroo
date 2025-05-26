from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
#Reading in data from training and color feature
df_training = pd.read_csv("../../data/trainingTesting/training_data.csv")
df_color_features = pd.read_csv("../../data/color/training_color_non_normalized_features.csv")
df_asymmetry_features = pd.read_csv("../../data/asymmetry/ass_scores_2Features.csv")
df_haralick_features = pd.read_csv("../../data/Haralick/HaralickFeatures.csv")
df_border_features = pd.read_csv("../../data/Border/Border_features.csv")
df_border_features["img_id"] = df_border_features["img_id"].str.replace(".png", "")
df_haralick_features["img_id"] = df_haralick_features["img_id"].str.replace(".png", "")
df_color_features["asymmetry_score"] = df_asymmetry_features["asymmetry_score"]
df_merged = df_color_features.merge(df_haralick_features, on="img_id")
df_color_features = df_merged
df_color_features = df_color_features.merge(df_border_features, on="img_id")
labels = df_training["cancerous/non-cancerous"].values

#Excluding img_id and cancer/noncancer columns from training data (should not be used as a predictor)
exclude_cols = ['img_id', 'cancerous/non-cancerous']
feature_columns = [col for col in df_color_features.columns if col not in exclude_cols]
#Creating an X_all which includes the 12 color features
X_all = df_color_features[feature_columns].values.astype(float)
#Defines the positive label in our dataset (cancerous in our case)
pos_label = "cancerous"
#Defines 5 splits for cross-validation
kf = KFold(n_splits=6, shuffle=True)
not_nan_mask = ~np.isnan(X_all).any(axis=1)
X_all = X_all[not_nan_mask]
labels = labels[not_nan_mask]
for fold, (train_index, val_index) in enumerate(kf.split(X_all), 1):
    X_train, X_val = X_all[train_index], X_all[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
   # Split the features
    n_color = 12
    n_asym = 2
    n_haralick = 13
    n_border = 4
    #Training
    X_train_color = X_train[:, :n_color]
    X_train_asymmetry = X_train[:, n_color:n_color+n_asym]
    X_train_haralick = X_train[:, n_color+n_asym:n_color+n_asym+n_haralick]
    X_train_border = X_train[:, n_color+n_asym+n_haralick:]

    #Validation
    X_val_color = X_val[:, :n_color]
    X_val_asymmetry = X_val[:, n_color:n_color+n_asym]
    X_val_haralick = X_val[:, n_color+n_asym:n_color+n_asym+n_haralick]
    X_val_border = X_val[:, n_color+n_asym+n_haralick:]
    # Scale color and haralick separately
    scaler_color = MinMaxScaler()
    print(X_train_asymmetry)

    X_train_color_scaled = scaler_color.fit_transform(X_train_color)
    X_val_color_scaled = scaler_color.transform(X_val_color)
    #Combine everything: color + asymmetry (not scaled) + haralick
    X_train_scaled = np.hstack([X_train_color_scaled ,X_train_asymmetry, X_train_border, X_train_haralick])

    X_val_scaled = np.hstack([X_val_color_scaled, X_val_asymmetry, X_val_border, X_val_haralick])

    #Applying PCA to reduce amount of features
    #pca = PCA(n_components = "mle")
    #X_train_scaled = pca.fit_transform(X_train_scaled)
    #X_val_scaled = pca.transform(X_val_scaled)

    #print(f"Fold {fold}: PCA reduced to {X_train_scaled.shape[1]} components.")
    # #Using a weighted average to get one final score, these values will be refined over time through training the model:
    # weights = np.array([
    #     1,  #light brown
    #     1,  #middle brown
    #     2,  #dark brown
    #     1,  #white
    #     2,  #black
    #     2,  #blue-grey
    #     0.5, 0.5, 0.5, #mean R, G, B
    #     1.5, 1.5, 1.5 #std R, G, B
    # ])
    # weights = weights / np.sum(weights)
    # #Turning 12 features into 1 and normalizing to [0, 1] scale, this only works due to dropping NaNs
    # train_scores = np.dot(X_train_scaled, weights).reshape(-1, 1)
    # val_scores = np.dot(X_val_scaled, weights).reshape(-1, 1)
    # print(train_scores)
    # score_min = train_scores.min()
    # score_max = train_scores.max()
    # train_scores = (train_scores - score_min) / (score_max - score_min)
    # val_scores = (val_scores - score_min) / (score_max - score_min)
    # val_scores = np.clip(val_scores, 0, 1)
    #Training a given model
    
    #model = LogisticRegression(class_weight={"cancerous": 1.25, "non-cancerous":1}, max_iter=1000000)
    model = LogisticRegression(max_iter=1000000)
    #model = SVC()
    model.fit(X_train_scaled, y_train)
    #Predicting based on validation
    y_pred = model.predict(X_val_scaled)
    probs = model.predict_proba(X_val_scaled)[:, 1]

    #Calculating metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_val, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_val, y_pred, pos_label=pos_label, zero_division=0)

    cm = confusion_matrix(y_val, y_pred, labels=[pos_label, "non-cancerous"])
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    print(probs)
    print(f"Fold {fold} Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(f" TP: {TP} | FN: {FN}")
    print(f" FP: {FP} | TN: {TN}")
    print("-" * 40)