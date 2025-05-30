import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def load_seed(seed_file="seedForModel.txt"):
    with open(seed_file, "r") as f:
        return int(f.read().strip())


def load_and_merge_data():
    # Training Data
    df_training = pd.read_csv("data/trainingTesting/training_data.csv")
    df_color = pd.read_csv("data/color/training_color_non_normalized_features.csv")
    df_asym = pd.read_csv("data/asymmetry/ass_scores_2Features.csv")
    df_haralick = pd.read_csv("data/Haralick/HaralickFeatures.csv")
    df_border = pd.read_csv("data/Border/Border_features.csv")

    # Cleaning image IDs
    for df in [df_asym, df_haralick, df_border]:
        df["img_id"] = df["img_id"].str.replace(".png", "")

    # Merging features
    df = df_color.merge(df_asym, on="img_id") \
                 .merge(df_haralick, on="img_id") \
                 .merge(df_border, on="img_id")

    return df_training, df


def split_and_scale_features(X_all, scaler_color):
    n_color, n_asym, n_haralick, n_border = 12, 2, 13, 4
    X_color = X_all[:, :n_color]
    X_asym = X_all[:, n_color:n_color + n_asym]
    X_haralick = X_all[:, n_color + n_asym:n_color + n_asym + n_haralick]
    X_border = X_all[:, n_color + n_asym + n_haralick:]

    X_color_scaled = scaler_color.fit_transform(X_color)
    X_final = np.hstack([X_color_scaled, X_asym, X_border, X_haralick])
    return X_final


def prepare_test_data(feature_columns, scaler_color):
    df_test = pd.read_csv("data/trainingTesting/test_data.csv")
    df_test_color = pd.read_csv("data/color/testing_color_non_normalized_features.csv")
    df_test_asym = pd.read_csv("data/asymmetry/asym_scores_test.csv")
    df_test_haralick = pd.read_csv("data/Haralick/HaralickFeaturesTest.csv")
    df_test_border = pd.read_csv("data/Border/Border_features_Test.csv")

    for df in [df_test_asym, df_test_border, df_test_haralick]:
        df["img_id"] = df["img_id"].str.replace(".png", "")

    df = df_test_color.merge(df_test_asym, on="img_id") \
                      .merge(df_test_haralick, on="img_id") \
                      .merge(df_test_border, on="img_id")

    test_labels = df_test["cancerous/non-cancerous"].values
    X_test = df[feature_columns].values.astype(float)

    not_nan_mask = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[not_nan_mask]
    test_labels = test_labels[not_nan_mask]

    n_color, n_asym, n_haralick = 12, 2, 13
    X_test_color = X_test[:, :n_color]
    X_test_asym = X_test[:, n_color:n_color + n_asym]
    X_test_haralick = X_test[:, n_color + n_asym:n_color + n_asym + n_haralick]
    X_test_border = X_test[:, n_color + n_asym + n_haralick:]

    X_test_color_scaled = scaler_color.transform(X_test_color)
    X_test_final = np.hstack([X_test_color_scaled, X_test_asym, X_test_border, X_test_haralick])

    img_ids = df["img_id"].values[not_nan_mask]
    return X_test_final, test_labels, img_ids


def evaluate_model(y_true, y_pred, pos_label="cancerous"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[pos_label, "non-cancerous"])
    TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print("Test Set Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(f" TP: {TP} | FN: {FN}")
    print(f" FP: {FP} | TN: {TN}")


def main():
    seed = load_seed()
    print(f"Seed: {seed}")

    # Load and merge training data
    df_training, df_features = load_and_merge_data()

    labels = df_training["cancerous/non-cancerous"].values
    exclude_cols = ["img_id", "cancerous/non-cancerous"]
    feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    X_all = df_features[feature_columns].values.astype(float)

    not_nan_mask = ~np.isnan(X_all).any(axis=1)
    X_all = X_all[not_nan_mask]
    labels = labels[not_nan_mask]

    # Scale and prepare training features
    scaler_color = MinMaxScaler()
    X_final = split_and_scale_features(X_all, scaler_color)

    # Train model
    model = LogisticRegression(max_iter=1000000, random_state=seed)
    model.fit(X_final, labels)

    print("Training complete.")
    print("Training data shape:", X_final.shape)

    # Prepare test data
    X_test_final, test_labels, img_ids = prepare_test_data(feature_columns, scaler_color)

    # Predict
    y_pred_test = model.predict(X_test_final)
    cancerous_index = np.where(model.classes_ == "cancerous")[0][0]
    probs_test = model.predict_proba(X_test_final)[:, cancerous_index]

    # Evaluate
    evaluate_model(test_labels, y_pred_test)

    # # Save probabilities
    # df_probs = pd.DataFrame({
    #     "img_id": img_ids,
    #     "predicted_probability_cancerous": probs_test
    # })
    # df_probs.to_csv("test_predicted_probabilities_cancerous_with_haralick.csv", index=False)
    # print("Saved predicted probabilities to test_predicted_probabilities_cancerous_with_haralick.csv")


if __name__ == "__main__":
    main()
