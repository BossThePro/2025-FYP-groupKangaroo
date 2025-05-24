import pandas as pd
import random

#Generating a random seed and saving it such that this can be reproduced at a later date (using the same dataset) (has already been used)

#seed = random.randint(0, 2**32 - 1)


#with open("sample_seed.txt", "w") as f:
   #f.write(str(seed))

#Reading in the random seed generated previously

with open("sample_seed.txt", "r") as f:
   seed = int(f.read())

#Getting the training and test data out using pandas, with our predefined seed for recreatability
df = pd.read_csv("../../data/metadata_clean_masks_only.csv")
df.rename(columns={"Unnamed: 0": "index"}, inplace=True)
training_data = df.sample(frac=0.73, random_state = seed)
test_data = df.drop(training_data.index)

training_data.to_csv("../../data/training_data.csv")
test_data.to_csv("../../data/test_data.csv")

#Getting split of each type of cancer and non-cancer to make sure that we have some representation of all skin disease types in both training and test data:

training_data_cancer_type = training_data.groupby("diagnostic")["index"].count()
test_data_cancer_type = test_data.groupby("diagnostic")["index"].count()
print("Training data: ")
print(training_data_cancer_type)
print("Test data: ")
print(test_data_cancer_type)

#Checking percentage of each cancer group to see that it roughly matches our training / test split from the sampling procedure (compared to total amount in dataset)
total_counts = training_data_cancer_type.add(test_data_cancer_type, fill_value=0)
split_df = pd.DataFrame({
    "train_count": training_data_cancer_type,
    "test_count": test_data_cancer_type,
    "total_count": total_counts
})
split_df["train_pct_of_group"] = split_df["train_count"] / split_df["total_count"] * 100
split_df["test_pct_of_group"] = split_df["test_count"] / split_df["total_count"] * 100

print(split_df)

