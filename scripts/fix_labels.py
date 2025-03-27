import pandas as pd
import os

# Load cleaned dataset
dataset_path = os.path.join(os.getcwd(), "datasets", "cleaned_data.csv")
df = pd.read_csv(dataset_path)

# Convert all labels to lowercase (standardize text)
df["label"] = df["label"].astype(str).str.lower()

# Define Fake and Real categories
fake_labels = {"1", "fake", "conspiracy", "hoax", "misleading", "False"}
real_labels = {"0", "real", "trustworthy", "True"}

# Assign "Fake" to known fake labels, "Real" to real labels
df["label"] = df["label"].apply(lambda x: "Fake" if x in fake_labels else "Real" if x in real_labels else "Unknown")

# Remove rows with "Unknown" labels (not useful for training)
df = df[df["label"] != "Unknown"]

# Save the fixed dataset
fixed_dataset_path = os.path.join(os.getcwd(), "datasets", "cleaned_data_fixed.csv")
df.to_csv(fixed_dataset_path, index=False)

print(f"Fixed labels saved to: {fixed_dataset_path}")
