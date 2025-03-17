import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned dataset (now using fixed labels)
dataset_path = os.path.join(os.getcwd(), "datasets", "cleaned_data_fixed.csv")
df = pd.read_csv(dataset_path)

# Count Fake vs. Real news articles
label_counts = df["label"].value_counts()

# Adjust figure size for better visualization
plt.figure(figsize=(8,6))

# Create bar chart
label_counts.plot(kind="bar", color=["red", "green"], edgecolor="black")

# Add labels and title
plt.title("Fake vs. Real News Distribution", fontsize=14)
plt.xlabel("News Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)

# Ensure bars are visible even if counts are imbalanced
plt.yscale("log")  # Log scale to improve visualization if needed

# Save the improved figure
output_path = os.path.join(os.getcwd(), "analysis", "class_distribution.png")
plt.savefig(output_path, dpi=300)

print(f"Class Distribution Chart Saved at: {output_path}")
