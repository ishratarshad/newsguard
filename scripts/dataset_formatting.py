import pandas as pd

# Standardization of news_sample.csv
df1 = pd.read_csv("datasets/news_sample.csv")

df1 = df1.drop_duplicates() # Removes duplicate rows
columns_to_keep1 = ["content", "type"] # Only keep necessary columns relevant to content and labels indicating truthfullness
df1 = df1[columns_to_keep1]
df1 = df1.rename(columns={"content": "text", "type": "label"})  # Standardize column names


# Standardization of BuzzFeed_fake_news_content.csv
df2 = pd.read_csv("datasets/BuzzFeed_fake_news_content.csv")

df2 = df2.drop_duplicates() # Removes duplicate rows
columns_to_keep2 = ["id", "text"] # Only keep necessary columns relevant to content and labels indicating truthfullness
df2 = df2[columns_to_keep2]
df2 = df2.rename(columns={ "id": "label"})  # Standardize column names


# Structuring of data, excluding FakeNewsNet source, as it doesn't necessarily include actual text that works for prompts,
# rather just titles, which might not be suitable for training a model
combo_df_1 = pd.concat([df1, df2], ignore_index=True)
combo_df_1.to_csv("datasets/structured_data_exclude.csv", index = False)


# Standardization of FakeNewsNet.csv
df3 = pd.read_csv("datasets/FakeNewsNet.csv")

df3 = df3.drop_duplicates() # Removes duplicate rows
columns_to_keep3 = ["title", "real"] # Only keep necessary columns relevant to content and labels indicating truthfullness
df3 = df3[columns_to_keep3]
df3 = df3.rename(columns={ "title": "text", "real" : "label"})  # Standardize column names


# Structuring of data, including all sources
combo_df_2 = pd.concat([df1, df2, df3], ignore_index=True)
combo_df_2.to_csv("datasets/structured_data.csv", index = False)