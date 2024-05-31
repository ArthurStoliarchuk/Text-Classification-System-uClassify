import pandas as pd

# Read the training data
train_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\train.csv")

# Filter the data to only include 'politics' category
politics_data = train_data[train_data['Class Index'] == 1]

# Keep only the 'Title' and 'Description' columns
politics_data = politics_data[['Title', 'Description']]

# Save the data to a new CSV file
politics_data.to_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\politics_data.csv", index=False)