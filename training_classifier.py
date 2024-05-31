import pandas as pd
from uclassify import uclassify

# Create an instance of the uclassify class
a = uclassify()

# Set the keys
a.setWriteApiKey('cKrbq726LvrT') 
a.setReadApiKey('cnytszVmNdlu') 

# Define the categories
categories = ["business", "sport", "politics", "tech", "entertainment"]

# Create the classifier and add classes
a.create("FinalTextClassifier")
a.addClass(categories, "FinalTextClassifier")

# Define the maximum input size
max_input_size = 2999999

# Read the training data
train_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Train.csv")

# Use 'Text' column as the text for training
train_data['text'] = train_data['Text']

# Train the classifier on the new data
for category in categories:
    # Filter the data for the current category
    category_data = train_data[train_data['Category'] == category]
    
    # Limit the training data to 20,000 samples or less
    sample_size = min(20000, len(category_data))
    category_data = category_data.sample(n=sample_size, random_state=1)
    
    # Split the data into chunks that are small enough for uClassify
    chunks = [category_data[i:i+100] for i in range(0, category_data.shape[0], 100)]
    
    for chunk in chunks:
        # Truncate the text to the maximum input size
        chunk['text'] = chunk['text'].apply(lambda x: x[:max_input_size])
        
        print(f"Training started for category {category}")
        a.train(chunk['text'], category, "FinalTextClassifier")
        print(f"Training finished for category {category}. Trained on {len(chunk)} texts.")