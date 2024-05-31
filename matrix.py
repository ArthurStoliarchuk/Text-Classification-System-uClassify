import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from uclassify import uclassify

# Create an instance of the uclassify class
a = uclassify()

# Set the keys
a.setWriteApiKey('cKrbq726LvrT') 
a.setReadApiKey('cnytszVmNdlu') 

# Define the categories
categories = ["business", "entertainment", "politics", "sport", "tech"]

# Function to classify text
def classify_text(text_to_classify):
    d = a.classify([text_to_classify], "FinalTextClassifier") # TextClassifier5topics + FinalTextClassifier
    scores = [float(score) for category, score in d[0][2]]
    total_score = sum(scores)
    percentages = [(score / total_score) * 100 for score in scores]
    max_percentage_index = percentages.index(max(percentages))
    max_category = d[0][2][max_percentage_index][0]
    return max_category

# Confusion matrix function
def plot_conf_matrix(yt, yp, label_order= ['sport','business','politics','entertainment','tech']):
    cfm = confusion_matrix(yt, yp, labels=label_order)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cfm, annot=True, fmt='.0f', cmap='Blues', xticklabels=label_order, yticklabels=label_order)
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.title('Confusion matrix')
    plt.show()

# Read the training data
train_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Train.csv")

# Classify each text in the training data and save the predicted categories
y_pred_train = [classify_text(text) for text in train_data['Text'].tolist()]

# Read the true categories from the training data
y_true_train = train_data['Category'].tolist()

# Plot the confusion matrix for the training data
plot_conf_matrix(y_true_train, y_pred_train)