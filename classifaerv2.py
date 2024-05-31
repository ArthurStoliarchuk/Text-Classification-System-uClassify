import pickle
from sklearn.metrics import confusion_matrix
import streamlit as st
from uclassify import uclassify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from streamlit_navigation_bar import st_navbar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

class TextClassifier:
    def __init__(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1, 2),
                                      min_df=5, norm='l2', encoding='latin-1', lowercase=True)
        self.svd = TruncatedSVD(n_components=5, random_state=0)
        self.le = LabelEncoder()
        self.model = LogisticRegression(random_state=0)

    def fit(self, X, y):
        features = self.tfidf.fit_transform(X).toarray()
        features_svd = self.svd.fit_transform(features)
        labels = self.le.fit_transform(y)
        self.model.fit(features_svd, labels)

    def predict(self, text):
        features = self.tfidf.transform([text]).toarray()
        features_svd = self.svd.transform(features)
        prediction = self.model.predict(features_svd)
        return self.le.inverse_transform(prediction)[0]



# Define the categories
categories = ["business", "entertainment", "politics", "sport", "tech"]

# Load the sklearn model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
svd = pickle.load(open('svd.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))

class UclassifyClassifier:
    def __init__(self):
        self.a = uclassify()
        self.a.setWriteApiKey('cKrbq726LvrT') 
        self.a.setReadApiKey('cnytszVmNdlu') 

    def classify_text(self, text_to_classify):
        try:
            d = self.a.classify([text_to_classify], "FinalTextClassifier")
            scores = [float(score) for category, score in d[0][2]]
            total_score = sum(scores)
            percentages = [(score / total_score) * 100 for score in scores]
            max_percentage_index = percentages.index(max(percentages))
            max_category = d[0][2][max_percentage_index][0]
            emojis = {
                "business": "👩‍💼",
                "entertainment": "🍿",
                "politics": "✊🏼",
                "sport": "🏃🏾🤸",
                "tech": "🤖"
            }
            result = f"{max_category.capitalize()} - {percentages[max_percentage_index]:.2f}% {emojis[max_category]}"
            return result, max_category
        except Exception as e:
            st.error(str(e))

# Read the data
data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Train.csv")

# Initialize classifiers
sklearn_classifier = TextClassifier()
sklearn_classifier.fit(data['Text'], data['Category'])
uclassify_classifier = UclassifyClassifier()

def generate_true_and_predicted_categories(data):
    y_true = data['Category'].tolist()
    y_pred = []
    for text in data['Text'].tolist():
        result, category = uclassify_classifier.classify_text(text)
        if category == "Error":
            continue  # skip this text
        y_pred.append(category)
    return y_true, y_pred

def generate_true_and_predicted_categories_sklearn(data):
    y_true = data['Category'].tolist()
    y_pred = []
    for text in data['Text'].tolist():
        predicted_category = sklearn_classifier.predict(text)
        y_pred.append(predicted_category)
    return y_true, y_pred

# Confusion matrix function
def plot_conf_matrix(yt, yp, label_order= ['sport','business','politics','entertainment','tech']):
    cfm = confusion_matrix(yt, yp, labels=label_order)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cfm, annot=True, fmt='.0f', cmap='Blues', xticklabels=label_order, yticklabels=label_order)
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    plt.title('Confusion matrix')
    st.pyplot(plt)

# Read the data
data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Train.csv")

# Add the 'Article_Length' column to the DataFrame
data['Article_Length'] = data['Text'].apply(lambda x: len(x.split()))

# List of pages
pages = ["Predict", "Test", "Visualize", "Feedback"]

# Page selection
page = st_navbar(["Predict", "Test", "Visualize", "Feedback"])

uclassify_classifier = UclassifyClassifier()

if page == "Predict":
    st.title("Text classification system💻")

    text = st.text_area("Enter the text for classification")

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Predict using my classifier'):
            if "results" not in st.session_state:
                st.session_state.results = []
            result, predicted_category = uclassify_classifier.classify_text(text)
            st.session_state.results.append(f"Predicted category - {result}")
            for result in st.session_state.results:
                st.info(result)

    with col2:
        if st.button('Predict with sklearn'):
            # Transform the user input into features
            user_features = tfidf.transform([text]).toarray()
            user_features_svd = svd.transform(user_features)

            # Use the sklearn classifier to predict the category
            prediction = model.predict(user_features_svd)
            prediction_label = le.inverse_transform(prediction)

            st.write("The predicted category is:", prediction_label[0])


elif page == "Test":
    st.title("Testing random text")

    col1, col2 = st.columns(2)

    with col1:
        if st.button('Test 100 random texts using my classifier'):
            # Select 100 random texts from the training data
            random_texts = data.sample(n=100)
            y_true, y_pred = generate_true_and_predicted_categories(random_texts)
            
            # Calculate the percentage of correct predictions
            correct_predictions = sum([true == pred for true, pred in zip(y_true, y_pred)])
            accuracy = correct_predictions / 100 * 100

            # Output the results
            st.info(f"Accuracy: {accuracy}%")

            # Plot confusion matrix
            plot_conf_matrix(y_true, y_pred)

    with col2:
        if st.button('Test 100 random texts using sklearn'):
            # Select 100 random texts from the training data
            random_texts = data.sample(n=100)
            y_true, y_pred = generate_true_and_predicted_categories_sklearn(random_texts)
            
            # Calculate the percentage of correct predictions
            correct_predictions = sum([true == pred for true, pred in zip(y_true, y_pred)])
            accuracy = correct_predictions / 100 * 100

            # Output the results
            st.info(f"Accuracy: {accuracy}%")

            # Plot confusion matrix
            plot_conf_matrix(y_true, y_pred)

elif page == "Visualize":
    st.title("Data visualization")

    # Calculate the number of articles in each category
    category_distribution = data['Category'].value_counts()

    # Visualization of article distribution by different categories
    st.subheader("Distribution of articles by categories")
    fig, ax = plt.subplots()
    sns.barplot(x=category_distribution.index, y=category_distribution.values, palette="viridis", ax=ax)
    ax.set_title('Distribution of articles by categories')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of articles')
    plt.xticks(rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.7)
    ax.set_ylim([0, 600])  # Set the limits of the Y axis
    st.pyplot(fig)

    # Visualization of article length distribution
    st.subheader("Distribution of article lengths")
    fig, ax = plt.subplots()
    sns.histplot(data['Article_Length'], bins=30, color="skyblue", kde=True, ax=ax)
    ax.set_title('Distribution of article lengths')
    ax.set_xlabel('Article length (number of words)')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.7)
    st.pyplot(fig)

    # Display the confusion matrix image
    st.subheader("Матриця неточностей")
    st.image("confusion.jpg")

elif page == "Feedback":
    st.title("Feedback")

    # Feedback form
    feedback = st.text_input("Enter your feedback here")
    if st.button("Send feedback"):
        # Save the feedback (for example, in a file or database)
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n")
        st.success("Thank you for your feedback!")