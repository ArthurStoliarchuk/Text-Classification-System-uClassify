import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Завантаження даних
train_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Train.csv")
test_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Test.csv")

# Перетворення тексту в числові вектори
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', ngram_range=(1, 2),
                        min_df=5, norm='l2', encoding='latin-1', lowercase=True)
train_features = tfidf.fit_transform(train_data['Text']).toarray()
test_features = tfidf.transform(test_data['Text']).toarray()

# Збереження TfidfVectorizer
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Зменшення розмірності за допомогою SVD
svd = TruncatedSVD(n_components=5, random_state=0)
train_features_svd = svd.fit_transform(train_features)
test_features_svd = svd.transform(test_features)

# Збереження TruncatedSVD
with open('svd.pkl', 'wb') as f:
    pickle.dump(svd, f)

# Перетворення міток категорій у числові мітки
le = LabelEncoder()
train_labels = le.fit_transform(train_data['Category'])

# Збереження LabelEncoder
with open('le.pkl', 'wb') as f:
    pickle.dump(le, f)

# Тренування моделі
model = LogisticRegression(random_state=0)
model.fit(train_features_svd, train_labels)

# Збереження моделі
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Передбачення на тренувальних даних
train_pred = model.predict(train_features_svd)

# Обчислення точності
train_acc = accuracy_score(train_labels, train_pred)
print("Точність на тренувальних даних:", train_acc)

# Обчислення матриці неточностей
conf_mat = confusion_matrix(train_labels, train_pred)

# Виведення матриці неточностей
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()