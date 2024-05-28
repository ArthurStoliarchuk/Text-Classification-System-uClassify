import random
import pandas as pd
from uclassify import uclassify


a = uclassify()

a.setReadApiKey('cnytszVmNdlu') 


test_data = pd.read_csv("bbc2/BBC News Test.csv")
solution_data = pd.read_csv("bbc2/BBC News Sample Solution.csv")


random_index = random.randint(0, len(test_data) - 1)
article_id = test_data.loc[random_index, 'ArticleId']
text = test_data.loc[random_index, 'Text']


text = text[:1000]


classifier_name = "TextClassifier5topics"
print("Starting classification...")
prediction = a.classify(a.readApiKey, classifier_name, text)
print("Classification finished.")


real_category = solution_data[solution_data['ArticleId'] == article_id]['Category'].values[0]

print(f"Predicted category for article {article_id}: {prediction}")
print(f"Real category for article {article_id}: {real_category}")