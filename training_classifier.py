import pandas as pd
from uclassify import uclassify


a = uclassify()


a.setWriteApiKey('cKrbq726LvrT') 
a.setReadApiKey('cnytszVmNdlu') 


categories = ["business", "entertainment", "politics", "sport", "tech"]

# Если классификатор не существует, создаем его и добавляем классы
try:
    a.train(["dummy text"], "business", "TextClassifier5topics")
except:
    a.create("TextClassifier5topics")
    a.addClass(categories, "TextClassifier5topics")


train_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Train.csv")


for category in categories:
    texts = train_data[train_data['Category'] == category]['Text']
    a.train(texts, category, "TextClassifier5topics")
    print(f"Trained on {len(texts)} texts for category {category}")