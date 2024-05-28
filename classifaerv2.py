import streamlit as st
from uclassify import uclassify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Створюємо екземпляр класу uclassify
a = uclassify()

# Встановлюємо ключі
a.setWriteApiKey('cKrbq726LvrT') 
a.setReadApiKey('cnytszVmNdlu') 

# Визначаємо категорії
categories = ["business", "entertainment", "politics", "sport", "tech"]

# Функція для класифікації тексту
def classify_text(text_to_classify):
    try:
        # Класифікуємо текст
        d = a.classify([text_to_classify], "TextClassifier5topics")

        # Нормалізуємо оцінки та перетворюємо їх у відсотки
        scores = [float(score) for category, score in d[0][2]]
        total_score = sum(scores)
        percentages = [(score / total_score) * 100 for score in scores]

        # Знаходимо категорію з найбільшим відсотком
        max_percentage_index = percentages.index(max(percentages))
        max_category = d[0][2][max_percentage_index][0]

        # Відповідність категорій та емодзі
        emojis = {
            "business": "👩‍💼",
            "entertainment": "🍿",
            "politics": "✊🏼",
            "sport": "🏃🏾🤸",
            "tech": "🤖"
        }

        # Виводимо результати класифікатора
        result = f"{max_category.capitalize()} - {percentages[max_percentage_index]:.2f}% {emojis[max_category]}"

        return result, max_category
    except Exception as e:
        st.error(str(e))

# Список сторінок
pages = ["Прогнозування категорії", "Тестування випадкового тексту", "Порожня сторінка"]

# Вибір сторінки
page = st.sidebar.selectbox("Оберіть сторінку", pages)

if page == "Прогнозування категорії":
    st.title("Система класифікації тексту💻")

    text = st.text_area("Введіть текст для класифікації")

   
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Прогнозувати за допомогою мого класифікатора'):
            if "results" not in st.session_state:
                st.session_state.results = []
            result, predicted_category = classify_text(text)
            st.session_state.results.append(f"Прогнозована категорія - {result}")
            for result in st.session_state.results:
                st.info(result)

elif page == "Тестування випадкового тексту":
    st.title("Тестування випадкового тексту")

    if st.button('Тестувати випадковий текст'):
        # Читаємо тестові дані
        test_data = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Test.csv")
        sample_solution = pd.read_csv("C:\\Users\\clash\\KursovayaTEXT_1\\bbc2\\BBC News Sample Solution.csv")
        # Об'єднуємо два DataFrame за стовпцем 'ArticleId'
        test_data = pd.merge(test_data, sample_solution, on='ArticleId')
        # Вибираємо випадкову категорію
        random_category = np.random.choice(categories)
        # Вибираємо випадковий текст з цієї категорії
        random_text = test_data[test_data['Category'] == random_category]['Text'].sample(n=1).values[0]
        # Класифікуємо випадковий текст
        result, predicted_category = classify_text(random_text)
        # Знаходимо справжню категорію
        actual_category = random_category
        # Виводимо результати
        st.info(f"Прогнозована категорія - {result}")
        st.info(f"Справжня категорія - {actual_category}")

elif page == "Порожня сторінка":
    st.title("Порожня сторінка")
    # Тут ви можете додати свій код