import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Налаштування сторінки
st.set_page_config(page_title="Bank Churn Prediction", layout="wide")

# Заголовок
st.title("🏦 Прогнозування відтоку клієнтів банку")
st.markdown("Цей веб-додаток аналізує дані клієнтів та прогнозує, чи залишить клієнт банк (Churn).")

# 1. Завантаження даних (з кешуванням, щоб не вантажити щоразу)
@st.cache_data
def load_data():
    # Переконайтеся, що файл лежить у тій самій папці
    df = pd.read_csv('Churn_Modelling.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Файл 'Churn_Modelling.csv' не знайдено. Будь ласка, додайте його в папку з проектом.")
    st.stop()

# 2. Навчання моделі (з кешуванням)
@st.cache_resource
def train_model(data):
    # Видаляємо непотрібні колонки
    X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = data['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Визначаємо типи колонок
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender']
    
    # Створюємо Pipeline для обробки та моделі
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Метрики
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }
    
    return model, metrics

model, metrics = train_model(df)

# Меню збоку
st.sidebar.title("Навігація")
page = st.sidebar.radio("Оберіть розділ:", ["📊 Аналіз даних", "⚙️ Метрики моделі", "🔮 Зробити прогноз"])

# --- Розділ 1: Аналіз даних ---
if page == "📊 Аналіз даних":
    st.header("Розвідувальний аналіз даних (EDA)")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Розподіл відтоку клієнтів")
        fig1 = px.pie(df, names='Exited', title='0 - Залишився, 1 - Пішов', color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Вік клієнтів та відтік")
        fig2 = px.histogram(df, x="Age", color="Exited", barmode="overlay", color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig2, use_container_width=True)

# --- Розділ 2: Метрики моделі ---
elif page == "⚙️ Метрики моделі":
    st.header("Оцінка якості моделі (Random Forest)")
    st.markdown("Модель натренована на 80% даних. Результати перевірки на тестовій вибірці:")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy (Точність)", f"{metrics['Accuracy']:.2%}")
    col2.metric("Precision (Влучність)", f"{metrics['Precision']:.2%}")
    col3.metric("Recall (Повнота)", f"{metrics['Recall']:.2%}")
    
    st.info("💡 Зверніть увагу: Recall часто є найважливішою метрикою для бізнесу, адже нам важливо не пропустити жодного клієнта, який збирається піти.")

# --- Розділ 3: Зробити прогноз ---
elif page == "🔮 Зробити прогноз":
    st.header("Прогноз для нового клієнта")
    st.markdown("Введіть дані клієнта, щоб дізнатися ймовірність його відтоку.")
    
    # Форма для введення даних
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Кредитний рейтинг", min_value=300, max_value=900, value=650)
            geography = st.selectbox("Країна", ['France', 'Spain', 'Germany'])
            gender = st.selectbox("Стать", ['Male', 'Female'])
            age = st.number_input("Вік", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Кількість років з банком", min_value=0, max_value=10, value=5)
            
        with col2:
            balance = st.number_input("Баланс на рахунку ($)", min_value=0.0, value=50000.0)
            num_products = st.number_input("Кількість продуктів банку", min_value=1, max_value=4, value=2)
            has_cr_card = st.selectbox("Чи має кредитну картку?", [1, 0])
            is_active = st.selectbox("Чи є активним клієнтом?", [1, 0])
            estimated_salary = st.number_input("Очікувана зарплата ($)", min_value=0.0, value=60000.0)
            
        submit_button = st.form_submit_button(label="Прогнозувати")
        
    # Блок обробки результату
    if submit_button:
        # Формуємо DataFrame з введених даних
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active],
            'EstimatedSalary': [estimated_salary]
        })
        
        # Отримуємо ймовірність відтоку (клас 1)
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        
        # Розподіл на 3 рівні ризику на основі ймовірності
        if prediction_proba >= 0.60:
            st.error(f"🔴 **Високий ризик!** Клієнт з великою ймовірністю залишить банк. (Ймовірність відтоку: {prediction_proba:.1%})")
        elif prediction_proba >= 0.30:
            st.warning(f"🟡 **Середній ризик.** Рекомендується запропонувати кращі умови або бонус. (Ймовірність відтоку: {prediction_proba:.1%})")
        else:
            st.success(f"🟢 **Низький ризик.** Клієнт ймовірно залишиться. (Ймовірність відтоку: {prediction_proba:.1%})")
