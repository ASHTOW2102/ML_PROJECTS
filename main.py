# app.py for Streamlit
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1]  # Features (YearsExperience)
y = dataset.iloc[:, -1]   # Target (Salary)

# Split dataset (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train mode
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Linear Regression: Predict Salary")

# Input from user
years_exp = st.number_input("Total Experience (Years)", min_value=0.0, max_value=50.0, step=0.01, format="%.2f")

if st.button("Predict"):
    predicted_salary = model.predict([[years_exp]])[0]
    st.markdown(f"### Predicted Salary: ${predicted_salary:,.2f}")