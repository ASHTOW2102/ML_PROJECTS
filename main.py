import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dummy fallback for illustration if you lack Salary_Data.csv
# Remove/comment this and use your own CSV file in production!
import numpy as np
np.random.seed(1)
dataset = pd.DataFrame({
    "YearsExperience": np.linspace(1, 10, 30),
    "Salary": np.linspace(40000, 120000, 30) + np.random.normal(0, 3000, 30)
})

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)

def predict_salary(years_experience):
    predicted_salary = model.predict([[years_experience]])[0]
    return f"${predicted_salary:,.2f}"

with gr.Blocks(css="""
body {
    background: radial-gradient(circle at top left, #fd5c63 0%, #2d50e6 100%);
    min-height: 100vh;
}
#main-card {
    background: rgba(33,39,62,0.95);
    border-radius: 28px;
    padding: 42px 30px 32px 30px;
    max-width: 400px;
    margin: 56px auto 0 auto !important;
    box-shadow: 0 8px 32px 0 rgba(60,70,130,0.14);
}
.heading {
    color: #fff;
    font-size: 1.8rem !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    font-weight: 800;
    text-align: left;
    letter-spacing: 0.01em;
    margin-bottom: 6px;
}
.subtitle {
    color: #d2d8ef;
    font-size: 1.08rem;
    margin-bottom: 32px;
}
#years-exp input {
    font-size: 1.25rem;
    padding: 10px 16px;
    border: 1.5px solid #6d7eff;
    border-radius: 14px;
    background: #242850;
    color: #fff;
    font-weight: 500;
}
#predict-button {
    background: linear-gradient(90deg, #fd5c63 0%, #3a71de 100%);
    color: #fff;
    font-weight: 700;
    border-radius: 14px;
    padding: 14px;
    font-size: 1.18rem;
    border: none;
    margin: 28px 0 0 0;
    box-shadow: 0 2px 8px rgba(45,80,230,0.08);
}
#predict-button:hover {
    filter: brightness(1.17);
}
#salary-output textarea {
    background: transparent;
    color: #74f8d9;
    font-weight: 800;
    text-align: left;
    font-size: 2.1rem;
    border: none;
    padding-left: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
""") as demo:
    with gr.Column(elem_id="main-card"):
        gr.Markdown(
            '<div class="heading">SALARY PREDICTION</div>'
            '<div class="subtitle">Linear regression model</div>',
        )
        years_exp = gr.Number(label="Years of Experience", value=5, minimum=0, maximum=50, precision=0, elem_id='years-exp')
        predict_btn = gr.Button("Predict", elem_id="predict-button")
        output = gr.Textbox(label="Predicted Salary", elem_id="salary-output", show_label=True)
        predict_btn.click(fn=predict_salary, inputs=years_exp, outputs=output)
        
if __name__ == "__main__":
    demo.launch(share=True)
