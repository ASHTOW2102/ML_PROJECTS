import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Profit

# Encode categorical column 'State'
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X_encoded = np.array(ct.fit_transform(X))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=1)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

def predict_profit_and_table(rnd_spend, admin_spend, marketing_spend, state):
    # Prepare input for prediction
    input_df = pd.DataFrame([[rnd_spend, admin_spend, marketing_spend, state]],
                            columns=["R&D Spend", "Administration", "Marketing Spend", "State"])
    input_encoded = np.array(ct.transform(input_df))
    predicted_profit = model.predict(input_encoded)[0]

    # Predictions for test set
    y_pred = model.predict(X_test)

    # Prepare actual vs predicted table
    results_df = pd.DataFrame({
        "Actual Profit": y_test.values,
        "Predicted Profit": np.round(y_pred, 2)
    }).reset_index(drop=True)

    return f"₹{predicted_profit:,.2f}", results_df

with gr.Blocks(css="""
body {
    background: radial-gradient(circle at top left, #43cea2 0%, #185a9d 100%);
    min-height: 100vh;
}
#main-card {
    background: rgba(255,255,255,0.95);
    border-radius: 28px;
    padding: 42px 30px 32px 30px;
    max-width: 500px;
    margin: 56px auto 0 auto !important;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.15);
}
.heading {
    color: #185a9d;
    font-size: 1.8rem !important;
    font-weight: 800;
    text-align: center;
    margin-bottom: 6px;
}
.subtitle {
    color: #444;
    font-size: 1.08rem;
    text-align: center;
    margin-bottom: 32px;
}
#predict-button {
    background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
    color: #fff;
    font-weight: 700;
    border-radius: 14px;
    padding: 14px;
    font-size: 1.18rem;
    border: none;
    margin: 28px 0 0 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
#predict-button:hover {
    filter: brightness(1.17);
}
#profit-output textarea {
    background: transparent;
    color: #185a9d;
    font-weight: 800;
    text-align: center;
    font-size: 2rem;
    border: none;
}
""") as demo:
    with gr.Column(elem_id="main-card"):
        gr.Markdown('<div class="heading">Startup Profit Prediction</div>'
                    '<div class="subtitle">Multiple Linear Regression</div>')
        rnd_input = gr.Number(label="R&D Spend", value=120000)
        admin_input = gr.Number(label="Administration Spend", value=90000)
        marketing_input = gr.Number(label="Marketing Spend", value=250000)
        state_input = gr.Dropdown(choices=dataset["State"].unique().tolist(), label="State")
        predict_btn = gr.Button("Predict Profit", elem_id="predict-button")
        output_text = gr.Textbox(label="Predicted Profit", elem_id="profit-output", interactive=False)
        results_table = gr.Dataframe(headers=["Actual Profit", "Predicted Profit"], label="Actual vs Predicted Profit")

        predict_btn.click(fn=predict_profit_and_table,
                          inputs=[rnd_input, admin_input, marketing_input, state_input],
                          outputs=[output_text, results_table])

if __name__ == "__main__":
    demo.launch(share=True)
