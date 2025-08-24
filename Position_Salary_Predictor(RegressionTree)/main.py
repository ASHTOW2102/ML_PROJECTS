import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# ====== Load dataset ======
dataset = pd.read_csv("Position_Salaries.csv")  
# Expected columns: Position, Level, Salary

# Mapping Position → Level
position_to_level = dict(zip(dataset["Position"], dataset["Level"]))

# ====== Functions ======
def get_level_from_position(position):
    """Given a position, return its numeric level."""
    return position_to_level.get(position, None)

def predict_with_forest(position):
    """Predict salary for given position using Random Forest Regression with test set evaluation."""
    # Prepare data
    X = dataset[["Level"]].values
    y = dataset["Salary"].values

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train Random Forest Regressor with 10 estimators
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)

    # Calculate R² on test set
    r2 = r2_score(y_test, y_pred_test)

    # Predict for selected position
    level = get_level_from_position(position)
    if level is None:
        return "Position not found", None, None, None
    predicted_salary = regressor.predict([[level]])[0]

    # Create results dataframe (only training + test merged for visualization)
    results_df = pd.DataFrame({
        "Level": X.ravel(),
        "Position": dataset["Position"],
        "Actual Salary": y,
        "Predicted Salary (Train/Test)": regressor.predict(X).round(2)
    })

    return f"${predicted_salary:,.2f}", level, results_df, f"R² Score (Test Set): {r2:.4f}"


# ====== Defaults ======
default_position = dataset["Position"].iloc[0]
default_level = position_to_level[default_position]

# ====== Gradio UI ======
with gr.Blocks(css=""" 
body {
    background: radial-gradient(circle at top left, #43cea2 0%, #185a9d 100%);
    min-height: 100vh;
}
#main-card {
    background: rgba(255,255,255,0.95);
    border-radius: 28px;
    padding: 42px 30px 32px 30px;
    max-width: 700px;
    margin: 56px auto 0 auto !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}
.heading {
    color: #185a9d;
    font-size: 1.8rem;
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
        gr.Markdown('<div class="heading">Salary Prediction</div>'
                    '<div class="subtitle">Random Forest Regression (10 Trees)</div>')

        position_input = gr.Dropdown(
            choices=dataset["Position"].tolist(),
            label="Select Position",
            value=default_position
        )

        level_output = gr.Number(label="Level", interactive=False, value=default_level)

        salary_output = gr.Textbox(label="Predicted Salary",
                                   elem_id="profit-output",
                                   interactive=False,
                                   value="")

        results_table = gr.Dataframe(headers=["Level", "Position", "Actual Salary", "Predicted Salary (Train/Test)"],
                                     label="Actual vs Predicted Salaries")

        r2_output = gr.Textbox(label="Model Performance (R² Score on Test Set)",
                               interactive=False,
                               value="")

        predict_btn = gr.Button("Predict Salary", elem_id="predict-button")

        # Auto-update level from dropdown
        position_input.change(lambda pos: get_level_from_position(pos),
                              position_input, level_output)

        # Predict salary + table + R²
        predict_btn.click(fn=predict_with_forest,
                          inputs=[position_input],
                          outputs=[salary_output, level_output, results_table, r2_output])

if __name__ == "__main__":
    demo.launch(share=True)
