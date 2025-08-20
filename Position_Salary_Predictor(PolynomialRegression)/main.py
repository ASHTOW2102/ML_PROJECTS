import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ====== Load dataset ======
dataset = pd.read_csv("Position_Salaries.csv")  
# Expected columns: Position, Level, Salary

# Mapping Position → Level
position_to_level = dict(zip(dataset["Position"], dataset["Level"]))

# ====== Functions ======
def get_level_from_position(position):
    """Given a position, return its numeric level."""
    return position_to_level.get(position, None)

def predict_with_degree(position, degree):
    """Predict salary for given position and show actual vs predicted table."""
    # Ensure degree is int
    try:
        degree = int(degree)
    except:
        return "Degree must be integer", None, None

    # Prepare data
    X = dataset[["Level"]]
    y = dataset["Salary"]
    
    # Train polynomial regression with custom degree
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    # Predict for all positions
    predicted_all = lin_reg.predict(X_poly)

    # Predict for selected position
    level = get_level_from_position(position)
    if level is None:
        return "Position not found", None, None
    predicted_salary = lin_reg.predict(poly_reg.transform([[level]]))[0]

    # Create results dataframe
    results_df = pd.DataFrame({
        "Position": dataset["Position"],
        "Actual Salary": dataset["Salary"],
        "Predicted Salary": predicted_all.round(2)
    })

    return f"${predicted_salary:,.2f}", level, results_df

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
                    '<div class="subtitle">Polynomial Regression (Custom Degree)</div>')

        position_input = gr.Dropdown(
            choices=dataset["Position"].tolist(),
            label="Select Position",
            value=default_position
        )

        degree_input = gr.Number(label="Polynomial Degree", value=4, precision=0)

        level_output = gr.Number(label="Level", interactive=False, value=default_level)

        salary_output = gr.Textbox(label="Predicted Salary",
                                   elem_id="profit-output",
                                   interactive=False,
                                   value="")

        results_table = gr.Dataframe(headers=["Position", "Actual Salary", "Predicted Salary"],
                                     label="Actual vs Predicted Salaries")

        predict_btn = gr.Button("Predict Salary", elem_id="predict-button")

        # Auto-update level from dropdown
        position_input.change(lambda pos: get_level_from_position(pos),
                              position_input, level_output)

        # Predict salary + table
        predict_btn.click(fn=predict_with_degree,
                          inputs=[position_input, degree_input],
                          outputs=[salary_output, level_output, results_table])

if __name__ == "__main__":
    demo.launch(share=True)