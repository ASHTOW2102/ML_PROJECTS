import gradio as gr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

# Load dataset
dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LinearRegression()
model.fit(X_train, y_train)

def predict_salary_and_plot(years_experience):
    # Create DataFrame for prediction input with correct column name
    input_df = pd.DataFrame([[years_experience]], columns=X.columns)
    predicted_salary = model.predict(input_df)[0]

    # Generate prediction curve on a smooth range
    X_plot = pd.DataFrame(np.linspace(0, 50, 200), columns=X.columns)
    y_plot = model.predict(X_plot)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot actual dataset points
    ax.scatter(X, y, color='#ffa500', label='Actual Data', alpha=0.7)
    
    # Plot prediction line
    ax.plot(X_plot, y_plot, label='Prediction Line', color='#2a52be')
    
    # Highlight predicted salary for input years
    ax.scatter([years_experience], [predicted_salary], color='#db4545', label='Input Prediction', zorder=5)
    
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary ($)')
    ax.set_title('Salary Prediction vs Years of Experience')
    ax.legend()
    ax.grid(True)

    # Save plot to a bytes buffer and convert to PIL Image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    return f"${predicted_salary:,.2f}", img

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
            '<div class="subtitle">Linear regression model with graph</div>',
        )
        years_exp = gr.Number(label="Years of Experience", value=5, minimum=0, maximum=50, precision=0, elem_id='years-exp')
        predict_btn = gr.Button("Predict", elem_id="predict-button")
        output = gr.Textbox(label="Predicted Salary", elem_id="salary-output", interactive=False, show_label=True)
        plot_image = gr.Image(type="pil", label="Prediction Plot")
        
        predict_btn.click(fn=predict_salary_and_plot, inputs=years_exp, outputs=[output, plot_image])

if __name__ == "__main__":
    demo.launch(share=True)
