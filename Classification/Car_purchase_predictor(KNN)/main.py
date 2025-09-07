import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ====== Load dataset ======
dataset = pd.read_csv("Car_Purchased.csv")  
# Expected columns: Age, Salary, Purchased (0/1)

X = dataset[["Age", "EstimatedSalary"]].values
y = dataset["Purchased"].values

# ====== Train/Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ====== Feature Scaling ======
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ====== Train KNN Classifier ======
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# ====== Model Evaluation (Test Set Only) ======
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


# ====== Prediction Function (User Input: Age & Salary) ======
def predict_purchase(age, salary):
    """Predict if user will purchase a car given age and salary."""
    features = sc.transform([[age, salary]])
    prediction = classifier.predict(features)[0]
    prob = classifier.predict_proba(features)[0][1]  # Probability of purchase
    return (
        "Yes 🚗" if prediction == 1 else "No ❌",
        f"{prob*100:.2f}%",
        accuracy,
        cm.tolist()
    )


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
#output-box textarea {
    background: transparent;
    color: #185a9d;
    font-weight: 800;
    text-align: center;
    font-size: 1.5rem;
    border: none;
}
""") as demo:

    with gr.Column(elem_id="main-card"):
        gr.Markdown('<div class="heading">Car Purchase Prediction</div>'
                    '<div class="subtitle">KNN Classifier Model (Test Set Evaluation)</div>')

        age_input = gr.Number(label="Age", value=30)
        salary_input = gr.Number(label="Salary", value=50000)

        purchase_output = gr.Textbox(label="Prediction (Purchased?)", elem_id="output-box")
        prob_output = gr.Textbox(label="Purchase Probability")
        acc_output = gr.Textbox(label="Model Accuracy (Test Set)")
        cm_output = gr.Dataframe(headers=["Predicted 0", "Predicted 1"], label="Confusion Matrix")

        predict_btn = gr.Button("Predict Purchase", elem_id="predict-button")

        predict_btn.click(fn=predict_purchase,
                          inputs=[age_input, salary_input],
                          outputs=[purchase_output, prob_output, acc_output, cm_output])

if __name__ == "__main__":
    demo.launch(share=True)
