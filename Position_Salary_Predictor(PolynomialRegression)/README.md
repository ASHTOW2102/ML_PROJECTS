
> Predict salaries based on job position level using **Polynomial Regression**  
> A minimal, interactive ML demo hosted on Hugging Face Spaces

---

## 📌 Features

- 🔹 Polynomial Regression with `scikit-learn`
- 🔹 Interactive UI using Gradio
- 🔹 Real-time salary predictions
- 🔹 Ideal for beginners & educational purposes

---

## 🚀 Live Demo

👉 Try it out on [Hugging Face](https://huggingface.co/spaces/ASHCHAT/Position_Salary_Predictor)

---

## 📈 Model Overview

The model is based on the polynomial equation:

\[
\text{salary} = a_0 + a_1 \cdot x + a_2 \cdot x^2 + \dots + a_n \cdot x^n
\]

Where:

- \( x \): position level
- \( a_i \): coefficients determined during training

Trained using **Ordinary Least Squares (OLS)** to minimize prediction error.

---

## ⚙️ Tech Stack

- Python 3.x
- `scikit-learn`
- `gradio`
- `pandas`, `numpy`

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/ASHTOW2102/ML_PROJECTS.git

# Navigate to project folder
cd ML_PROJECTS/Position_Salary_Predictor

# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```
