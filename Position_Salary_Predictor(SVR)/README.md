# 💼 Position Salary Predictor (SVR)

Predict salaries based on job position levels using **Support Vector Regression (SVR)**.  
A minimal, interactive ML demo hosted on Hugging Face Spaces.

---

## 📌 Features

- SVR with scikit-learn
- Feature scaling applied to both input (levels) and output (salaries)
- Interactive UI using Gradio
- Real-time salary predictions
- Captures **non-linear salary patterns** using the **RBF kernel**
- Ideal for beginners & educational purposes

---

## 🚀 Live Demo

👉 Try it out on Hugging Face: [Position Salary Predictor (SVR)](https://huggingface.co/spaces/AshishChaturvedi7/Position_Salary_PredictorSVR)

---

## 📈 Model Overview

This model uses **Support Vector Regression (SVR)** with the **RBF (Radial Basis Function) kernel**, which is effective for modeling non-linear relationships.

Equation of kernel function:

\[
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
\]

- \(x\): position level
- \(K(x_i, x_j)\): similarity between points
- RBF helps the model adapt smoothly to complex salary curves

---

## ⚙️ Tech Stack

- Python 3.x
- scikit-learn
- gradio
- pandas, numpy

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/ASHTOW2102/ML_PROJECTS.git

# Navigate to project folder
cd ML_PROJECTS/Position_Salary_Predictor(SVR)

# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```
