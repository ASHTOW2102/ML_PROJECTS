# 💼 Position Salary Predictor (Random Forest)

Predict salaries based on job position levels using **Random Forest Regression**.  
A minimal, interactive ML demo hosted on Hugging Face Spaces.

---

## 📌 Features

- Random Forest Regression with scikit-learn
- Multiple decision trees combined for stable and accurate predictions
- Interactive UI using Gradio
- Real-time salary predictions
- Avoids overfitting by averaging across many trees
- Great for understanding **ensemble learning**

---

## 🚀 Live Demo

👉 Try it out on Hugging Face: [Position Salary Predictor (Random Forest)]((https://huggingface.co/spaces/AshishChaturvedi7/RegressionTree))

---

## 📈 Model Overview

This model uses **Random Forest Regression**, which is an **ensemble method**.  
Instead of relying on a single decision tree, Random Forest builds multiple trees and averages their predictions.

Key points:

- `n_estimators` → number of trees in the forest (more trees generally increase accuracy but may cause overfitting if too large).
- Helps reduce variance and improves generalization.
- Performance evaluated using **R² score**.

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
cd ML_PROJECTS/Position_Salary_Predictor(RandomeForestTree)

# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```
