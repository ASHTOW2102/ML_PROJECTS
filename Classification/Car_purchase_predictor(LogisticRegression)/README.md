# Car Purchase Predictor (Logistic Regression)

A simple, interactive ML demo that predicts whether a customer will purchase a car based on **Age** and **Salary**, implemented using **Logistic Regression** and deployed on Hugging Face Spaces.

---

## Features

- **Logistic Regression** using scikit-learn
- Predicts categorical outcomes (Purchased: Yes or No)
- Uses a **sigmoid function** to map model outputs to probabilities between 0 and 1
- Predictions based on maximum likelihood estimation
- Real-time prediction interface with Gradio
- Evaluation with **Confusion Matrix** and **Accuracy Score** displayed for the test set

---

## Live Demo

Check it out on Hugging Face:  
[Car Purchase Predictor (Logistic Regression)](https://huggingface.co/spaces/AshishChaturvedi7/LogisiticRegression)

---

## Model Overview

- Logistic Regression fits an **S-shaped (sigmoid) curve** to model the probability of purchase
- Optimizes parameters using **maximum likelihood estimation**
- Converts probabilities to predictions using a 0.5 threshold (≥ 0.5 → Yes, else No)
- Evaluates performance using a **Confusion Matrix** and **Accuracy Score**

---

## Tech Stack

- **Python 3.x**
- **scikit-learn**
- **Gradio**
- **pandas**, **numpy**

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/ASHTOW2102/ML_PROJECTS.git

# Navigate to the project folder
cd ML_PROJECTS/Classification/Car_purchase_predictor(LogisticRegression)

# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```
