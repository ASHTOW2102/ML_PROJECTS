
# 🧠 SalaryScout

> Predict salary based on experience using Simple Linear Regression  
> A minimal, interactive ML demo hosted on Hugging Face Spaces

---

## 📌 Features

- 🔹 Simple Linear Regression with `scikit-learn`
- 🔹 Interactive UI using Gradio
- 🔹 Real-time salary predictions
- 🔹 Ideal for beginners & educational purposes

---

## 🚀 Live Demo

👉 Try it out on [Hugging Face](https://huggingface.co/spaces/ASHCHAT/SalaryScout)

---

## 📈 Model Overview

The model is based on the equation:  
salary = m * experience + c


- `m`: slope (increment per year of experience)  
- `c`: base salary with zero experience  

Trained using ordinary least squares (OLS) to minimize prediction error.

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
git clone https://github.com/ASHTOW2102/Salary_Predictor.git

# Navigate to project folder
cd Salary_Predictor/Salary_Predictor(SimpleLinearRegression)

# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```

📊 Sample Input/Output
| Experience (years) | Predicted Salary |
| ------------------ | ---------------- |
| 1                  | ₹30,000          |
| 5                  | ₹55,000          |
| 10                 | ₹80,000          |


