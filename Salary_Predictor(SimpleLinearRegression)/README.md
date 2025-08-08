
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

# Clone the repo
git clone https://huggingface.co/spaces/ASHCHAT/SalaryScout

# Navigate into project
cd SalaryScout

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py

📊 Sample Input/Output
| Experience (years) | Predicted Salary |
| ------------------ | ---------------- |
| 1                  | ₹30,000          |
| 5                  | ₹55,000          |
| 10                 | ₹80,000          |


