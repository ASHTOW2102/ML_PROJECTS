# 📊 Multiple Linear Regression – Real Estate Profit Prediction

This project demonstrates **Multiple Linear Regression** using Python to predict **profit** based on multiple business parameters such as **R&D Spend, Administration, and Marketing Spend**.  

Unlike **Simple Linear Regression** (which uses only one input feature), Multiple Linear Regression considers **multiple inputs** to better understand the relationship between independent variables and the dependent variable.

---

## 🚀 What’s New in This Project?

In the previous project, we explored **Simple Linear Regression**.  
This time:
- We are using **Multiple Linear Regression** with **3 input features** and **1 output**.
- We apply **OneHotEncoding** on categorical data.
- We introduce **Backward Elimination** to remove statistically insignificant features based on **p-values**.
- Finally, we compare **Actual vs Predicted** profit values.

---

## 🛠️ Tech Stack & Libraries
- **Python 3.x**
- `numpy`
- `pandas`
- `scikit-learn`

---

## 📂 Dataset
The dataset contains:
- **R&D Spend** – Amount invested in research & development
- **Administration** – Administrative expenses
- **Marketing Spend** – Marketing budget
- **State** – Categorical variable representing location
- **Profit** – Target variable to predict

---

## ⚡ How It Works
1. **Data Preprocessing**
   - Handle missing data (if any)
   - Encode categorical data using **ColumnTransformer + OneHotEncoder**
   - Avoid dummy variable trap

2. **Model Training**
   - Train Multiple Linear Regression model using `LinearRegression` from `sklearn`

3. **Backward Elimination**
   - Use `statsmodels` to find features with **p-value > 0.05** and remove them

4. **Prediction**
   - Compare **Actual vs Predicted** values

---

## 🖥️ Run Locally
```bash
# Clone the repository
git clone https://github.com/ASHTOW2102/ML_PROJECTS.git

# Navigate to the project directory
cd ML_PROJECTS/StartupProfitPredictor\(MultipleLinearRegression\)

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # on Unix-based systems

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
