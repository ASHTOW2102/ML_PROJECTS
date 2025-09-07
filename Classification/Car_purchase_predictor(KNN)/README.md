# Car Purchase Predictor (K-Nearest Neighbors)

A simple ML demo that predicts whether a customer will purchase a car based on Age and Salary, implemented using K-Nearest Neighbors (KNN) and deployed on Hugging Face Spaces.

## Features

- K-Nearest Neighbors using scikit-learn
- Predicts categorical outcomes (Purchased: Yes or No)
- Uses Euclidean distance (p=2) to measure similarity
- Decision made by majority voting among k nearest neighbors
- Real-time prediction interface with Gradio
- Evaluation with Confusion Matrix and Accuracy Score for the test set

## Live Demo

[Car Purchase Predictor (KNN)](https://huggingface.co/spaces/AshishChaturvedi7/Car_purchase_predictorKNN)

## Model Overview

- KNN is a non-parametric, instance-based learning algorithm
- Classifies a new data point by looking at the k nearest neighbors
- Uses a distance metric (Euclidean or Minkowski) to find nearest points
- Final prediction is based on majority class vote
- Performance evaluated using Confusion Matrix and Accuracy Score

## Tech Stack

- Python 3.x
- scikit-learn
- Gradio
- pandas, numpy

## Run Locally

```bash
# Clone the repo
git clone https://github.com/ASHTOW2102/ML_PROJECTS.git

# Navigate to the project folder
cd ML_PROJECTS/Classification/Car_purchase_predictorKNN

# Install dependencies
pip install -r requirements.txt

# Launch the app
python main.py
```
