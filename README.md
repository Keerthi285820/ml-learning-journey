📌 Project Title

Titanic Dataset — Machine Learning Models (Week 1)

📊 Overview

This project is part of my machine learning learning journey where I applied fundamental ML models on a real-world dataset.

The goal was to understand how different models behave, how to preprocess data, and how to evaluate results properly.

📁 Dataset
Titanic dataset
Contains passenger details like age, gender, class, and survival status
⚙️ Models Used
🔹 1. Linear Regression
Used to predict Fare (continuous value)
Applied log transformation to handle skewed data
Evaluated using Mean Squared Error (MSE)

👉 Insight:
Model performance improved significantly after handling skewness in data

🔹 2. Logistic Regression
Used for classification (Survived: Yes/No)
Evaluated using Accuracy & Confusion Matrix

👉 Insight:
Model achieved very high accuracy, indicating strong patterns in the dataset

🔹 3. K-Nearest Neighbors (KNN)
Distance-based classification model
Applied feature scaling for better performance

👉 Insight:
Produced similar predictions to Logistic Regression, showing strong feature influence

🔧 Key Steps
Data Cleaning (handling missing values)
Feature Encoding (categorical → numerical)
Feature Transformation (log transformation for skewed data)
Feature Scaling (for KNN)
Model Training & Evaluation
📈 Results
Model	Performance
Linear Regression	Low MSE after transformation
Logistic Regression	High Accuracy
KNN	High Accuracy (after scaling)
🧠 Key Learnings
Data preprocessing has a huge impact on model performance
Different models can give similar outputs when data has strong patterns
Evaluation is not just about accuracy, but also understanding model behavior
