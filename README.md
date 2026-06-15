# Guna-Titanic-Survival-Prediction-Using-ML--2026
A machine Learning Project that predicts passenger survival on the titanic dataset using classification algorithms and model evaluation techniques

## Project Overview

This project uses Machine Learning techniques to predict whether a passenger survived the Titanic disaster based on passenger information such as age, gender, ticket fare, and passenger class.

The project demonstrates the complete Machine Learning workflow including data preprocessing, model training, evaluation, and performance visualization.

## Objective

To build a predictive model capable of classifying passenger survival using historical Titanic dataset records.

## Dataset Information

Dataset: Titanic Passenger Dataset

Features Used:

- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Fare

Target Variable:

- Survived
  - 0 = Did Not Survive
  - 1 = Survived
  
## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Git & GitHub
  
## Project Workflow

1. Load Titanic Dataset
2. Data Cleaning and Preprocessing
3. Handle Missing Values
4. Convert Categorical Data to Numerical Format
5. Split Dataset into Training and Testing Sets
6. Train Random Forest Classifier
7. Evaluate Model Performance
8. Generate Visualizations
9. Analyze Feature Importance

## Machine Learning Algorithm

### Random Forest Classifier

Random Forest combines multiple Decision Trees and makes predictions using majority voting.

Reasons for choosing Random Forest:

- High Accuracy
- Handles Complex Patterns
- Reduces Overfitting
- Reliable for Classification Tasks

## Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | 79.33% |
| Model Type | Random Forest |
| Problem Type | Classification |

## Visualizations Generated

### Confusion Matrix

Evaluates prediction performance by comparing actual and predicted values.

### ROC Curve

Measures the model's ability to distinguish between survival and non-survival classes.

### Feature Importance

Identifies which passenger attributes contributed most to the prediction.

## Key Insights

- Gender was the most influential factor in survival prediction.
- Passenger Fare significantly impacted survival chances.
- First-class passengers had higher survival rates.
- Age also contributed to prediction performance.

## Project Structure

```text
Guna-Titanic-Survival-Prediction-Using-ML
│
├── train.csv
├── predictive_model.py
├── README.md
├── confusion_matrix.png
├── roc_curve.png
└── feature_importance.png
```

## Future Improvements

- Compare Multiple ML Models
- Hyperparameter Tuning
- Deploy Using Streamlit
- Create Interactive Dashboard
- Improve Prediction Accuracy

## Author

Guna Soundhari S

Aspiring Data Scientist | Machine Learning Enthusiast

## Project Outcome

Successfully developed a Machine Learning classification model capable of predicting Titanic passenger survival with nearly 80% accuracy while gaining hands-on experience in data preprocessing, model training, evaluation, and visualization.
