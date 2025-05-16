# Automated ML Pipeline

This repository contains an end-to-end automated machine learning pipeline implemented using Python and GitHub Actions. The pipeline covers data preprocessing, model training, testing, and artifact management.

---

## Project Overview

The pipeline performs the following tasks:

1. **Data Loading**  
   Loads a public dataset (e.g., Iris dataset from `sklearn.datasets`).

2. **Data Preprocessing**  
   Handles missing values, feature scaling, and prepares data for training.

3. **Model Training**  
   Trains a classification model (Logistic Regression) using scikit-learn.

4. **Model Testing**  
   Includes unit tests for preprocessing functions and model performance checks.

5. **Model Saving**  
   Saves the trained model artifact (`model.pkl`) using `joblib`.

6. **Continuous Integration with GitHub Actions**  
   Automates the pipeline execution on every push or pull request.  
   The workflow installs dependencies, runs tests, trains the model, and uploads the trained model as an artifact.
