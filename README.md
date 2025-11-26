# MNIST Binary Classifier (Digit 5)

A machine learning project that classifies handwritten digits from the MNIST dataset. The goal was to build a binary classifier capable of distinguishing the number '5' from all other digits (0-4, 6-9), with a specific focus on maximizing Precision to minimize False Positives.

## Project Overview

* **Dataset:** MNIST (70,000 handwritten digit images).
* **Problem Type:** Binary Classification (Imbalanced Dataset).
* **Final Model:** Random Forest Classifier.
* **Optimization Strategy:** Hyperparameter tuning via RandomizedSearchCV and custom threshold adjustment to achieve >99% Precision.

## Key Results

The model was evaluated on the unseen Test Set (10,000 images) using a custom decision threshold to prioritize "trustworthiness" (Precision) over raw coverage (Recall).

* **Precision:** 99.24% (The model rarely misidentifies non-5s).
* **Recall:** 88.00% (The model captures most valid 5s but misses ambiguous ones).
* **Accuracy:** 98.87%

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/mnist-binary-classifier.git](https://github.com/YOUR_USERNAME/mnist-binary-classifier.git)
    cd mnist-binary-classifier
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Analysis
Open `5_classifier.ipynb` in Jupyter Notebook or VS Code to see the full training pipeline, including:
* Data preprocessing and exploration.
* Model comparison (SGD vs. Logistic Regression vs. Random Forest).
* Hyperparameter tuning using RandomizedSearchCV.
* Precision-Recall curve analysis.

### Loading the Trained Model
The final model is saved as a serialized object. You can load it in Python to make predictions without retraining:

```python
import joblib

# Load the model
model = joblib.load("mnist_random_forest_precision_99.pkl")

# Predict on new data (expects 28x28 flattened array)
# Note: Ensure you apply the custom threshold logic if needed, 
# though this specific model file was saved with the best estimator settings.
prediction = model.predict(new_image_data)