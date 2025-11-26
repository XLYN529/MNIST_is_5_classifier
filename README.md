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

### 1. Generate the Model
Since the trained model file is too large for GitHub, you must generate it locally:
1. Open `5_classifier.ipynb`.
2. Run all cells.
3. The notebook will save the trained model as `mnist_random_forest_precision_99.pkl` in your folder.

### 2. Load the Model (After Generation)
Once you have run the notebook, you can use the model in your own scripts:

```python
import joblib

# Load the model you just generated
model = joblib.load("mnist_random_forest_precision_99.pkl")

# Predict on new data
# Note: This model is tuned for High Precision (99%)
prediction = model.predict(new_image_data)
