# Heart Disease Predictor
This project develops and evaluates several machine learning models to predict the presence of heart disease based on clinical measurements. The goal is to compare different algorithms, assess how reliable they are on a small medical dataset, and understand which features are most influential in the predictions.
---

## Project Overview

The workflow in this project is:

1. Load and inspect a cleaned heart disease dataset.
2. Explore the data with summary statistics, histograms, and a correlation heatmap.
3. Standardize features and split the data into a training set (250 patients) and a held‑out test set (52 patients).
4. Train five classification models:
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - XGBoost  
   - LightGBM
5. Evaluate all models on the test set using accuracy, precision, recall, and F1‑score.
6. Run 5‑fold cross‑validation to get more reliable performance estimates.
7. Perform hyperparameter tuning for Logistic Regression, Decision Tree, and Random Forest.
8. Evaluate the tuned models on the held‑out test set.
9. Generate visualizations: feature distributions, correlation heatmap, confusion matrices, ROC curves, accuracy and cross‑validation comparison charts, and a feature importance plot.
---

## Dataset Used

- **File:** `https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset`  
- **Samples:** 302 patient records  
- **Features:** 13 clinical features (e.g., age, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise‑induced angina, ST depression, etc.)  
- **Target:** `target`  
  - `1` = heart disease present  
  - `0` = no heart disease  
- **Data quality:**  
  - No missing values  
  - Class distribution is roughly balanced (about 54% with heart disease, 46% without)

This file is a cleaned version of the heart disease dataset, all the duplicate values have been removed.

---
## Methodology and Results

### Models

The following supervised learning models are trained and evaluated:

- Logistic Regression  
- Decision Tree
- Random Forest  
- XGBoost  
- LightGBM  

All models use standardized features. The first 250 rows are used for training and the remaining 52 rows are used as a held‑out test set.

### Evaluation

1. **Baseline test performance**

   Each model is trained on the training set and evaluated on the held‑out test set.

2. **5‑fold cross‑validation**

   To obtain a more robust estimate of performance on this small dataset, 5-fold cross-validation is performed on the full dataset for all models. 

3. **Hyperparameter tuning**

   Grid search with 5‑fold cross‑validation.
   
   The tuned versions are then evaluated on the held‑out test set and their test accuracies are reported.

### Key Visual Outputs

The script saves the following figures into the `outputs/` folder:

- Histograms for all numeric features
- Pearson correlation heatmap
- Confusion matrices for each model
- Combined ROC curves for all models
- Test accuracy comparison bar chart
- 5‑fold cross‑validation accuracy comparison bar chart
- Random Forest feature importance chart

---
### File Structure
- **`outputs/`**
- **`heart-kaggle-refine.xlsx`** 
- **`heart_disease_prediction.py`**  
- **`README.md`**

---
