# Lung Cancer Classification in Imbalanced Data

This project investigates machine learning models for detecting a rare positive class (lung cancer) in highly imbalanced medical data.

The study evaluates multiple classifiers and data balancing techniques to maximize cancer detection performance.

## Objectives

- Detect lung cancer cases in imbalanced datasets
- Compare classical and ensemble ML models
- Evaluate resampling strategies
- Optimize hyperparameters
- Maximize recall for rare class detection

### Models

- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression
- XGBoost
- LightGBM
- Voting Ensemble (XGBoost + LightGBM + Random Forest)

### Imbalance Handling

- SMOTE
- Tomek Links
- NearMiss

### Hyperparameter Tuning

- GridSearchCV
- RandomizedSearchCV

## Results

KNN combined with Tomek Links and NearMiss balancing achieved the best practical performance in detecting lung cancer cases in the test dataset.

Ensemble models (LightGBM, XGBoost, VotingClassifier) achieved slightly higher recall but significantly lower precision and overall accuracy.

## Conclusion

For this rare-event medical classification task, where maximizing cancer detection is the primary objective, the NearMiss-balanced KNN model provides the most effective solution.

## Project Structure
notebooks/
feature_engineering.ipynb
modeling.ipynb

validation/
validation.ipynb

data/
lung_cancer_prediction_dataset.csv

## Independent Validation Study

In addition to the main lung cancer classification study, this project includes an independent validation of another group's machine learning project focused on depression prediction.

The goal of this validation was to assess model generalization using provided validation data and alternative training splits.

### Original Models Evaluated

The original project tested:

- Logistic Regression  
- KNN Classifier  
- Decision Tree  
- Random Forest  
- Deep Neural Networks (DNN)  

The neural network model could not be validated due to reproducibility limitations. All remaining models were independently evaluated.

### Validation Procedure

- Training and validation datasets were swapped to test robustness across different training distributions  
- Each model was evaluated using:
  - Confusion Matrix  
  - Classification Report (precision, recall, F1-score)  

This allowed a complete assessment of model performance on unseen data.

### Validation Results

Random Forest achieved the best performance in depression prediction, with the highest precision, recall, and F1-score, particularly for detecting depression cases.

### Interpretation

The validation confirms that Random Forest provides the most reliable classification for depression prediction among the evaluated models, demonstrating strong generalization on alternative training data.

## How to run

Open the notebooks in Jupyter and execute cells sequentially.

## Notes

The project is organized into multiple notebooks to reflect consecutive stages of the classification workflow: feature engineering, modeling, and validation.
