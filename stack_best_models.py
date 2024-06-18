"""
File: stack_best_models.py
Name: Andy Lin
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection, metrics, preprocessing, ensemble, decomposition
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import time

TRAIN_DATA = 'boston_housing/train.csv'
TEST_DATA = 'boston_housing/test.csv'


def main():
    """
    This file heritage from the result of boston_housing_competition_1.py.
    I choose two of best-performed models to stake: Ridge Regression, XGBoost and use GridSearchCV to find best params.
    In the beginning, I tend to combine PCA with Ridge and XGBoost. Sadly, after training with PCA, the RMS error
    didn't decrease as I expected, so I drop the PCA model.
    Also, the reason why I didn't choose Bagging Ridge Regression is because that the RMS error is not lower than
    simply use Ridge Regression:
    Best score for Bagging Ridge Regression: 3.365480237689973
    Best score for XGBoost: 3.205411410780319
    Stacking Model RMS Error: 3.1335374382551135
    The outcome are the best score in all of the training models.
    Best score for Ridge Regression: 3.329732035938434
    Best score for XGBoost: 3.205411410780319
    Best Ridge Model Parameters: {'alpha': 0.3593813663804626, 'poly_phi_degree': 3}
    Best XGBoost Model Parameters: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.2, 'poly_phi_degree': 2}
    Stacking Model RMS Error: 3.0739997458837887
    """
    data = pd.read_csv(TRAIN_DATA)                          # Read Training Data
    data, y = data_preprocessing(data, mode='Train')        # Preprocessing train data
    normalizer = preprocessing.MinMaxScaler()               # Use MinMaxScaler as normalizer
    data = normalizer.fit_transform(data)                   # Normalize data
    # Split train data and validation data. 30% for validation, 70% for training
    x_train, x_val, y_train, y_val = model_selection.train_test_split(data, y, test_size=0.3, random_state=42)

    """
    After using PCA, the RMS error didn't smaller than not using PCA, so I dropped it.
    Best score for Ridge Regression with PCA: 5.923608788239186
    Best score for XGBoost with PCA: 3.936793934158813
    Stacking Model RMS Error: 4.182548151529072
    """
    # Define Models
    poly_phi = preprocessing.PolynomialFeatures()           # Create polynomial feature
    ridge = linear_model.Ridge()                            # Create Ridge model
    xgboost = xgb.XGBRegressor(random_state=42)             # Create XGBoost model

    # Define param grids
    param_grids = {
        'Ridge Regression': {
            'polynomialfeatures__degree': [1, 2, 3],
            'ridge__alpha': np.logspace(-4, 4, num=10)
        },
        'XGBoost': {
            'polynomialfeatures__degree': [1, 2, 3],
            'xgbregressor__max_depth': [3, 4, 5],                       # 3
            'xgbregressor__n_estimators': [100, 200, 300, 400, 500],    # 400
            'xgbregressor__learning_rate': [0.01, 0.1, 0.2, 0.3]        # 0.2
        }
    }
    # Find best parameters for Ridge Regression
    best_ridge_model, best_ridge_params, best_ridge_score = find_best_model(
        'Ridge Regression', make_pipeline(poly_phi, ridge), param_grids['Ridge Regression'],
        x_train, y_train, x_val, y_val)
    # Find best parameters for XGBoost
    best_xgb_model, best_xgb_params, best_xgb_score = find_best_model(
        'XGBoost', make_pipeline(poly_phi, xgboost), param_grids['XGBoost'],
        x_train, y_train, x_val, y_val)
    print(f'Best Ridge Model Parameters: {best_ridge_params} with RMS Error: {best_ridge_score}')
    print(f'Best XGBoost Model Parameters: {best_xgb_params} with RMS Error: {best_xgb_score}')

    # Create and train stacking regressor with the best model
    stacking_regressor = ensemble.StackingRegressor(
        estimators=[('ridge', best_ridge_model), ('xgboost', best_xgb_model)],
        final_estimator=ridge
    )
    # Training with staking regressor
    stacking_regressor.fit(x_train, y_train)
    # Predict validation
    val_prediction = stacking_regressor.predict(x_val)
    # Calculate stacking regressor's RMS score
    stacking_score = np.sqrt(metrics.mean_squared_error(y_val, val_prediction))
    # Print stacking RMS error
    print(f'Stacking Model RMS Error: {stacking_score}')

    # Predict test data
    test_data = pd.read_csv(TEST_DATA)                                          # Read Test Data
    test_id = test_data.pop('ID')                                               # Remove 'ID' as y
    test_data = normalizer.transform(test_data)                                 # Normalize test data
    stacking_prediction = stacking_regressor.predict(test_data)                 # Predict test data
    out_file(test_id, stacking_prediction, f'stacking_best_submission.csv')     # Out file prediction


def find_best_model(model_name, model, param_grids, x_train, y_train, x_val, y_val):
    print(f'Processing {model_name} training...')
    start = time.time()
    # Use randomize search to find best params
    grid_search = GridSearchCV(model, param_grids, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    # Training randomize search model
    grid_search.fit(x_train, y_train)
    # Find best tuned model
    best_estimator = grid_search.best_estimator_
    # Predict validation
    val_predictions = best_estimator.predict(x_val)
    # Check RMS error of stacking model
    score = np.sqrt(metrics.mean_squared_error(y_val, val_predictions))
    end = time.time()
    print(f'Best score for {model_name}: {score} (Processing time: {end-start} seconds)')
    return best_estimator, grid_search.best_params_, score


def data_preprocessing(data, mode='Train'):
    if mode == 'Train':
        data.drop(columns=['ID'], inplace=True)
        y = data.pop('medv')
        return data, y


def out_file(test_id, predictions, filename):
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        for i, ans in zip(test_id, predictions):
            out.write(f'{i},{ans}\n')
    print('===============================================')


if __name__ == '__main__':
    main()
