"""
File: find_best_models.py
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
from sklearn import linear_model, model_selection, metrics, preprocessing, svm, ensemble
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import time
import logging

TRAIN_DATA = 'boston_housing/train.csv'
TEST_DATA = 'boston_housing/test.csv'


def main():
	"""
	In this file, I will test different models and find each model's RMS error.
	The following models are paired testing for single model and bagging model:
	1. Linear Regression
		Best RMS score for Linear Regression: 5.324489614528667
		Best Linear Regression Parameters: {'polynomialfeatures__degree': 1}
	2. Bagging Linear Regression
		Best RMS score for Bagging Linear Regression: 5.274442697141327
		Best Bagging Linear Regression Parameters: {'baggingregressor__n_estimators': 500, 'polynomialfeatures__degree': 1}
	3. Ridge Regression
		Best RMS score for Ridge Regression: 3.5051286622188953
		Best Ridge Regression Parameters: {'polynomialfeatures__degree': 3, 'ridge__alpha': 0.8286427728546842}
	4. Bagging Ridge Regression
		Best RMS score for Bagging Ridge Regression: 3.451171925961784
		Best Bagging Ridge Regression Parameters: {'baggingregressor__base_estimator__alpha': 0.5689866029018293,
		'baggingregressor__n_estimators': 50, 'polynomialfeatures__degree': 3}
	5. SVM Regression
		Best RMS score for SVM Regression: 4.038415902249551
		Best SVM Regression Parameters: {'polynomialfeatures__degree': 3, 'svr__C': 19, 'svr__gamma': 0.08685113737513521}
	6. Bagging SVM Regression
		Best RMS score for Bagging SVM Regression: 4.464853073327095
		Best Bagging SVM Regression Parameters: {'baggingregressor__base_estimator__C': 9,
		'baggingregressor__base_estimator__gamma': 0.1, 'baggingregressor__n_estimators': 50, 'polynomialfeatures__degree': 2}
	7. Random Forest
		Best RMS score for Random Forest: 3.6673632700767325
		Best Random Forest Parameters: {'polynomialfeatures__degree': 2, 'randomforestregressor__max_depth': 6,
		'randomforestregressor__min_samples_leaf': 3}
	8. Bagging Random Forest
		Best RMS score for Bagging Random Forest: 3.9554638002233093
		Best Bagging Random Forest Parameters: {'baggingregressor__base_estimator__max_depth': 5,
		'baggingregressor__base_estimator__min_samples_leaf': 3, 'baggingregressor__n_estimators': 10,
		'polynomialfeatures__degree': 3}
	9. XGBoost
		Best RMS score for XGBoost: 3.205411410780319
		Best XGBoost Parameters: {'polynomialfeatures__degree': 2, 'xgbregressor__learning_rate': 0.2,
		'xgbregressor__max_depth': 3, 'xgbregressor__n_estimators': 400}

	10. Bagging XGBoost
		Best RMS score for Bagging XGBoost: 3.4415607793645075
		Best Bagging XGBoost Parameters: {'baggingregressor__base_estimator__learning_rate': 0.2,
		'baggingregressor__base_estimator__max_depth': 3, 'baggingregressor__base_estimator__n_estimators': 400,
		'baggingregressor__n_estimators': 10, 'polynomialfeatures__degree': 1}
	After running all possible models, I choose top 2 performance model to do further stacking model. I display
	the stacking model in boston_housing_competition_2.py.
	"""
	# Read data
	data = pd.read_csv(TRAIN_DATA)
	# Data preprocessing
	data, y = data_preprocessing(data, mode='Train')
	# Normalize data
	normalizer = preprocessing.MinMaxScaler()
	data = normalizer.fit_transform(data)
	# Split train data and validation data
	x_train, x_val, y_train, y_val = model_selection.train_test_split(data, y, test_size=0.3, random_state=42)
	# Read test data
	test_data = pd.read_csv(TEST_DATA)
	test_id = test_data.pop('ID')

	# Define extractors and models
	poly_phi = preprocessing.PolynomialFeatures()			# Polynomial feature
	lr = linear_model.LinearRegression()					# Linear Regression
	ridge = linear_model.Ridge()							# Ridge Regression
	svr = svm.SVR()											# SVM
	rf = ensemble.RandomForestRegressor(random_state=42)    # Random Forest
	xgboost = xgb.XGBRegressor(random_state=42)				# XGBoost

	# Define model dictionary for GridSearch
	models = {
		'Linear Regression': make_pipeline(poly_phi, lr),
		'Bagging Linear Regression': make_pipeline(poly_phi, ensemble.BaggingRegressor(base_estimator=lr, random_state=42)),
		'Ridge Regression': make_pipeline(poly_phi, ridge),
		'Bagging Ridge Regression': make_pipeline(poly_phi, ensemble.BaggingRegressor(base_estimator=ridge, random_state=42)),
		'SVM Regression': make_pipeline(poly_phi, svr),
		'Bagging SVM Regression': make_pipeline(poly_phi, ensemble.BaggingRegressor(base_estimator=svr, random_state=42)),
		'Random Forest': make_pipeline(poly_phi, rf),
		'Bagging Random Forest': make_pipeline(poly_phi, ensemble.BaggingRegressor(base_estimator=rf, random_state=42)),
		'XGBoost': make_pipeline(poly_phi, xgboost),
		'Bagging XGBoost': make_pipeline(poly_phi, ensemble.BaggingRegressor(base_estimator=xgboost, random_state=42))
	}

	# Define parameter grids
	param_grids = {
		'Linear Regression': {
			'polynomialfeatures__degree': [1, 2, 3]
		},
		'Bagging Linear Regression': {
			'polynomialfeatures__degree': [1, 2, 3],
			'baggingregressor__n_estimators': [10, 50, 100, 500, 1000]
		},
		'Ridge Regression': {
			'polynomialfeatures__degree': [1, 2, 3],
			'ridge__alpha': np.logspace(-4, 4, num=50)
		},
		'Bagging Ridge Regression': {
			'polynomialfeatures__degree': [1, 2, 3],
			'baggingregressor__base_estimator__alpha': np.logspace(-4, 4, num=50),
			'baggingregressor__n_estimators': [10, 50, 100, 500, 1000]
		},
		'SVM Regression': {
			'polynomialfeatures__degree': [1, 2, 3],
			'svr__gamma': np.logspace(-4, 4, num=50),
			'svr__C': range(1, 20, 1)
		},
		'Bagging SVM Regression': {
			'polynomialfeatures__degree': [1, 2, 3],
			'baggingregressor__base_estimator__gamma': np.logspace(-3, 3, num=10),
			'baggingregressor__base_estimator__C': range(1, 10, 1),
			'baggingregressor__n_estimators': [10, 50, 100, 500, 1000]
		},
		'Random Forest': {
			'polynomialfeatures__degree': [1, 2, 3],
			'randomforestregressor__max_depth': range(3, 11),
			'randomforestregressor__min_samples_leaf': range(3, 11)
		},
		'Bagging Random Forest': {
			'polynomialfeatures__degree': [1, 2, 3],
			'baggingregressor__base_estimator__max_depth': range(3, 6),
			'baggingregressor__base_estimator__min_samples_leaf': range(3, 6),
			'baggingregressor__n_estimators': [10, 50, 100, 500, 1000]
		},
		'XGBoost': {
			'polynomialfeatures__degree': [1, 2, 3],
			'xgbregressor__max_depth': range(3, 11),
			'xgbregressor__n_estimators': [100, 200, 300, 400, 500],
			'xgbregressor__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
		},
		'Bagging XGBoost': {
			'polynomialfeatures__degree': [1, 2, 3],
			'baggingregressor__base_estimator__max_depth': range(3, 6),
			'baggingregressor__base_estimator__n_estimators': [100, 300, 400],
			'baggingregressor__base_estimator__learning_rate': [0.01, 0.2, 0.5],
			'baggingregressor__n_estimators': [10, 50, 100, 500, 1000]
		}
	}
	# Training every model
	for model_name, model in models.items():
		best_model, best_params, best_score = find_best_model(model_name, model, param_grids[model_name],
															  x_train, y_train, x_val, y_val)
		print(f'Best {model_name} Parameters: {best_params} with RMS error: {best_score}')
		logging.info(f'Best {model_name} Parameters: {best_params} with RMS error: {best_score}')
		# Normalize test data
		test_data = normalizer.transform(test_data)
		# Predict test data
		best_prediction = best_model.predict(test_data)
		# Outfile
		out_file(test_id, best_prediction, f'best_{model_name.lower().replace(" ", "_")}_submission.csv')


def find_best_model(model_name, model, param_grid, x_train, y_train, x_val, y_val):
	logging.info(f'Processing {model_name}...')
	start = time.time()
	# Grid Search model
	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=3, cv=5, n_jobs=-1)
	# Fit train data
	grid_search.fit(x_train, y_train)
	# Find out best estimator
	best_estimator = grid_search.best_estimator_
	# Predict validation data
	val_predictions = best_estimator.predict(x_val)
	score = np.sqrt(metrics.mean_squared_error(y_val, val_predictions))
	# Calculated training time
	end = time.time()
	logging.info(f'Best RMS score for {model_name}: {score} (Processing time: {end-start})')
	return best_estimator, grid_search.best_params_, score


def data_preprocessing(data, mode='Train'):
	if mode == 'Train':
		# Validation Cross -> Split into k folds
		data.drop(columns=['ID'], inplace=True)
		y = data.pop('medv')
		return data, y


def out_file(test_id, predictions, filename):
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		start_i = 0
		for ans in predictions:
			out.write(str(test_id[start_i]) + ',' + str(ans) + '\n')
			start_i += 1
	print('===============================================')


if __name__ == '__main__':
	main()
