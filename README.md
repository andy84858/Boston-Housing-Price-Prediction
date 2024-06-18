# Boston-Housing-Price-Prediction

## Introduction

This portfolio is modified from the SC201 Mar 2024 Kaggle competition and demonstrates the process of predicting Boston housing prices using various machine learning models. Initially, Linear Regression, Ridge Regression, SVM, Random Forest, and XGBoost are employed, followed by using these models in a bagging approach for further prediction. GridSearch is then used to find the best hyperparameters for each model. Finally, the two best-performing models are selected and stacked to create a model with the best predictive ability.

## Dataset

The dataset used in this Notebook is the Boston Housing dataset. The dataset can be found on Kaggle Datasets [here].
https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset

The Boston Housing Dataset is derived from information collected by the U.S. Census Service concerning housing in the area of Boston, MA. The following describes the dataset columns:

- crim: per capita crime rate by town
- zn: proportion of residential land zoned for lots over 25,000 sq.ft.
- indus: proportion of non-retail business acres per town
- chas: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- nox: nitric oxides concentration (parts per 10 million)
- rm: average number of rooms per dwelling
- age: proportion of owner-occupied units built prior to 1940
- dis: weighted distances to five Boston employment centers
- rad: index of accessibility to radial highways
- tax: full-value property-tax rate per \$10,000.
- ptratio: pupil-teacher ratio by town.
- black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
- lstat: lower status of the population (percent).
- medv: median value of owner-occupied homes in \$1000s.


## Evaluation
The scoring is based on the RMSE, that is RMSE(y) = (∑(hi -yi)^(2))^(0.5)

## File Explanation
- In `find_best_models.py` we will test different models (single model and bagging model of Linear Regression, Ridge Regression, SVM Regression, Random Forest and XGBoost) and find each model's RMS error.
- In `stack_best_models.py`, we further explore the result of find_best_models.py by stacking two of best-performed models: Ridge Regression, XGBoost and use GridSearchCV to find best params.

## Result

### Single models
The following are best performance of each models with their RMS error and params:
1. Linear Regression
```
	 Best RMS score for Linear Regression: 5.324489614528667
	 Best Linear Regression Parameters: {'polynomialfeatures__degree': 1}
```
3. Bagging Linear Regression
```
	 Best RMS score for Bagging Linear Regression: 5.274442697141327
	 Best Bagging Linear Regression Parameters: {'baggingregressor__n_estimators': 500,
                                                     'polynomialfeatures__degree': 1}
```
4. Ridge Regression
```
	 Best RMS score for Ridge Regression: 3.5051286622188953
	 Best Ridge Regression Parameters: {'polynomialfeatures__degree': 3,
                                            'ridge__alpha': 0.8286427728546842}
```
5. Bagging Ridge Regression
```
	 Best RMS score for Bagging Ridge Regression: 3.451171925961784
	 Best Bagging Ridge Regression Parameters:
                                      {'baggingregressor__base_estimator__alpha':0.5689866029018293,
                                       'baggingregressor__n_estimators': 50,
                                       'polynomialfeatures__degree': 3}
```
6. SVM Regression
```
	 Best RMS score for SVM Regression: 4.038415902249551
	 Best SVM Regression Parameters: {'polynomialfeatures__degree': 3,
                                          'svr__C': 19,
                                          'svr__gamma': 0.08685113737513521}
```
7. Bagging SVM Regression
```
	 Best RMS score for Bagging SVM Regression: 4.464853073327095
	 Best Bagging SVM Regression Parameters: {'baggingregressor__base_estimator__C': 9,
                                                  'baggingregressor__base_estimator__gamma': 0.1,
                                                  'baggingregressor__n_estimators': 50,
                                                  'polynomialfeatures__degree': 2}
```
8. Random Forest
```
	 Best RMS score for Random Forest: 3.6673632700767325
	 Best Random Forest Parameters: {'polynomialfeatures__degree': 2,
                                         'randomforestregressor__max_depth': 6,
                                         'randomforestregressor__min_samples_leaf': 3}
```
9. Bagging Random Forest
```
	 Best RMS score for Bagging Random Forest: 3.9554638002233093
	 Best Bagging Random Forest Parameters: {'baggingregressor__base_estimator__max_depth': 5,
		 				 'baggingregressor__base_estimator__min_samples_leaf': 3,
                                           	 'baggingregressor__n_estimators': 10,
		 				 'polynomialfeatures__degree': 3}
```
10. XGBoost
```
	Best RMS score for XGBoost: 3.205411410780319
	Best XGBoost Parameters: {'polynomialfeatures__degree': 2, 
				  'xgbregressor__learning_rate': 0.2,
    				  'xgbregressor__max_depth': 3,
    				  'xgbregressor__n_estimators': 400}
```

12. Bagging XGBoost
```
        Best RMS score for Bagging XGBoost: 3.4415607793645075
        Best Bagging XGBoost Parameters: {'baggingregressor__base_estimator__learning_rate': 0.2,
                                          'baggingregressor__base_estimator__max_depth': 3,
                                          'baggingregressor__base_estimator__n_estimators': 400,
                                          'baggingregressor__n_estimators': 10,
                                          'polynomialfeatures__degree': 1}
```

### Stacking models

In the beginning, we tend to combine PCA with Ridge and XGBoost. Sadly, after training with PCA, the RMS errordidn't decrease as we expected, so we droped the PCA model.
Also, the reason why we didn't choose Bagging Ridge Regression is because that the RMS error is not lower than simply use Ridge Regression:
   
    Best score for Bagging Ridge Regression: 3.365480237689973
    Best score for XGBoost: 3.205411410780319
    Stacking Model RMS Error: 3.1335374382551135
    
The outcome are the best score in all of the training models.
    
    Best score for Ridge Regression: 3.329732035938434
    Best score for XGBoost: 3.205411410780319
    Best Ridge Model Parameters: {'alpha': 0.3593813663804626, 'poly_phi_degree': 3}
    Best XGBoost Model Parameters: {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.2, 'poly_phi_degree': 2}
    Stacking Model RMS Error: 3.0739997458837887
