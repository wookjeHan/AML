# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:12:41 2024

@author: Admin
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class DecisionTree_classifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        self.model = Pipeline([
            ('dt', DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split))
        ])

    def train(self, train_x, train_y, val_x, val_y):
        param_grid = {
            'dt__max_depth': [5, 10, 20],
            'dt__min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(train_x, train_y)

        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)

        self.model = grid_search.best_estimator_  # Update self.model with the best estimator
        self.model.fit(train_x, train_y)

        # Evaluate the best model on the validation set
        val_score = self.model.score(val_x, val_y)
        print("Validation Score:", val_score)

        return self.model
        
    def predict(self, X):
        return self.model.predict(X)
