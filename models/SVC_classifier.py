from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV

class SVC_classifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = Pipeline([
            ('svc', SVC(kernel=kernel, C=C, gamma=gamma,verbose=True))
        ])

    def train(self, train_x, train_y, val_x, val_y):
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto']
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

    #def score(self, X, y):
     #   return self.model.score(X, y)