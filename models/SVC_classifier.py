from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SVC_Image:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=kernel, C=C, gamma=gamma))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)