from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNN_classifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def train(self, train_X, train_y):
        self.model.fit(train_X, train_y)
        
    def predict(self, test_X):
        return self.model.predict(test_X)
    

