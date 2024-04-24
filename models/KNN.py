from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
class KNN_classifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def train(self, train_X, train_y):
        self.model.fit(train_X, train_y)
        
    def predict(self, test_X):
        return self.model.predict(test_X)
    
    def evaluate(self, test_X, test_y):
        predictions = self.predict(test_X)
        
        accuracy = np.mean(predictions == test_y)
        f1 = f1_score(test_y, predictions, average='weighted')
        
        return accuracy, f1
    

