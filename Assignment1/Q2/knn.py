import numpy as np

class KNN:

    def __init__(self,k):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X_test):

        preds = []

        for x in X_test:

            dist = np.sqrt(((self.X-x)**2).sum(axis=1))

            idx = np.argsort(dist)[:self.k]

            labels = self.y[idx]

            values,counts = np.unique(labels,return_counts=True)

            preds.append(values[np.argmax(counts)])

        return np.array(preds)
