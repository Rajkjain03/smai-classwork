import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from knn import KNN


def cross_validation(X,y):

    kf = KFold(n_splits=5)

    best_k = 1
    best_score = 0

    for k in range(1,31):

        scores = []

        for train_idx,val_idx in kf.split(X):

            model = KNN(k)

            model.fit(X[train_idx],y[train_idx])

            pred = model.predict(X[val_idx])

            acc = np.mean(pred==y[val_idx])

            scores.append(acc)

        score = np.mean(scores)

        if score>best_score:

            best_score=score
            best_k=k

    return best_k


def evaluate(model,X_test,y_test):

    pred=model.predict(X_test)

    acc=np.mean(pred==y_test)

    f1=f1_score(y_test,pred,average="macro")

    return acc,f1
