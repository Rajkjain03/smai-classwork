import numpy as np
import math


class MinMaxScaler:
    def fit(self,X):
        self.min = np.min(X,axis=0)
        self.max = np.max(X,axis=0)

    def transform(self,X):
        return (X-self.min)/(self.max-self.min+1e-8)


class MeanNormalizer:
    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        self.min = np.min(X,axis=0)
        self.max = np.max(X,axis=0)

    def transform(self,X):
        return (X-self.mean)/(self.max-self.min+1e-8)


class MaxAbsScaler:
    def fit(self,X):
        self.maxabs = np.max(np.abs(X),axis=0)

    def transform(self,X):
        return X/(self.maxabs+1e-8)


class DecimalScaling:
    def fit(self,X):
        max_val = np.max(np.abs(X))
        self.j = np.ceil(np.log10(max_val+1))

    def transform(self,X):
        return X/(10**self.j)


class RobustScaler:
    def fit(self,X):
        self.median = np.median(X,axis=0)
        q1 = np.percentile(X,25,axis=0)
        q3 = np.percentile(X,75,axis=0)
        self.iqr = q3-q1

    def transform(self,X):
        return (X-self.median)/(self.iqr+1e-8)


class ZScoreScaler:
    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)

    def transform(self,X):
        return (X-self.mean)/(self.std+1e-8)


class ModifiedZScore:
    def fit(self,X):
        self.median = np.median(X,axis=0)
        self.mad = np.median(np.abs(X-self.median),axis=0)

    def transform(self,X):
        return 0.6745*(X-self.median)/(self.mad+1e-8)


class ParetoScaler:
    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)

    def transform(self,X):
        return (X-self.mean)/(np.sqrt(self.std)+1e-8)


class LogTransform:
    def fit(self,X):
        self.c = abs(np.min(X))+1

    def transform(self,X):
        return np.log(X+self.c)


class ReciprocalTransform:
    def fit(self,X):
        pass

    def transform(self,X):
        return 1/(X+1e-8)


class SqrtTransform:
    def fit(self,X):
        self.shift = abs(np.min(X))

    def transform(self,X):
        return np.sqrt(X+self.shift)


class BoxCox:
    def fit(self,X):
        self.shift = abs(np.min(X))+1
        self.l = 0.5

    def transform(self,X):
        X = X+self.shift
        return (np.power(X,self.l)-1)/self.l


class YeoJohnson:
    def fit(self,X):
        self.l = 0.5

    def transform(self,X):
        X_new = np.zeros_like(X)

        pos = X>=0
        neg = X<0

        X_new[pos]=((X[pos]+1)**self.l-1)/self.l
        X_new[neg]=-( (-X[neg]+1)**(2-self.l)-1)/(2-self.l)

        return X_new


class TanhScaler:
    def fit(self,X):
        pass

    def transform(self,X):
        return np.tanh(X)


class SigmoidScaler:
    def fit(self,X):
        pass

    def transform(self,X):
        return 1/(1+np.exp(-X))


class L1Normalizer:
    def fit(self,X):
        pass

    def transform(self,X):
        norm = np.sum(np.abs(X),axis=1,keepdims=True)
        return X/(norm+1e-8)


class L2Normalizer:
    def fit(self,X):
        pass

    def transform(self,X):
        norm = np.linalg.norm(X,axis=1,keepdims=True)
        return X/(norm+1e-8)


class SoftmaxScaler:
    def fit(self,X):
        pass

    def transform(self,X):
        e = np.exp(X)
        return e/np.sum(e,axis=1,keepdims=True)


class QuantileNormalizer:
    def fit(self,X):
        pass

    def transform(self,X):
        ranks = np.argsort(np.argsort(X,axis=0),axis=0)
        return ranks/(len(X)-1)

class RankGauss:

    def fit(self, X):
        pass

    def transform(self, X):

        ranks = np.argsort(np.argsort(X, axis=0), axis=0)

        uniform = ranks / (len(X) - 1)

        uniform = np.clip(uniform, 1e-6, 1 - 1e-6)

        return np.sqrt(2) * np.log(uniform / (1 - uniform))
