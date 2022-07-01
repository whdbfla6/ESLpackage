import numpy as np
from numpy.linalg import inv

def prob(X,beta):
    return np.exp(np.matmul(X,beta))/(1+np.exp(np.matmul(X,beta)))

class LogisticRegression:

    def __init__(
            self,
            max_iter: int = 100,
            error: float = 3e-2
            ):

        self.max_iter = max_iter
        self.error = error
    
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
            ):

        n,p = X.shape
        # 1 append
        X_one = np.insert(X,0,np.ones(n),axis=1)
        beta = [0]*(p+1)

        p = prob(X_one,beta)
        W = np.diag(p*(1-p))

        # Newton-Raphson method
        while self.max_iter > 0 :
            self.max_iter -= 1
            beta = inv(X_one.T @ W @ X_one) @ X_one.T @ W @ (X_one@beta + inv(W)@(y-prob(X_one,beta)))
            pred = prob(X_one,beta)
            self.error = sum(np.abs(pred-y))/n
            if self.error < 0.01 : break

        self.beta = beta

    def predict(
            self,
            X: np.ndarray
            ):
        n,p = X.shape
        X_one = np.insert(X,0,np.ones(n),axis=1)
        p = prob(X_one,self.beta)
        pred = np.where(p>=0.5,1,0)

        return pred