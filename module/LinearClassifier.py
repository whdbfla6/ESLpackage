import numpy as np
from numpy.linalg import inv,det

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

class LinearDiscriminantFunction:

    def discriminant_function(self,x,mu,sigma,prior):
        return x@inv(sigma)@mu - 0.5*mu.T@inv(sigma)@mu + np.log(prior)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
            ):

        n= X.shape[0]
        self.unique_class = np.unique(y)
        class_num = len(self.unique_class)
        self.prior = [[] for _ in range(class_num)]
        self.mu = [[] for _ in range(class_num)]
        sigma = 0

        for c in self.unique_class:
            ind = np.where(y==c)[0] #클래스 c에 속하는 데이터 인덱스
            
            self.prior[c] = len(ind)/n
            self.mu[c] = np.mean(X[ind],axis=0)
            sigma += np.matmul(X[ind].T,X[ind])

        sigma /= (n-class_num)
        self.s = sigma
        self.sigma = [sigma for _ in range(class_num)] # qda에서 predict함수 그대로 사용하기 위해 차원을 맞춰줌 -> 동일한 값을 클래스 수만큼 생성

        self.df = [[] for _ in range(class_num)]

    def predict(
            self,
            X: np.ndarray
            ):

        n = X.shape[0]
        DF = np.array([-100000]*n)
        pred = np.array([-1]*n)

        for c in self.unique_class:
            df = self.discriminant_function(X,self.mu[c],self.sigma[c],self.prior[c]) #lda: self.sigma[c]는 모두 동일한 값
            self.df = df
            pred = np.where(df > DF,c,pred)
            DF = np.where(df > DF,df,DF)

        return pred


class QuadraticDiscriminantFunction(LinearDiscriminantFunction):

    def discriminant_function(self, x, mu, sigma, prior):
        return np.diag(-0.5*np.log(det(sigma)) - 0.5*(x-mu)@inv(sigma)@(x-mu).T + np.log(prior)) #(nxn) -> (nx1) 대각성분만 이용

    def fit(self, X: np.ndarray, y: np.ndarray):
        n= X.shape[0]
        self.unique_class = np.unique(y)
        class_num = len(self.unique_class)
        self.prior = [[] for _ in range(class_num)]
        self.mu = [[] for _ in range(class_num)]
        self.sigma = [[] for _ in range(class_num)] #self.sigma를 클래스 별로 계산해준다

        for c in self.unique_class:
            ind = np.where(y==c)[0] #클래스 c에 속하는 데이터 인덱스
            self.prior[c] = len(ind)/n
            self.mu[c] = np.mean(X[ind],axis=0)
            self.sigma[c] = np.matmul(X[ind].T,X[ind])/(len(ind)-1)

        self.df = [[] for _ in range(class_num)]

    def predict(self, X: np.ndarray):
        return super().predict(X)
