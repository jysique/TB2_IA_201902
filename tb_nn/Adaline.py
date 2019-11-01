import numpy as np

class AdalineGD(object):
    def __init__(self, eta=0.05, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        X = X.astype(float)
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) > 0.0,0,
                        np.where(self.activation(X) < 0.0,1,2))

    def think(self, X):
        X = X.astype(float)
        return np.where(self.activation(X) > 0.0,"Basico",
                        np.where(self.activation(X) < 0.0,"Acido","Neutro"))

