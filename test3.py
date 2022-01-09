from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem

train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

class Model(object):

    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        kernel = DotProduct(sigma_0=0, sigma_0_bounds="fixed")
        self.model = GaussianProcessRegressor(normalize_y=True, kernel = kernel)

    def predict(self, x):
        x_transformed = self.feature_map_nystroem.transform(x)
        output = self.model.predict(x_transformed, return_std = True)
        gp_mean = output[0]
        gp_std = output[1]
        predictions = gp_mean
        return predictions, gp_mean, gp_std

    def fit_model(self, train_x, train_y):
        self.feature_map_nystroem = Nystroem(n_components=300)
        train_x_transformed = self.feature_map_nystroem.fit_transform(train_x)
        self.model.fit(train_x_transformed,train_y)


model = Model()
model.fit_model(train_x,train_y)
output = model.predict(test_x)
print(output[2])
