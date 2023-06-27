import numpy as np 
import matplotlib.pyplot as plt 
from typing import Callable


import numpy as np
from numpy.matlib import repmat

def l2_distance(X, Z=None):
	"""
	function D=l2distance(X,Z)
	
	Computes the Euclidean distance matrix.
	Syntax:
	D=l2distance(X,Z)
	Input:
	X: dxn data matrix with n vectors (columns) of dimensionality d
	Z: dxm data matrix with m vectors (columns) of dimensionality d
	
	Output:
	Matrix D of size nxm
	D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
	
	call with only one input:
	l2distance(X)=l2distance(X,X)
	"""

	if Z is None:
		## fill in code here
		## << ms2666
		n, d = X.shape
		s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
		D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
		D = D1 + repmat(s1.T, n, 1)
		np.fill_diagonal(D, 0)
		D = np.sqrt(np.maximum(D, 0))
		## >> ms2666
	else:
		## fill in code here
		## << ms2666
		n, d = X.shape
		m, _ = Z.shape
		s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
		s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
		D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
		D = D1 + repmat(s2, n, 1)
		D = np.sqrt(np.maximum(D, 0))
		## >> ms2666
	return D


def rbf_kernel(X, Z, width):
	l2distance = l2_distance(X,Z)
	K = np.exp(-(1/width**2)*l2distance**2)
	return K

class GaussianProcess:

	def __init__(self, noise:float = 0.1, sigma:float = 1.0):
		self._noise = noise
		self.sigma = sigma

	def fit(self, X_train, y_train):
		self.y_train = y_train
		self.X_train = X_train
		n, d = self.X_train.shape

		noise = self._noise**2*(np.eye(n))
		self.K = self.kernel(self.X_train, self.X_train,self.sigma) + noise

	def predict(self, X_test):
		K_test_train = self.kernel(self.X_train, X_test, self.sigma)
		K_train_test = self.kernel(X_test, self.X_train, self.sigma)
		K_test_test = self.kernel(X_test, X_test, self.sigma)
		
		n, d = X_test.shape

		# print(K_test_train.shape)
		# print(K_train_test.shape)
		# print(K_test_test.shape)
		# print(self.K.shape)
		# print(self.y_train.shape)
	
		# (2, 10) @ (10, 10) @ (10, 1) 
		mean = K_test_train.T @ np.linalg.inv(self.K) @ self.y_train
		cov = K_test_test + self._noise**2*(np.eye(n)) - K_test_train.T @ np.linalg.inv(self.K) @ K_train_test.T

		return mean, cov
	
	def kernel(self, X, Z, width = 1.0):
		return rbf_kernel(X, Z, width)

X_train = np.linspace(0.1, 2, 20).reshape(-1,1)
y_train = np.sin(2 * np.pi * X_train).reshape(-1, 1)

X_test = np.linspace(0.1, 2, 10).reshape(-1, 1)
y_test = np.sin(2 * np.pi * X_test).reshape(-1,1)

gp = GaussianProcess(sigma = 2)
gp.fit(X_train,y_train)
mean, cov = gp.predict(X_test)

plt.scatter(X_train, y_train)
plt.fill_between(X_test.flatten(), mean.flatten() - np.sqrt(np.diag(cov)), 
                 mean.flatten() + np.sqrt(np.diag(cov)), 
                 color='orange', alpha=0.3, label='± 1 Standard Deviation')
plt.show()
