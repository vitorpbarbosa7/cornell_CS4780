import numpy as np 

xa = [1,2,3]
xb = [3,2,1]

X = np.array([xa,xb])

print(np.cov(X))

n = X.shape[1]
print(n)
mi = np.mean(X,axis = 1)
print(mi)
Xc = (X.T - mi).T
print(Xc)

X_pre_cov = Xc @ Xc.T

cov = (1/(n-1))*(X_pre_cov)
print(cov)


def my_covariance(X):
	'''
	Calculates the covariance matrix
	
	Parameters:
	----------------
	X: (m,n)
		- m features
		- n samples
	'''
	
	n = X.shape[1]
	mi = np.mean(X, axis = 1)
	Xc = (X.T - mi).T
	X_pre_cov = Xc @ Xc.T

	cov = (1/(n-1))*(X_pre_cov)

	return cov

xa = [1,1,1,2,2,3,3,4,5,6,7,8,8,9,9,10]
xb = [1,2,1,2,3,2,4,5,5,6,7,8,8,6,5,8]
X = np.array([xa,xb])
print(np.mean(X, axis = 1))
print(my_covariance(X))



