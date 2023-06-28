import matplotlib.pyplot as plt

# X  | Y 
# -1 | 5
# 1  | 5
# 2  | 8

#f(x) = GP(m,k)
#m(x) = 0


import numpy as np

X_train = np.matrix([-1,1,2])
X_test = np.matrix([-2,0])
y_train = np.matrix([5,5,8])
y_test = np.matrix([8,4])

# it will be used a zero mean function and the following kernel for the covariance matrix:
# k(x,x') = (x . x' + 1 )**2


# a) what is the mean and covariance of your GP prior?

mean_prior = np.matrix([0,0,0]).T
print(mean_prior)

def kernel(X,Z):

	n = X.shape[0]
	m = Z.shape[0]

	return np.square((X.T @ Z + np.ones((n,m))))

cov_prior = kernel(X_train, X_train)
print(cov_prior)

# b) with the following test points, what is the man and covariance of your GP posterior?

## posterior mean
K_train_test = kernel(X_train,X_test)

mean_posterior = K_train_test.T @ np.linalg.inv(cov_prior) @ y_train.T
print(mean_posterior)

print(mean_posterior.flatten() == y_test)

## posterior covariance

K_test_test = kernel(X_test,X_test)
K_test_train = kernel(X_test,X_train)

cov_posterior = K_test_test - K_train_test.T @ np.linalg.inv(cov_prior) @ K_test_train.T
print(cov_posterior)

