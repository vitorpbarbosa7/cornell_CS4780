import numpy as np

X = np.array([[1,1,1,1,1],[-2,-1,0,1,2]]).T
y = np.array([7,4,3,4,7]).T

P = np.diag([1,1,2,2,2])
Lambda = np.diag([1.1]*X.shape[1])

Xmulti = np.dot(np.dot(X.T,P),X)
Xmulti_lambda = Xmulti - Lambda
Xmulti_inv = np.linalg.inv(Xmulti_lambda)

w = np.dot(Xmulti_inv,np.dot(np.dot(X.T,P),y))

print(f'w \n {w}')
