import numpy as np

X = np.array([[1,1,1,1,1],[-2,-1,0,1,2]])
y = np.array([7,4,3,4,7])

Xt = X.T

Xmulti = np.dot(X,Xt)
Xmulti_inv = np.linalg.inv(Xmulti)

print(Xmulti_inv)

w = np.dot(Xmulti_inv,np.dot(X,y))

print(f'w \n {w}')
