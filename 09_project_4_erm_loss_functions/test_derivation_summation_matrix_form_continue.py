import numpy as np 
from numpy import array as ar


w = ar([2,3,4])

X = np.array([[1,2,3],[4,5,8],[2,7,4]])
y = np.array([1,-1,1])

wX = np.dot(w.T,X.T)
u = np.ones(w.shape[0])

wX_y = wX - y

matrix_first_term = np.dot(wX_y,X)

loop_sum = 0
for i in range(X.shape[0]):
	# weight vector dot product with each row
	first_term = np.dot(w,X[i,:]) - y[i] 

	loop_sum += np.dot(first_term.T,X[i,:])




print(f'Matrix operation result: {matrix_first_term}')
print(f'Looping sumation result: {loop_sum}')
