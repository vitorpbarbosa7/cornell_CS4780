from numpy import array as ar
import numpy as np

X = ar([[5,-1],[6,1]])
y = ar([+1,-1])

w = np.zeros(X.shape[1])
while True:	
	m = 0
	for i in range(X.shape[0]):

	# if misclassified, must be updated
		if y[i]*(np.dot(w,X[i])) <= 0:
			w = w + y[i]*X[i]
			m += 1
			print(f'Updated w :{w}')
	if m == 0:
		break



for i in range(X.shape[1]):
	sign = np.sign(np.dot(w,X[i]))
	print(sign)
