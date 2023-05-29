from numpy import array as ar
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

positive = ar([[100,2,1],[100,4,1],[500,4,1]])
negative = ar([[300,1,2],[300,2,2]])
new_point = ar([[500,1,3]])

X = np.vstack((positive, negative,new_point))

min_max = MinMaxScaler()
x_norm = MinMaxScaler().fit_transform(X[:,0].reshape(-1,1))
y_norm = MinMaxScaler().fit_transform(X[:,1].reshape(-1,1))

#plt.scatter(x = x_norm, y = y_norm, c = X[:,2].reshape(-1,1))
#plt.show()

normalized = np.hstack((x_norm, y_norm))

def euclidean_distance(point_a, point_b):
	return np.sqrt(np.sum((point_a - point_b)**2))


X_points = X[:,0:2]

distances = np.empty([X_points.shape[0],X_points.shape[0]])
for i in range(X_points.shape[0]):
	for j in range(X_points.shape[0]):
#		distances[i,j] = euclidean_distance(X_points[i], X_points[j])
		distances[i,j] = euclidean_distance(normalized[i], normalized[j])

print(distances[5,:])

point_index_min_distance = np.argmin(distances[5,:])
print(point_index_min_distance)
