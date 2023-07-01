# kdtree alternates between dimensions to make the splits

import numpy as np
import matplotlib.pyplot as plt

points = [[1,1],[1,2],[2,1],[2,2],
          [1,4],[1,5],[2,4],[2,5],
          [4,1],[4,2],[5,1],[5,2],
          [4,4],[4,5],[5,4],[5,5]]

# points = [[2,4],[4,2],[5,1]]

points = np.array(points)

m = points.shape[0] // 2 

# first dimension
d = 0
# using the x to make the splits
x = sorted(points[:,d])
print(x)
divisor = x[m]
print(divisor)
left = points[points[:,d]<divisor]
right  = points[points[:,d]>=divisor]

# make the next split in another dimension
d = 1
y = sorted(points[:, d])
print(y)
divisor = y[m]
print(divisor)
left_left = left[left[:,d]<divisor]
left_right  = left[left[:,d]>=divisor]

right_left = right[right[:,d]<divisor]
right_right  = right[right[:,d]>=divisor]


plt.scatter(left_left[:,0], left_left[:,1], color = 'red')
plt.scatter(left_right[:,0], left_right[:,1], color = 'blue')
plt.scatter(right_left[:,0], right_left[:,1], color = 'green')
plt.scatter(right_right[:,0], right_right[:,1], color = 'black')
plt.show()


