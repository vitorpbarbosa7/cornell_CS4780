import numpy as np
import matplotlib.pyplot as plt


def prob(z,y):

	value = 1/(1 + np.exp(-z*y))

	return value

zs = np.arange(-10,10,0.1)
print(zs)

values = [prob(z, -1) for z in zs]

plt.scatter(x = zs, y = values)
plt.show()
