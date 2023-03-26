# curse of dimensionality
import matplotlib.pyplot as plt
import numpy as np

k = 10
n = 1000
ds = np.arange(1, 1000, 1)

def calc_l(d):
	
	return (k/n)**(1/d)

l = [calc_l(d) for d in ds]

plt.scatter(x = ds, y = l)
plt.show()
