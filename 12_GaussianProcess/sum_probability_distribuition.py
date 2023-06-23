import numpy as np
import matplotlib.pyplot as plt

N = 1000
X = np.random.normal(loc = -5, scale = 1, size = N+1)
Z = np.random.normal(loc = 5, scale = 2, size = N)


# Pad the smaller distribution with zeros to match the size of the larger distribution
if len(X) < len(Z):
    X = np.pad(X, (0, len(Z) - len(X)), 'constant')
else:
    Z = np.pad(Z, (0, len(X) - len(Z)), 'constant')


plt.hist(X)
plt.hist(Z)
plt.show()

XZ = X*Z
X_Z = X + Z

plt.hist(X_Z)
plt.hist(X)
plt.hist(Z)
plt.show()

plt.hist(XZ)
plt.hist(X)
plt.hist(Z)
plt.hist(X_Z)
plt.show()
