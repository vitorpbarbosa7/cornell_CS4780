import numpy as np
import matplotlib.pyplot as plt

# Set the means and covariance matrix
means = [5, 4]
covariance = [[11, 7], [7, 6]]

# Generate the random variables
np.random.seed(0)
XA, XB = np.random.multivariate_normal(means, covariance, 1000).T

# Calculate the joint probability distribution
joint_prob, x_edges, y_edges = np.histogram2d(XA, XB, bins=50, density=True)

# Plot the heatmap
plt.imshow(joint_prob.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
           origin='lower', cmap='hot')
plt.colorbar()
plt.xlabel('XA')
plt.ylabel('XB')
plt.title('Joint Probability Distribution')
#plt.show()
