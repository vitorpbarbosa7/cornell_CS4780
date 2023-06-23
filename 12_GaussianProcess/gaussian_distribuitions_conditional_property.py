import numpy as np
from scipy.stats import multivariate_normal

# Define the parameters of the joint Gaussian distribution
mean = [0, 0]
covariance_matrix = [[1, 0.5], [0.5, 1]]

# Define the marginal Gaussian distributions
marginal_a = multivariate_normal(mean=mean[0], cov=covariance_matrix[0][0])
marginal_b = multivariate_normal(mean=mean[1], cov=covariance_matrix[1][1])

# Define the conditional Gaussian distribution
conditional_a_given_b = multivariate_normal(mean=mean[0] + np.dot(covariance_matrix[0][1] / covariance_matrix[1][1], mean[1]), 
                                           cov=covariance_matrix[0][0] - np.dot(covariance_matrix[0][1] / covariance_matrix[1][1], covariance_matrix[0][1]))

# Generate some sample data
samples = conditional_a_given_b.rvs(size=1000)

# Calculate the conditional probability
p_a_given_b = conditional_a_given_b.pdf(samples) / marginal_b.pdf(samples)

# Plot the conditional probability distribution
import matplotlib.pyplot as plt

plt.hist(p_a_given_b, bins=30, density=True, alpha=0.5, label='p(a|b)')
plt.xlabel('a')
plt.ylabel('Probability')
plt.title('Conditional Probability Distribution p(a|b)')
plt.legend()
plt.show()
