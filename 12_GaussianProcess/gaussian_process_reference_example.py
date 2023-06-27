import numpy as np
import matplotlib.pyplot as plt

class GaussianProcess:
    def __init__(self, kernel, noise_std=0.1):
        self.kernel = kernel
        self.noise_std = noise_std
        self.X_train = None
        self.y_train = None
        self.K = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.K = self.kernel(X_train, X_train) + self.noise_std**2 * np.eye(len(X_train))
        print(self.K.shape)
        print('\n')

    def predict(self, X_test):
        K_test_train = self.kernel(self.X_train, X_test)
        K_train_test = self.kernel(X_test, self.X_train)
        K_test_test = self.kernel(X_test, X_test)

        print(K_test_train.shape)         #     (20, 100)
        print(K_train_test.shape)  #(100, 20)
        print(K_test_test.shape) #(100, 100)

        print(self.K.shape) # (20,20)
        print(self.y_train.shape)

        # Calculate posterior mean and covariance
        mean = K_test_train.T @ np.linalg.inv(self.K) @ self.y_train
        
        # (100,100) - (100,20) @ (20,20) @ (20, 100)
        cov = K_test_test - K_test_train.T @ np.linalg.inv(self.K) @ K_train_test.T

        return mean, cov

# Define the kernel function (Gaussian RBF)
def rbf_kernel(X1, X2, l=1.0, sigma=1.0):
    dist_sq = np.sum((X1[:, np.newaxis] - X2)**2, axis=-1)
    return sigma**2 * np.exp(-0.5 * dist_sq / l**2)

# Generate synthetic data
np.random.seed(0)
X_train = np.random.rand(20, 1)  # Training inputs
y_train = np.sin(2 * np.pi * X_train) + np.random.normal(0, 0.1, size=(20, 1))  # Training targets
X_test = np.linspace(0, 1, 100)[:, np.newaxis]  # Test inputs

# Create a Gaussian Process regressor with RBF kernel
gp = GaussianProcess(kernel=rbf_kernel, noise_std=0.1)

# Fit the Gaussian Process to the training data
gp.fit(X_train, y_train)

# Make predictions on the test data
y_pred_mean, y_pred_cov = gp.predict(X_test)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_test, y_pred_mean, color='red', label='Predicted Mean')
plt.fill_between(X_test.flatten(), y_pred_mean.flatten() - np.sqrt(np.diag(y_pred_cov)), 
                 y_pred_mean.flatten() + np.sqrt(np.diag(y_pred_cov)), 
                 color='orange', alpha=0.3, label='± 1 Standard Deviation')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

