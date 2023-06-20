import numpy as np

def rbf_kernel(X1, X2, sigma):
    """
    Computes the RBF kernel between two sets of vectors.

    Args:
    X1: ndarray of shape (m, d), representing the first set of vectors.
    X2: ndarray of shape (n, d), representing the second set of vectors.
    sigma: float, the bandwidth parameter of the RBF kernel.

    Returns:
    K: ndarray of shape (m, n), the RBF kernel matrix.
    """

    m, d1 = X1.shape
    n, d2 = X2.shape

    if d1 != d2:
        raise ValueError("Input dimensions mismatch.")

    K = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            diff = X1[i] - X2[j]
            norm = np.dot(diff, diff)
            K[i, j] = np.exp(-norm / (2 * sigma**2))

    return K



