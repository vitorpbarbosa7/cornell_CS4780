import cvxpy as cp
import numpy as np

# @ : element-wise multiplication
# np.dot : dot product

np.random.seed(1)

m = 15 # rows
n = 10 # columns
p = 5 # constraints

# feature space
P = np.random.randn(n,n)

# covariance matrix (gram matrix)
P = P.T @ P
q = np.random.randn(n)

# >> constraints

# Matrix to multiply with x, for inequality constraings
G = np.random.randn(m, n)

# upper limit inequality constraint
h = G @ np.random.randn(n)

# Matrix to multiply with x, for equality constraints
A = np.random.randn(p,n)

# equality values for equality constraints
b = np.random.randn(p)

# different values of x (each column, as each value to find the hyperplane in Rn space)
x = cp.Variable(n)

# objective function in quadratic form 
objective = (1/2)*cp.quad_form(x,P) + q.T @ x

# constraints (inequality and equality)
constraints = [G @ x <= h, A @ x == b]

problem = cp.Problem(cp.Minimize(objective), constraints)

problem.solve()

print(f'Optimal value is {problem.value}')
print(f'Solution is {x.value}')
print(f'Dual Solution {constraints[0].dual_value}')