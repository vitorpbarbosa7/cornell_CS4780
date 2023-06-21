import numpy as np
import cvxpy as cp

#<GRADED>
def primalSVM(xTr, yTr, C=1):
    """
    function (classifier,w,b) = primalSVM(xTr,yTr;C=1)
    constructs the SVM primal formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (n)
        C     | the SVM regularization parameter
    
    Output:
        fun   | usage: predictions=fun(xTe); predictions.shape = (n,)
        wout  | the weight vector calculated by the solver
        bout  | the bias term calculated by the solver
    """
    
    n, d = xTr.shape
    
    C = np.array([C]*n)
    
    w = cp.Variable(d) # Rd space features
    b = cp.Variable() # bias term
    Xi = cp.Variable(n) # Slack variables
    
    I = np.eye(d)
    objective = cp.quad_form(w, I) + C.T @ Xi
    
    constraints = [cp.multiply(-yTr, ((xTr @ w) + b)) <= -1 + Xi, -Xi <= np.zeros(n)]
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    problem.solve(verbose=False, solver = 'CVXOPT')
        
    wout = w.value
    bout = b.value
    
    print('\nHyperplane vectors')
    print(wout)
    
    print('\n bias term')
    print(bout)
    
    print('\n objective function value')
    print(problem.value)
    
    print('\n constraints dual problem values (kkt multipliers)')
    print(constraints[0].dual_value)

    fun = lambda xTe: xTe @ wout + bout
    
    return fun, wout, bout
#</GRADED>