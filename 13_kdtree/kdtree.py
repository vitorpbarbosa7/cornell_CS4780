import numpy as np


class kdTree:
    def __init__(self, P, d=0, min_points = 1):

        # d : dimension

        # get median point from this dimension
        n = len(P)

        #storing a single point 
        self.point = P[m]
        d = d + 1
        # using the median
        m = n // 2
        P.sort(key = lambda x: x[d])

        # if I have points, than I will continue to build the tree
        if m >= min_points:
            # first halft 
            self.left_node = kdTree(P[:m])

        if n - (m+1)  >= min_points:
            self.right_node = kdTree(P[(m+1):])



        # that is, to call again my function which builds the trees
        # basically my node is always input to the KdTree class function


        # second half
        