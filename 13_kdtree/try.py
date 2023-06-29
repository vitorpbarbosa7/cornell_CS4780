# TODO - store the structure
# TODO - store the divisor

points = [1,1,2,3,4,5,6,7,8,9,10,11,12]

class kdTree:

    def __init__(self, points):

        # if it has points, so continue to add nodes to the left and to the right
        if len(points) > 1:            

            points = sorted(points)
            m = len(points) // 2

            self.divisor = m

            print(points)
            print(self.divisor)
            # breakpoint()

            self.left = kdTree(points[:m])
            self.right = kdTree(points[m:])

        if len(points) == 1:

            self.left = points
            self.right = points


kdtree = kdTree(points)
print(dir(kdtree))
