# TODO - store the structure -- OK
# TODO - store the divisor -- OK 
# TODO - method to with new node, find the nearest neighbors

points = [1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,23]

points = [1,1,2,3,4,5,6,8]

class kdTree:

    def __init__(self, points, min_points_node:int = 2):

        # if it has points, so continue to add nodes to the left and to the right
        if len(points) > 1:            

            points = sorted(points)
            m = len(points) // 2

            # if its equal or greater than this one, it will go right
            self.divisor = points[m]

            print(points)
            print(self.divisor)
            # breakpoint()

            self.left = kdTree(points[:m])
            self.right = kdTree(points[m:])

        if len(points) == min_points_node:
            
            # create a object node_points only at the last level
            # here we can be at right or left side, important it is only at last level
            self.node_points = points

    def new_point(self, npoint):

        # stop criteria
        # we're at the last level if there are points in it
        if hasattr(self, 'node_points'):
            print(f'Nearest neighbors are: {self.node_points}')
            return self.node_points

        if npoint >= self.divisor:
            print(f'passou aqui com {npoint} e divisor {self.divisor}')
            # go right

            # the object to go further into the search will be the right node (which can also contain more nodes, and also contains the method new_point as it is from this class)
            self = self.right
            self.new_point(npoint)
        else:
            # same comment here, same thing
            self = self.left
            self.new_point(npoint)




kdtree = kdTree(points)
print(dir(kdtree))
