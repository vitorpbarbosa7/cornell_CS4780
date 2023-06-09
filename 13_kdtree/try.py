# TODO, make it calculating distances
# TODO compare the distance to current nearest neighbor to the upside wall

import numpy as np

points = [1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,20,23]

# points = [1,1,2,3,4,5,6,8]

def euclidean_distance(a, b):
    return np.linalg.norm(a,b)

class kdTree:

    def __init__(self, points, min_points_node:int = 2, parent = None):
        '''
        Node structure. Several nodes makes a tree, and that is it
        '''

        # if it has points, so continue to add nodes to the left and to the right
        if len(points) > 1:

            points = sorted(points)
            m = len(points) // 2

            # if its equal or greater than this one, it will go right
            self.divisor = points[m]

            print(points)
            print(self.divisor)

            # let's grow the tree and keep track of parent node
            # first time it will be None
            # next time it will be a node, (second time it will be the root node)
            # third time will be the nodes which are children from the root node
            self._parent = parent
            self.left = kdTree(points = points[:m], parent = self)
            self.right = kdTree(points = points[m:], parent = self)

        if len(points) == min_points_node:
            
            # create a object node_points only at the last level
            # here we can be at right or left side, important it is only at last level
            self.node_points = points

    def new_point(self, npoint):

        # stop criteria
        # we're at the last level if there are points in it
        if hasattr(self, 'node_points'):
            print(f'Nearest neighbors are: {self.node_points}')
        
            self.distance_leaf_nodes = np.sqrt((np.mean(self.node_points) - npoint)**2)
            print(f'Distance to the mean position of this leaf node is {self.distance_leaf_nodes}')

            if self._parent.divisor is not None:
                distance_to_parent_wall = np.sqrt(self._parent.divisor - npoint)**2
                print(f'Distance to parent wall {distance_to_parent_wall}')
                if self.distance_leaf_nodes > distance_to_parent_wall:
                    self._parent.new_point(npoint)

            # go one level up in the tree, and calculate the distance to the wall
            # ??? chatgpt, put the code here, or another location

            # those points are my nearest neighbors
            return self.node_points

        if npoint >= self.divisor:
            print(f'passou aqui com {npoint} e divisor {self.divisor}')
            # go right

            # the object to go further into the search will be the right node (which can also contain more nodes, and also contains the method new_point as it is from this class)

            # keep track of parent node
            self.right.new_point(npoint)
        else:
            # same comment here, same thing
            self = self.left
            self.new_point(npoint)

if __name__ == '__main__':
    kdtree = kdTree(points)

    kdtree.new_point(5)

