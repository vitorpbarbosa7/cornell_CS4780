# TODO, make it calculating distances
# TODO compare the distance to current nearest neighbor to the upside wall

import numpy as np
import matplotlib.pyplot as plt

points = [[1,1],[1,2],[2,1],[2,2],
          [1,4],[1,5],[2,4],[2,5],
          [4,1],[4,2],[5,1],[5,2],
          [4,4],[4,5],[5,4],[5,5]]

points = np.array(points)

def euclidean_distance(a, b):
    return np.linalg.norm(a,b)

class kdTree:

    def __init__(self, points, d=0, min_points_node:int = 4, parent = None):
        '''
        Node structure. Several nodes makes a tree, and that is it

        Initial dimension will be 0. It could first asses what is the dimension
        with the most spread
        '''
        # just to keep track
        self._parent = parent
        
        # Initialize it as None, if satifies the sequence of criterias it will receive or not new nodes
        self.left = None
        self.right  = None

        d = int(d)
        self._d = d

        # if there are enough points, continue to add
        if len(points) > min_points_node:
            m = len(points) // 2

            print(f'Dimension to split: {self._d}')
            # dimension to split
            d_split = sorted(points[:, d])
            
            # if its equal or greater than this one, it will go right
            self.divisor = d_split[m]

            _left = points[points[:,d]<self.divisor]
            _right = points[points[:,d]>=self.divisor]

            print(points)
            print(self.divisor)

            # let's grow the tree and keep track of parent node
            # first time it will be None
            # next time it will be a node, (second time it will be the root node)
            # third time will be the nodes which are children from the root node

            # TODO make it alternate for several dimension
            self.left = kdTree(points = _left, parent = self, d = not d)
            self.right = kdTree(points = _right, parent = self, d = not d)

        else:
            
            # create a object leaf_points only at the last level
            # here we can be at right or left side, important it is only at last level
            self.leaf_points = points

    def new_point(self, npoint):

        # stop criteria
        # we're at the last level if there are points in it
        if hasattr(self, 'leaf_points'):
            print(f'Nearest neighbors are: {self.leaf_points}')
        
            self.distance_leaf_nodes = np.sqrt((np.mean(self.leaf_points) - npoint)**2)
            print(f'Distance to the mean position of this leaf node is {self.distance_leaf_nodes}')

            if self._parent.divisor is not None:
                distance_to_parent_wall = np.sqrt(self._parent.divisor - npoint)**2
                print(f'Distance to parent wall {distance_to_parent_wall}')
                if self.distance_leaf_nodes > distance_to_parent_wall:
                    self._parent.new_point(npoint)
                    # TODO continue this part

            # those points are my nearest neighbors
            return self.leaf_points

        if npoint >= self.divisor:
            print(f'passou aqui com {npoint} e divisor {self.divisor}')
            # go right

            # the object to go further into the search will be the right node (which can also contain more nodes, and also contains the method new_point as it is from this class)
            self.right.new_point(npoint)
        else:
            # same comment here, same thing
            self.left.new_point(npoint)

    # def traverse_count_leaf_nodes(self):

    #     if hasattr(self, 'leaf_points'):
    #         self.qtd_leaf_nodes = 

    def traverse_and_plot(self):

        if hasattr(self, 'leaf_points'):
            plt.scatter(self.leaf_points[:,0], self.leaf_points[:,1])

            # if last_node:
            #     plt.show()

        else:
            if self.left is not None:
                self.left.traverse_and_plot()
            if self.right is not None:
                self.right.traverse_and_plot()

    
if __name__ == '__main__':
    kdtree = kdTree(points)
    kdtree.traverse_and_plot()
    plt.show()

    # kdtree.new_point([4.5,2.5])

