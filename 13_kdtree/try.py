# TODO - store the structure
# TODO - store the divisor

points = [1,1,2,3,4,5,6,7,8,9,10,11,12]

class kdTree:

    def __init__(self, points):

        # if it has points, so continue to add nodes to the left and to the right
        if len(points) > 1:

            points = sorted(points)
            m = len(points) // 2

            print(points)
            breakpoint()

            self.left = kdTree(points[:m])
            self.right = kdTree(points[m:])

kdtree = kdTree(points)
print(kdtree)
exit()

tree = []
points = sorted(points)
n = len(points)
m = n // 2 

left = points[:m]
right = points[m:]

print(left)
print(right)

# ----------------------

# continue
m = len(left) // 2
left_left = left[:m]
left_right = left[m:]

print(left_left, left_right)

m = len(right) // 2
right_left = right[:m]
right_right =  right[m:]

print(right_left, right_right)


tree.append([[[left_left],[left_right]],[[right_left],[right_right]]])
print(tree)