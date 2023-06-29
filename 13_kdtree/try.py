# TODO - store the structure
# TODO - store the divisor

import numpy as np


points = [1,1,2,3,4,5,6,7,8,9,10,11,12]

tree = []
points = sorted(points)
n = len(points)
m = n // 2 

left = points[:m]
right = points[m:]

print(left)
print(right)

# ----------------------
tree.append([left,right])

# continue
m = len(left) // 2
left_left = left[:m]
left_right = left[m:]

print(left_left, left_right)

m = len(right) // 2
right_left = right[:m]
right_right =  right[m:]

print(right_left, right_right)

tree.append([left,right])