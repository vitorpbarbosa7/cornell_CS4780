import numpy as np


points = [1,1,2,3,4,5,6,7,8]

points = sorted(points)
n = len(points)
m = n // 2 

left = points[:m]
right = points[m:]

print(left)
print(right)

# continue
m = len(left) // 2
left = left[:m]

m = len(right) // 2
right =  right[m:]

print(left)
print(right)