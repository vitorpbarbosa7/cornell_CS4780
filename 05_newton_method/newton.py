import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(x0:float = 0.5):

	x = x0
	fx = x**3 + 4*(x**2) + 1	
	while fx > 0.001:
		fx = x**3 - 4*(x**2) + 1
		f1x = 3*(x**2) - 8*x
		
		new_x = x - fx/f1x

		x = new_x

		print(x)
		print(new_x)
		print('\n')
		
		
	return x, fx


x, fx = newton_raphson()

print(x)
print(fx)

 



