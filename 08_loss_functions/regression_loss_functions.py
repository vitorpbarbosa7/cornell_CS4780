import numpy as np
from matplotlib import pyplot as plt

class RegressionLosses:

	def __init__(self, hw_x, y):
		self._hw_x = hw_x
		self._y = y
	
	def squared_loss(self):
		return (self._hw_x - self._y)**2

	def absolute_loss(self):
		return np.abs(self._hw_x-self._y)
	
	def huber_loss(self, delta):
		'''
		## greater deltas allows a quadratic behaviour even far away from zero
		## smaller deltas makes the quadratic behaviour only appear very close to zero, and
		the linear behaviour from absolute_loss in the remaining points, which are the greater differences
		
		'''
		losses = []
		abs_losses = self.absolute_loss()
	
		for i in range(len(abs_losses)):
			if abs_losses[i] >= delta:
				losses.append(delta*(abs_losses[i] - delta/2))
			else:
				losses.append(0.5*self._squared_loss(self._hw_x[i], self._y[i]))
		return losses
	
	def log_cosh_loss(self):
		return np.log(np.cosh(self._hw_x - self._y))	
	
	def _squared_loss(self, hw_x, y):
		return (hw_x - y)**2

hw_x_values = np.arange(-4, 4, 0.01)
_len = int((len(hw_x_values)/2))
y_values = [+1]*_len + [+1]*_len
ls = RegressionLosses(hw_x_values, y_values)

losses = []
losses.append(ls.squared_loss())
losses.append(ls.absolute_loss())
losses.append(ls.huber_loss(delta = 1))
losses.append(ls.huber_loss(delta = 5))
losses.append(ls.log_cosh_loss())

names = ['squared_loss','absolute_loss','huber_loss_delta_1', 'huber_loss_delta_5','log_cosh_loss']

for i in range(len(losses)):
	plt.scatter(x = hw_x_values, y = losses[i], marker = '.', label = names[i])
plt.axis([-5, 5, -1, +5])
plt.legend()
plt.show()

	   
