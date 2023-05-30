import numpy as np
from matplotlib import pyplot as plt

class Losses:

    def __init__(self, hw_x, y):
        self._hw_x = hw_x
        self._y = y
    
    def hinge_loss(self):
        return [max(1 - self.hw_x*self.y, 0) for self.hw_x, self.y in zip(self._hw_x, self._y)]

    def log_loss(self):
        return [np.log(1+np.exp(-self.hw_x*self.y)) for self.hw_x, self.y in zip(self._hw_x, self._y)]
    
    def exponential_loss(self):
        return [np.exp(-self.hw_x*self.y) for self.hw_x, self.y in zip(self._hw_x, self._y)]
    
    def zero_one_loss(self):
        return [0 if np.sign(self.hw_x) == self.y else 1 for self.hw_x, self.y in zip(self._hw_x, self._y)]


hw_x_values = np.arange(-4, 4, 0.01)
_len = int((len(hw_x_values)/2))
y_values = [+1]*_len + [+1]*_len
ls = Losses(hw_x_values, y_values)

losses = []
losses.append(ls.hinge_loss())
losses.append(ls.log_loss())
losses.append(ls.exponential_loss())
losses.append(ls.zero_one_loss())

names = ['hinge_loss','log_loss','exponential_loss', 'zero_one_loss']

for i in range(len(losses)):
    plt.scatter(x = hw_x_values, y = losses[i], marker = '.', label = names[i])
plt.axis([-5, 5, -1, +5])
plt.legend()
plt.show()

'''
what exactly does the hw_x values mean ? I ask if you could give me more praticle examples of how we get to hw_x values from logistic regression or svm for example, why it gets greater than 1 or less than -1
----------------------------
In the given code, hw_x_values represents the values of the decision function or hypothesis, denoted as h(x)w in the context of logistic regression or SVM. These values are computed for different data points or samples. The term hw_x is a combination of the hypothesis h(x) and the weight vector w, which represents the model parameters.

In logistic regression, the decision function h(x)w is often the dot product of the feature vector x and the weight vector w, i.e., h(x)w = w^T * x. The feature vector x contains the input features or attributes of a data point, and the weight vector w contains the coefficients or weights associated with each feature. The dot product of x and w results in the predicted value or score for a given data point.

Similarly, in SVM, the decision function h(x)w is also the dot product of x and w, which determines the position of the data point relative to the decision boundary. The value of h(x)w can be greater than 1 or less than -1 depending on which side of the decision boundary a data point lies. Positive values indicate one class, while negative values indicate the other class.

In practice, the hw_x_values can be obtained by evaluating the decision function h(x)w on a range of input data points. These input data points can be generated using the linspace function in NumPy or obtained from the dataset you are working with. The hw_x values greater than 1 or less than -1 indicate the data points' positions relative to the decision boundary, providing insights into their predicted class labels.

For example, in binary classification with logistic regression or SVM, if a data point's hw_x value is positive and greater than 1, it indicates a high confidence in predicting it as one class. Conversely, if the hw_x value is negative and less than -1, it suggests a high confidence in predicting it as the other class. Values between -1 and 1 indicate a lower level of confidence or proximity to the decision boundary.

It's important to note that the specific range and interpretation of hw_x values may depend on the specific formulation and implementation of the model. The values can be scaled or transformed depending on the requirements and design choices made in the modeling process.
'''


        
