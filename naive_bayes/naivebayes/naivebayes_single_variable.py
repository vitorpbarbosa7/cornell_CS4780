import numpy as np
from numpy import array as ar
import pandas as pd

class NaiveBayesSingleVar:

    def __init__(self):
         pass
        
    def fit(self, X_train, Y_train):

        self.Y_unique = np.unique(Y_train)
        
        self.y_count = len(self.Y_unique)
        self.X_unique = np.unique(X_train)
        
        self.x_count = len(self.X_unique)
        self.__X_given_y = [X_train[Y_train == y] for y in np.unique(Y_train)]

        prob_table = np.zeros((self.x_count,self.y_count))
        priors = np.zeros(self.y_count)
        x_normalization = np.zeros(self.x_count)
        n_points = len(Y_train)
        for i, x in enumerate(self.X_unique):
            x_normalization[i] = len(X_train[X_train == x])/n_points
            for j, y in enumerate(self.Y_unique):
                denominator = len(Y_train[Y_train == y])
                
                numerator = len(X_train[(X_train == x) & (Y_train == y)])
                prob = numerator/denominator
                prob_table[i,j] = prob

        for j, y in enumerate(self.Y_unique):
            denominator = len(Y_train[Y_train == y])
            priors[j] = denominator/n_points

        self.prob_table = pd.DataFrame(prob_table.T, columns = self.X_unique, index = self.Y_unique)
        self.priors = pd.DataFrame(priors, index = self.Y_unique)
        self.x_normalization = pd.DataFrame(x_normalization, index = self.X_unique)


    def predict_proba(self, X_test):

        x_test_len = len(X_test)

        results = np.ones((x_test_len, len(self.Y_unique)))
        for i, x_test in enumerate(X_test):
            for j,y in enumerate(self.Y_unique):
                likelihood = self.prob_table[x_test][y]
                prior = self.priors.loc[y]
                x_norm =self.x_normalization.loc[x_test]
                
                results[i,j] = (likelihood*prior)/x_norm

        return results


if __name__ == '__main__':
    X_train = ar([3,3,3,3,3,2,2,2,2,2])
    Y_train = ar([1,1,1,1,1,0,0,0,0,0])

    naivebayes_single_var = NaiveBayesSingleVar()
    naivebayes_single_var.fit(X_train, Y_train)

    X_test = ar([3,2])

    results = naivebayes_single_var.predict_proba(X_test)

    print(results)
