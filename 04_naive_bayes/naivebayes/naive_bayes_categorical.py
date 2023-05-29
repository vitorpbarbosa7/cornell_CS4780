import numpy as np
from numpy import array as ar
import pandas as pd

from .naivebayes_single_variable import NaiveBayesSingleVar


class NaiveBayesCategorical:

    def __init__(self):
        pass


    def fit(self, X_train:np.array, Y_train:np.array):
        self.X_train = X_train
        self.__prob_table_dd = []
        self.__X_normalization_dd = []

        for X_train_single_feature in self.X_train[:,]:
            naivebayes_single_var = NaiveBayesSingleVar()
            naivebayes_single_var.fit(X_train_single_feature, Y_train)

            self.__prob_table_dd.append(naivebayes_single_var.prob_table)
            self.__X_normalization_dd.append(naivebayes_single_var.x_normalization)

        self.__priors = naivebayes_single_var.priors

        self.Y_unique = naivebayes_single_var.Y_unique


    def predict_proba(self, X_test:np.array):
        results = np.zeros((X_test.shape[0], len(self.Y_unique)))
        for i, x_test in enumerate(X_test):
            for j, y in enumerate(self.Y_unique):

                likelihood = []
                prior = self.__priors.loc[y][0]
                x_norm = []
                
                # Naive Bayes Assumption of independence among dimensions
                for d in range(self.X_train[:,].shape[0]):
                    x_test_d = x_test[d]
                    
                    likelihood.append(self.__prob_table_dd[d][x_test_d][y])
                    x_norm.append(self.__X_normalization_dd[d].loc[x_test_d])
                
                likelihood_prod = np.prod(likelihood)
                posterior_numerator = likelihood_prod*prior
                posterior_denominator = np.prod(x_norm)

                posterior = posterior_numerator/posterior_denominator
                
                results[i,j] = posterior


        return results


