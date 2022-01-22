""""
Author: Jayson C. Garrison
Dates: 01/15/2022
Course: CS-5333 (Machine Learning)
Description: Multi-variate Bernoulli event model implementation
GitHub: https://github.com/jayson-garrison/ML-Naive-Bayes
"""
from models.model import GenericModel as GM

class BernoulliModel(GM):

    def __init__(self, data_set, label_set, num_features):
        super().__init__(data_set, label_set, num_features)

    def classify(self):
        return super().classify() # ret?

    # bernoulli implementation here
    def train(self, k):
        super().train() # ret?
        # now we populate statistical_inference with bernoulli methodology
        for classification in range(len(self.labels)): # was range(10)
            for feat in range(1, self.n_features):
                self.statistical_inference[classification][feat-1] = (self.feature_instances[classification][feat] + k) / ( self.class_occurances[classification] + self.n_features * k )
                # self.statistical_inference[classification][feat-1] = (self.feature_instances[classification][feat] + k) / ( self.class_occurances[classification] + self.feature_instances[classification][feat] * k )            

    def stats(self):
        super().stats()