""""
Author: Jayson C. Garrison
Dates: 01/14/2022
Course: CS-5333 (Machine Learning)
Description: Machine Learning model that classifies objects using Naive Bayes classification
GitHub: https://github.com/jayson-garrison/ML-Naive-Bayes
"""
from abc import ABC, abstractmethod
import numpy as np

class GenericModel(ABC):

    def __init__(self, data_set, label_set, num_features):
        self.labels = label_set
        self.n_features = num_features
        self.guesses = []
        self.accuracies = []
        # model that we are using in supervised learning
        #self.statistical_inference = dict.fromkeys(label_set, np.zeros(num_features - 1))
        self.statistical_inference = []
        for index in range(10):
            self.statistical_inference.append(np.zeros(num_features - 1))

        # complete partitioned dataset
        # data_set is a tuple (tr, te)
        self.data_set = data_set
        # dictionary keyed by class each key is associated with an array that is as long
        # as there are features
        # first value is junk
        self.feature_instances = dict.fromkeys(label_set, np.zeros(num_features) ) # img has 784 features
        self.class_occurances = dict.fromkeys(label_set, 0)
        pass

    # classify as in classify(datapoint) for a guess based on the trained model
    # this routine should also be generalized to accomodate for 
    # bernoulli event and multinomial events
    # we apply rigor here
    @abstractmethod
    def classify(self):
        # now that we have trained the model, we now can use our gained probabilities (organized in some DS)
        # to classify an unseen value
        for partition_set in self.data_set:
            correct_guess = 0
            #test_set = self.data_set[1]
            possibilities = dict.fromkeys(self.labels, 0)
            for image in partition_set:
                # init each possibility with ln(P(y_i))
                for n in range(len(self.labels)):
                    possibilities[n] = np.log( self.class_occurances[n] / len(partition_set) )
            
                #print('possibilities after reset: ', possibilities)

                for feat in range(1, self.n_features):
                    for n in range(len(self.labels)):
                        if image[feat] == 0: # consider the compliment
                            possibilities[n] += np.log(1 - self.statistical_inference[n][feat - 1])
                        else: 
                            possibilities[n] += np.log(self.statistical_inference[n][feat - 1])

                #print('possibilities after cycle: ', possibilities)
                self.guesses.append(list(possibilities.keys())[list(possibilities.values()).index(max(possibilities.values()))])

                if list(possibilities.keys())[list(possibilities.values()).index(max(possibilities.values()))] == image[0]: # get the key
                    correct_guess += 1

            self.accuracies.append(correct_guess / len(partition_set))
        
            print('correct guesses: ', correct_guess)
            print('total length of set: ', len(partition_set))

        # return a list with the accuracies
        return self.accuracies
                
    # train the model on the partitioned data set according to 5 fold cross validation
    # each data point in the set ought to be linear for generalization
    # train_set is tr in (tr, te) where tr is a list of np.array type of len(feat)
    @abstractmethod
    def train(self):
        # we essentially obtain the probabilities P(feat|class) for all such feats in each class
        # we then use this model, which should probably be stored in a self.var to classify()
        train_set = self.data_set[0] # tr

        # populate the feature instances and class occurances
        for image in train_set:
            label = image[0]
            # increase occurances
            self.class_occurances[label] += 1
            # increase feature instances
            self.feature_instances[label] = np.add(self.feature_instances[label], image)
        # train statistical_inference varies based on implementation
        

    # print stats like training accuracy, test accuracy, etc.
    def stats(self):
        print('class occurances: ', self.class_occurances)
        print('length of feat occur: ', len(self.feature_instances[0]))
        # obtain training accuracy
        # obtain testing accuracy
        print('training accuracy: ', self.accuracies[0])
        print('test accuracy: ', self.accuracies[1])
