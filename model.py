class Model:

    def __init__(self):
        # create some variables here
        pass

    # classify as in classify(datapoint) for a guess based on the trained model
    # this routine should also be generalized to accomodate for 
    # bernoulli event and multinomial events
    def classify(unseen_value):
        # now that we have trained the model, we now can use our gained probabilities (organized in some DS)
        # to classify an unseen value
        pass

    # train the model on the partitioned data set according to 5 fold cross validation
    # each data point in the set ought to be linear for generalization
    def train(data_set):
        # we essentially obtain the probabilities P(feat|class) for all such feats in each class
        # we then use this model, which should probably be stored in a self.var to classify()
        pass

    # print stats like training accuracy, test accuracy, etc.
    def stats():
        pass
