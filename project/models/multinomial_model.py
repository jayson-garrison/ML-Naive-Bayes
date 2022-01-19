from models.model import GenericModel as GM 

class MultinomialModel(GM):

    def __init__(self, data_set, label_set, num_features):
        super().__init__(data_set, label_set, num_features)

    
    def classify(self):
        return super().classify()

    # multinomial implementation here
    def train(self, k):
        super().train()
        # now we populate statistical_inference with bernoulli methodology
        for classification in range(len(self.labels)): # was range(10)
            all_feats = sum(self.feature_instances[classification])
            for feat in range(1, self.n_features):
                self.statistical_inference[classification][feat-1] = (self.feature_instances[classification][feat] + k) / ( all_feats + self.n_features * k )
                # self.statistical_inference[classification][feat-1] = (self.feature_instances[classification][feat] + k) / ( self.class_occurances[classification] + self.feature_instances[classification][feat] * k )

    def stats(self):
        super().stats()
