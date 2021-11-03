from aggregators.aggregator import Aggregator


class NoAggregator(Aggregator):
    '''
    Empty aggregator class for use with a single detector model. It simple returns the predictions of the single detector model
    '''

    def __init__(self, hyperparameters=None):
        self.model = None

    def train(self, preds, y):
        pass

    def aggregate(self, preds):
        return preds.squeeze()

    def get_weights(self):
        return None
