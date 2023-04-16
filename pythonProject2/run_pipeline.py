from model import Model
from preprocessor import Preprocessor

class Pipeline:
    def __init__(self,):
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, X, test=False):
        if test:
        # call preprocessor and model for testing
        else:
        # call preprocessor and model for training