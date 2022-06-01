import sklearn
from sklearn.svm import SVC
import numpy as np



class Model():

    def __init__(self, hyperParams):

        self.hyperParams = hyperParams
        
        self.model = SVC(C=hyperParams.C, probability=True)

    def flattenFCs(self, FCs):

        triuIndices = np.triu_indices(FCs.shape[1], k=1)
        FCs = FCs[:, triuIndices[0], triuIndices[1]]

        return FCs

    def fit(self, FCs, labels):
        
        """
            FCs : (#ofTrains, N, N)
            labels : (#ofTrains)
        """

        # flatten the FCs
        FCs = self.flattenFCs(FCs)
        self.model.fit(FCs, labels)  

    def predict(self, FCs):
        
        FCs = self.flattenFCs(FCs)
        return self.model.predict(FCs)
      
    def predict_proba(self, FCs):

        FCs = self.flattenFCs(FCs)
        return self.model.predict_proba(FCs)


