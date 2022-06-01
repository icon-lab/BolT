import time
import numpy as np
from Models.BolT.run import train
from utils import Option

from Models.SVM.model import Model
from Models.SVM.util import corrcoef, ledoit_wolf_corrcoef
from Dataset.dataset import getDataset


def calculateFC(timeseries):

    FCs = []

    for timeseries_ in timeseries:
        timeseries_ = np.array(timeseries_)
        FCs.append(corrcoef(timeseries_))

    return np.array(FCs)


def extractDataLoader(dataLoader):

    timeseries = []
    labels = []

    for data in dataLoader:
        timeseries.extend(data["timeseries"].tolist())
        labels.extend(data["label"].tolist())

    labels = np.array(labels)

    return timeseries, labels


def run_svm(hyperParams, datasetDetails, device=None): # here device is added for compatibility with other deep learning gang 
    
    # extract datasetDetails
    foldCount = datasetDetails.foldCount


    dataset = getDataset(datasetDetails)

    results = []

    for fold in range(foldCount):

        print("Running svm fold : {}".format(fold))

        train_dataLoader = dataset.getFold(fold, train=True)
        train_timeseries, train_labels = extractDataLoader(train_dataLoader)
        
        test_dataLoader = dataset.getFold(fold, train=False)
        test_timeseries, test_labels = extractDataLoader(test_dataLoader)

        train_FCs = calculateFC(train_timeseries)
        test_FCs = calculateFC(test_timeseries)

        model = Model(hyperParams)
        
        model.fit(train_FCs, train_labels)

        train_probs = model.predict_proba(train_FCs)
        test_probs = model.predict_proba(test_FCs)

        train_predictions = train_probs.argmax(axis=1)
        test_predictions = test_probs.argmax(axis=1)

        result = {
            
                "train" : {
                   
                    "labels" : train_labels,
                    "predictions" : train_predictions,
                    "probs" : train_probs
                    
                    },

                "test" : {
                    
                    "labels": test_labels,
                    "predictions" : test_predictions,
                    "probs" : test_probs

                    }
        }

        results.append(result) 

    return results

