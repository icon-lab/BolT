from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys

from datetime import datetime

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option
from utils import Option, calculateMetric

from Models.BolT.model import Model
from Dataset.dataset import getDataset

def train(model, dataset, fold, nOfEpochs):

    dataLoader = dataset.getFold(fold, train=True)

    for epoch in range(nOfEpochs):

            preds = []
            probs = []
            groundTruths = []
            losses = []

            for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                
                xTrain = data["timeseries"] # (batchSize, N, dynamicLength)
                yTrain = data["label"] # (batchSize, )

                # NOTE: xTrain and yTrain are still on "cpu" at this point

                train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, train=True)

                torch.cuda.empty_cache()

                preds.append(train_preds)
                probs.append(train_probs)
                groundTruths.append(yTrain)
                losses.append(train_loss)

            preds = torch.cat(preds, dim=0).numpy()
            probs = torch.cat(probs, dim=0).numpy()
            groundTruths = torch.cat(groundTruths, dim=0).numpy()
            losses = torch.tensor(losses).numpy()

            metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
            print("Train metrics : {}".format(metrics))                  


    return preds, probs, groundTruths, losses



def test(model, dataset, fold):

    dataLoader = dataset.getFold(fold, train=False)

    preds = []
    probs = []
    groundTruths = []
    losses = []        

    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, train=False)
        
        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()          

    metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
    print("\n \n Test metrics : {}".format(metrics))                
    
    return preds, probs, groundTruths, loss
    


def run_bolT(hyperParams, datasetDetails, device="cuda:3", analysis=False):


    # extract datasetDetails

    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs


    dataset = getDataset(datasetDetails)


    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "nOfClasses" : datasetDetails.nOfClasses,
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs
    })


    results = []

    for fold in range(foldCount):

        model = Model(hyperParams, details)


        train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, fold, nOfEpochs)   
        test_preds, test_probs, test_groundTruths, test_loss = test(model, dataset, fold)

        result = {

            "train" : {
                "labels" : train_groundTruths,
                "predictions" : train_preds,
                "probs" : train_probs,
                "loss" : train_loss
            },

            "test" : {
                "labels" : test_groundTruths,
                "predictions" : test_preds,
                "probs" : test_probs,
                "loss" : test_loss
            }

        }

        results.append(result)


        if(analysis):
            targetSaveDir = "./Analysis/TargetSavedModels/{}/seed_{}/".format(datasetDetails.datasetName, datasetSeed)
            os.makedirs(targetSaveDir, exist_ok=True)
            torch.save(model, targetSaveDir + "/model_{}.save".format(fold))


    return results
