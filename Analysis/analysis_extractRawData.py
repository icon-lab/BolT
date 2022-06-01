import argparse
from genericpath import exists
import sys
import os
from sklearn.manifold import TSNE
import gc
import os

os.chdir("..")
sys.path.append("./")


import torch


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="hcpTask")
parser.add_argument("-c", "--device", type=str, default="0")
parser.add_argument("-s", "--seed", type=str, default=-1)
parser.add_argument("-f", "--fold", type=int, default = -1)

print("cwd = {}".format(os.getcwd()))

argv = parser.parse_args()

import numpy as np
import torch
from Dataset.datasetDetails import datasetDetailsDict
from Dataset.dataset import getDataset
from utils import Option
from tqdm import tqdm

from Analysis.relevanceCalculator import generate_relevance

datasetName = argv.dataset
datasetDetails = datasetDetailsDict[datasetName]


if(datasetName == "abide1"):
    foldCount = 10
    targetSeeds = [0,1,2,3,4] 
else:
    foldCount = 5
    targetSeeds = [0]

if(argv.seed != -1):
    targetSeeds = [argv.seed]

for seed in targetSeeds: 

    if(datasetName == "abide1"):
        dataset = getDataset(Option({

                "batchSize" : None,
                "dynamicLength" : None,
                "foldCount" : foldCount,
                "datasetSeed" : seed,
                
                "targetTask" : "disease",
                "atlas" : "schaefer7_400",
                "datasetName" : datasetName

            }))

    elif(datasetName == "hcpRest"):
        dataset = getDataset(Option({

                "batchSize" : None,
                "dynamicLength" : None,
                "foldCount" : foldCount,
                "datasetSeed" : seed,
                
                "targetTask" : "gender",
                "atlas" : "schaefer7_400",
                "datasetName" : datasetName

            }))

    elif(datasetName == "hcpTask"):
        dataset = getDataset(Option({

                "batchSize" : None,
                "dynamicLength" : None,
                "foldCount" : foldCount,
                "datasetSeed" : seed,
                
                "targetTask" : "DoesNotMatter",
                "atlas" : "schaefer7_400",
                "datasetName" : datasetName

            }))



    # load model here
    datasetNameToModelPathMapper = {
        "hcpRest" : "./Analysis/TargetSavedModels/hcpRest/seed_{}/".format(seed),
        "hcpTask" : "./Analysis/TargetSavedModels/hcpTask/seed_{}/".format(seed),
        "abide1" : "./Analysis/TargetSavedModels/abide1/seed_{}/".format(seed),
    }

    datasetNameToFolder = {
        "hcpRest" : "./Analysis/Data/hcpRest/seed_{}/".format(seed),
        "hcpTask" : "./Analysis/Data/hcpTask/seed_{}/".format(seed),
        "abide1" : "./Analysis/Data/abide1/seed_{}/".format(seed),
    }

    device = "cuda:{}".format(argv.device)    

    targetFolds = []
    if(argv.fold == -1):
        targetFolds = range(foldCount)
    else:
        targetFolds = [argv.fold]

    for fold in targetFolds:

        print("\n extracting for seed {}, fold {}\n".format(seed, fold))

        dataset.setFold(fold, train=True)

        data = dataset.data
        labels = dataset.labels
        subjIds = dataset.subjectIds

        trainIdx = dataset.trainIdx
        testIdx = dataset.testIdx


        for i, subjId in enumerate(tqdm(subjIds, ncols=60)):

            targetModelFile = datasetNameToModelPathMapper[datasetName] + "model_{}.save".format(fold) # sanity checker, fix fold+1 to fold
            modell = torch.load(targetModelFile, map_location="cpu")

            model = modell.model.to("cuda:{}".format(argv.device))

            torch.cuda.empty_cache()

            isInTrain = i in trainIdx

            timeseries = torch.tensor(data[i]).float().to(device)
            label = torch.tensor(labels[i]).long().to(device)

            targetDumpFolder = datasetNameToFolder[datasetName]
            if(isInTrain):
                targetDumpFolder += "FOLD_{}/TRAIN/{}-{}".format(fold, int(label), subjId)
            else:
                targetDumpFolder += "FOLD_{}/TEST/{}-{}".format(fold, int(label),subjId)
 
            os.makedirs(targetDumpFolder, exist_ok=True)

            timeseries = ( timeseries - timeseries.mean(dim=1, keepdims=True) ) / timeseries.std(dim=1, keepdims=True)
            timeseries = timeseries[None, :, :]

            model.eval()
            inputToken_relevances = generate_relevance(model, timeseries, None)#label) # (nW, T)

            
            viz = inputToken_relevances.detach().cpu().numpy().mean(axis=0)[None,:].repeat(400,axis=0)

            np.save(targetDumpFolder + "/label.npy", label.cpu().numpy())
            np.save(targetDumpFolder + "/clsRelevancyMap.npy", inputToken_relevances.detach().cpu().numpy())

            # SAVE TOKENS

            # FIRST INPUT ITSELF


            token_0 = timeseries.detach().cpu().numpy()[0].T
            np.save(targetDumpFolder + "/token_layerIn.npy", token_0)
            


            layerCount = len(model.blocks)

            for layer in range(layerCount):

                token_layer = model.tokens[layer][0].cpu().detach().numpy()
                
                attentionMaps = model.blocks[layer].transformer.attention.attentionMaps.cpu().detach().numpy()
                np.save(targetDumpFolder + "/attentionMaps_layer{}.npy".format(layer), attentionMaps)

                relative_position_bias_table = model.blocks[layer].transformer.attention.relative_position_bias_table.cpu().detach().numpy()
                np.save(targetDumpFolder + "/relative_position_bias_table_layer{}.npy".format(layer), relative_position_bias_table)
            
            # clean previous caches values
            for token in model.tokens:
                del token
            del model.tokens  
            model.tokens = [] 
            for i in range(len(model.blocks)):
                model.blocks[i].transformer.attention.handle.remove()
                del model.blocks[i].transformer.attention.attentionGradients
                del model.blocks[i].transformer.attention.attentionMaps  

            del token_0
            del timeseries
            del viz
            del label
            #del attentionMaps
            
            del relative_position_bias_table
            del inputToken_relevances
            del model
            del modell

            torch.cuda.empty_cache()

