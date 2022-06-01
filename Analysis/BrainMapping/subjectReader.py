from glob import glob
import numpy as np

import sys
sys.path.append("../")

def getSubjects(dataset, seed, fold, isTrain):
    
    datasetNameToFolderDict = {
        "hcpRest" : "./Analysis/Data/hcpRest",
        "hcpTask" : "./Analysis/Data/hcpTask",
        "abide1" : "./Analysis/Data/abide1"
    }

    targetFolder = datasetNameToFolderDict[dataset]
    targetFolder += "/seed_{}/FOLD_{}".format(seed, fold)

    if(isTrain):
        targetFolder += "/TRAIN"
    else:
        targetFolder += "/TEST"

    print(targetFolder)

    subjects = glob(targetFolder + "/*")

    return subjects

def readSubject(folder):
    attentionMaps = []
    for i in range(4):
        attentionMaps.append(np.load(folder + "/attentionMaps_layer{}.npy".format(i)))
    clsRelevancyMap = np.load(folder + "/clsRelevancyMap.npy")
    label = np.load(folder + "/label.npy")
    inputTokens = np.load(folder + "/token_layerIn.npy")

    return attentionMaps, clsRelevancyMap, label, inputTokens