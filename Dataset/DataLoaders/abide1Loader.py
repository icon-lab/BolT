import torch
import numpy as np
import os
import sys



datadir = "./Dataset/Data"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    

def abide1Loader(atlas, targetTask):

    """
        x : (#subjects, N)
    """

    dataset = torch.load(datadir + "/dataset_abide_{}.save".format(atlas))

    x = []
    y = []
    subjectIds = []

    for data in dataset:
        
        if(targetTask == "disease"):
            label = int(data["pheno"]["disease"]) - 1 # 0 for autism 1 for control

        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):

            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))

    return x, y, subjectIds
