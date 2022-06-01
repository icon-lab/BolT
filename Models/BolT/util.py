import torch
from einops import rearrange, repeat
import numpy as np

def windowBoldSignal(boldSignal, windowLength, stride):
    
    """
        boldSignal : (batchSize, N, T)
        output : (batchSize, (T-windowLength) // stride, N, windowLength )
    """

    T = boldSignal.shape[2]

    # NOW WINDOWING 
    windowedBoldSignals = []
    samplingEndPoints = []

    for windowIndex in range((T - windowLength)//stride + 1):
        
        sampledWindow = boldSignal[:, :, windowIndex * stride  : windowIndex * stride + windowLength]
        samplingEndPoints.append(windowIndex * stride + windowLength)

        sampledWindow = torch.unsqueeze(sampledWindow, dim=1)
        windowedBoldSignals.append(sampledWindow)
        

    windowedBoldSignals = torch.cat(windowedBoldSignals, dim=1)

    return windowedBoldSignals, samplingEndPoints
