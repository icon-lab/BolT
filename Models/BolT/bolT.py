import torch
from torch import nn

import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# import transformers

from Models.BolT.bolTransformerBlock import BolTransformerBlock


class BolT(nn.Module):
    def __init__(self, hyperParams, details):

        super().__init__()

        dim = hyperParams.dim
        nOfClasses = details.nOfClasses

        self.hyperParams = hyperParams

        self.inputNorm = nn.LayerNorm(dim)

        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = []

        shiftSize = int(hyperParams.windowSize * hyperParams.shiftCoeff)
        self.shiftSize = shiftSize
        self.receptiveSizes = []

        for i, layer in enumerate(range(hyperParams.nOfLayers)):
            
            if(hyperParams.focalRule == "expand"):
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * i * hyperParams.fringeCoeff * (1-hyperParams.shiftCoeff))
            elif(hyperParams.focalRule == "fixed"):
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * 1 * hyperParams.fringeCoeff * (1-hyperParams.shiftCoeff))

            print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            self.receptiveSizes.append(receptiveSize)

            self.blocks.append(BolTransformerBlock(
                dim = hyperParams.dim,
                numHeads = hyperParams.numHeads,
                headDim= hyperParams.headDim,
                windowSize = hyperParams.windowSize,
                receptiveSize = receptiveSize,
                shiftSize = shiftSize,
                mlpRatio = hyperParams.mlpRatio,
                attentionBias = hyperParams.attentionBias,
                drop = hyperParams.drop,
                attnDrop = hyperParams.attnDrop
            ))

        self.blocks = nn.ModuleList(self.blocks)


        self.encoder_postNorm = nn.LayerNorm(dim)
        self.classifierHead = nn.Linear(dim, nOfClasses)

        # for token painting
        self.last_numberOfWindows = None

        # for analysis only
        self.tokens = []


        self.initializeWeights()

    def initializeWeights(self):
        # a bit arbitrary
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def calculateFlops(self, T):

        windowSize = self.hyperParams.windowSize
        shiftSize = self.shiftSize
        focalSizes = self.focalSizes
 
        macs = []

        nW = (T-windowSize) // shiftSize  + 1

        C = 400 # for schaefer atlas
        H = self.hyperParams.numHeads
        D = self.hyperParams.headDim

        for l, focalSize in enumerate(focalSizes):

            mac = 0

            # MACS from attention calculation

                # projection in
            mac += nW * (1+windowSize) * C * H * D * 3

                # attention, softmax is omitted
            
            mac += 2 * nW * H * D * (1+windowSize) * (1+focalSize) 

                # projection out

            mac += nW * (1+windowSize) * C * H * D


            # MACS from MLP layer (2 layers with expand ratio = 1)

            mac += 2 * (T+nW) * C * C
        
            macs.append(mac)

        return macs, np.sum(macs) * 2 # FLOPS = 2 * MAC


    def forward(self, roiSignals, analysis=False):
        
        """
            Input : 
            
                roiSignals : (batchSize, N, dynamicLength)

                analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 
            
            Output:

                logits : (batchSize, #ofClasses)


        """


        roiSignals = roiSignals.permute((0,2,1))

        batchSize = roiSignals.shape[0]
        T = roiSignals.shape[1] # dynamicLength

        nW = (T-self.hyperParams.windowSize) // self.shiftSize  + 1
        cls = self.clsToken.repeat(batchSize, nW, 1) # (batchSize, #windows, C)
        
        # record nW and dynamicLength, need in case you want to paint those tokens later
        self.last_numberOfWindows = nW
        
        if(analysis):
            self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        for block in self.blocks:
            roiSignals, cls = block(roiSignals, cls, analysis)
            
            if(analysis):
                self.tokens.append(torch.cat([cls, roiSignals], dim=1))
        
        """
            roiSignals : (batchSize, dynamicLength, featureDim)
            cls : (batchSize, nW, featureDim)
        """

        cls = self.encoder_postNorm(cls)

        if(self.hyperParams.pooling == "cls"):
            logits = self.classifierHead(cls.mean(dim=1)) # (batchSize, #ofClasses)
        elif(self.hyperParams.pooling == "gmp"):
            logits = self.classifierHead(roiSignals.mean(dim=1))

        torch.cuda.empty_cache()

        return logits, cls


