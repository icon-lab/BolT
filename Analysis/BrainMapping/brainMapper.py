from brainRegressor import brainRegressor, saveImportanceResults
from impTokenExtractor import tokenExtractor
import torch
import os
import numpy as np
import argparse
import shutil
import sys

os.chdir("../../")
sys.path.append("./")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="abide1")
argv = parser.parse_args()



saveFolder = "./Analysis/DataExtracted/{}/Results".format(argv.dataset)
if(os.path.exists(saveFolder)):
    shutil.rmtree()
os.makedirs(saveFolder, exist_ok=True)

if(argv.dataset == "abide1"):
    targetSeeds = [0]#,1,2,3,4] 
    startKs = [0]#, 10, 15, 20, 25, 30, 35, 40, 45, 50]
elif(argv.dataset == "hcpRest"):
    targetSeeds = [0]
    startKs = [0]#, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
elif(argv.dataset == "hcpTask"):
    targetSeeds = [0]
    startKs = [0]#, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


topK = 5


results_mean = []
results_std = []

testAccs_s_k = []
averageRelevancies_s_k = []
allImportance_0_k = []
allImportance_1_k = []
allImportances_k = []

for startK in startKs:
    
    testAccs_s = []
    averageRelevancies_s = []

    allImportances_s = []

    for seed in targetSeeds:
        tokenExtractor(argv.dataset, seed, topK, startK)
        test_accuracies_static, averageRelevancies, allImportances = brainRegressor(argv.dataset, seed, startK)

        testAccs_s.append(test_accuracies_static)
        averageRelevancies_s.append(averageRelevancies)

        allImportances_s.append(allImportances)

    testAccs_s_k.append(testAccs_s)
    averageRelevancies_s_k.append(averageRelevancies_s)

    allImportances_k.append(np.array(allImportances_s).mean(axis=0))

allImportances_final = np.array(allImportances_k).mean(axis=0)

saveImportanceResults(allImportances_final, saveFolder)


pandaResults_acc = []
pandaResults_relScore = []

for i, testAccs_s in enumerate(testAccs_s_k):
    for testAccs in testAccs_s:    
        for testAcc in testAccs:
            pandaResults_acc.append([startKs[i], testAcc])

for i, averageRelevancies_s in enumerate(averageRelevancies_s_k):
    for averageRelevancies in averageRelevancies_s:    
        for averageRelevancy in averageRelevancies:
            pandaResults_relScore.append([startKs[i], averageRelevancy])
                
import pandas as pd
import torch


pandaResults_acc = pd.DataFrame(data = pandaResults_acc, columns=["Group Index", "Accuracy"])
pandaResults_relScore = pd.DataFrame(data = pandaResults_relScore, columns=["Group Index", "Relevancy Score"])



torch.save(testAccs_s_k, saveFolder+"/testAccs_s_k.save")
torch.save(pandaResults_acc, saveFolder+"/pandaResults_acc.save")
torch.save(pandaResults_relScore, saveFolder+"/pandaResults_relScore.save")



figSave = "./Analysis/Figures/{}".format(argv.dataset)

import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale = 2)

fig1 = sns.relplot(data=pandaResults_acc, x="Group Index", y="Accuracy", color="b", kind="line", aspect=12/8.0)
fig1.savefig(figSave + "/{}_acc.png".format(argv.dataset), dpi=600)

fig2 = sns.relplot(data=pandaResults_relScore, x="Group Index", y="Relevancy Score", color="r", kind="line", aspect=12/8.0)
fig2.savefig(figSave + "/{}_relScore.png".format(argv.dataset), dpi=600)















