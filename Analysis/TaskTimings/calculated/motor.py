import pandas as pd
import sys
import numpy as np

sys.path.append("../")

motor = None # pd.read_csv("somePath/MOTOR_run1_TAB.txt", sep='\t')
procedures = motor["Procedure[Trial]"]
trialTypes = motor["BlockType"]
trialOnsetTimes = motor["ExperimenterWindow.OnsetTime"]
initialOnsets = motor["CountDownSlide.OnsetTime"]
initialFinishes = motor["CountDownSlide.FinishTime"]



# things to dump
durations = []
procedureNames = []
trialNames = []

restOffset = 4
fixRestDuration = 15000

# durations
# initial rest
durations.append(initialFinishes[restOffset-1] - initialOnsets[0])
procedureNames.append("REST")
trialNames.append("REST")

restDuration = 0

windowOnsets = []

prevTrialName = None

def findNextOnset(i):
    for j in range(len(trialOnsetTimes[i+1:])):
        if(not np.isnan(trialOnsetTimes[j+i+1])):
            return trialOnsetTimes[j+i+1]

for i, trialType in enumerate(trialTypes[restOffset:]):

    if(trialType == "Fixation"):
        durations.append(fixRestDuration)
        procedureNames.append("REST")
        trialNames.append("REST")
        continue

    if(trialType == prevTrialName):
        continue
    else:
        duration = findNextOnset(i+restOffset) - trialOnsetTimes[i+restOffset]
        
        durations.append(duration)
        procedureNames.append(procedures[i+restOffset])
        trialNames.append(trialType)
        prevTrialName = trialType


