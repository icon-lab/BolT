from numpy import tri
import pandas as pd
import sys
import numpy as np

sys.path.append("../")

social = None # pd.read_csv("somePath/SOCIAL_run1_TAB.txt", sep='\t')
procedures = social["Procedure"].to_numpy()
types = social["Type"].to_numpy()
trialOnsetTimes = social["ExperimenterWindow.OnsetTime[Block]"].to_numpy()
initialOnsets = social["CountDownSlide.OnsetTime"].to_numpy()
initialFinishes = social["CountDownSlide.FinishTime"].to_numpy()

fixationDuration = 15000


# things to dump
durations = []
procedureNames = []
trialNames = []

restOffset = 4

trialOnsetTimes = trialOnsetTimes[restOffset:]
procedures = procedures[restOffset:]
types = types[restOffset:]


# durations
# initial rest
durations.append(initialFinishes[restOffset-1] - initialOnsets[0])
procedureNames.append("Rest")
trialNames.append("Rest")

currentTask = None

prevTaskDur = None

for i, (procedure, onset) in enumerate(zip(procedures, trialOnsetTimes)):

    if(np.isnan(onset)):

        procedureNames.append("Rest")
        trialNames.append("Rest")
        durations.append(fixationDuration)
    else:
        if(i+2 < len(trialOnsetTimes)):
            duration = trialOnsetTimes[i+2] - onset - fixationDuration
            prevTaskDur = duration
        else:
            duration = prevTaskDur

        procedureNames.append(procedure)
        durations.append(duration)
        trialNames.append(types[i])

procedureNames = np.array(procedureNames)
durations = np.array(durations)
trialNames = np.array(trialNames)