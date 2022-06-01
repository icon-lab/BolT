from numpy import tri
import pandas as pd
import sys
import numpy as np

sys.path.append("../")

emotion = None # pd.read_csv("somePath/EMOTION_run1_TAB.txt", sep='\t')
procedures = emotion["Procedure"].to_numpy()
trialOnsetTimes = emotion["ExperimenterWindow.OnsetTime"].to_numpy()
initialOnsets = emotion["SyncSlide.OnsetTime"].to_numpy()
initialFinishes = emotion["SyncSlide.FinishTime"].to_numpy()


# things to dump
durations = []
procedureNames = []
trialNames = []

restOffset = 4

trialOnsetTimes = trialOnsetTimes[restOffset:]
procedures = procedures[restOffset:]

# durations
# initial rest
durations.append(initialFinishes[restOffset-1] - initialOnsets[0])
procedureNames.append("Rest")

currentTask = None

prevTaskDur = None

for i, (procedure, onset) in enumerate(zip(procedures, trialOnsetTimes)):

    taskName = currentTask

    if("Prompt" in procedure):
        if("Shape" in procedure):
            currentTask = "Shape"
        else:
            currentTask = "Face"
        
        taskName = "Rest"

    if(i+1 < len(trialOnsetTimes)):
        duration = trialOnsetTimes[i+1] - onset
        prevTaskDur = duration
    else:
        duration = prevTaskDur

    durations.append(duration)
    procedureNames.append(taskName)

durations = np.array(durations)
procedureNames = np.array(procedureNames)