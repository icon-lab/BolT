import re
from numpy import tri
import pandas as pd
import sys
import numpy as np

sys.path.append("../")

relational = None # pd.read_csv("somePath/RELATIONAL_run1_TAB.txt", sep='\t')
procedures = relational["Procedure"].to_numpy()
trials = relational["BlockType"].to_numpy()
trialOnsetTimes = relational["ExperimenterWindow.OnsetTime"].to_numpy()
initialOnsets = relational["SyncSlide.OnsetTime"].to_numpy()
initialFinishes = relational["SyncSlide.FinishTime"].to_numpy()

relationalPromptOnsets = relational["RelationalPrompt.StartTime"].to_numpy()
relationalPromptFinishes = relational["RelationalPrompt.FinishTime"].to_numpy()

controlPromptOnsets = relational["ControlPrompt.OnsetTime"].to_numpy()
controlPromptFinishes = relational["ControlPrompt.FinishTime"].to_numpy()

# things to dump
durations = []
procedureNames = []
trialNames = []

fixationDuration = 16000

restOffset = 4

trialOnsetTimes = trialOnsetTimes[restOffset:]
procedures = procedures[restOffset:]
trials = trials[restOffset:]

relationalPromptOnsets = relationalPromptOnsets[restOffset:]
relationalPromptFinishes = relationalPromptFinishes[restOffset:]
controlPromptOnsets = controlPromptOnsets[restOffset:]
controlPromptFinishes = controlPromptFinishes[restOffset:]

# durations
# initial rest
durations.append(initialFinishes[restOffset-1] - initialOnsets[0])
procedureNames.append("Rest")
trialNames.append("Rest")

previousTrialDuration = None

for i in range(len(trialOnsetTimes)):

    trialOnset = trialOnsetTimes[i]
    procedure = procedures[i]
    trial = trials[i]

    if("Fixation" in procedure):
        durations.append(fixationDuration)
        procedureNames.append("Rest")
        trialNames.append("Rest")
        continue

    if("ControlPromptPROC" in procedure):
        durations.append(controlPromptFinishes[i] - controlPromptOnsets[i])
        procedureNames.append("ControlPrompt")
        trialNames.append("Rest")
        continue

    if("RelationalPromptPROC" in procedure):
        durations.append(relationalPromptFinishes[i] - relationalPromptOnsets[i])
        procedureNames.append("RelationalPrompt")    
        trialNames.append("Rest")
        continue

    else:

        procedureNames.append(procedures[i])
        trialNames.append(trials[i])

        if(not np.isnan(trialOnsetTimes[i+1])):

            duration = trialOnsetTimes[i+1] - trialOnsetTimes[i]
            previousTrialDuration = duration
        else:
            duration = previousTrialDuration

        durations.append(duration)

        continue

durations = np.array(durations)
procedureNames = np.array(procedureNames)
trialNames = np.array(trialNames)

