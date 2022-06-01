from math import comb
import pandas as pd
import sys
import numpy as np

sys.path.append("../")

language = None # pd.read_csv("somePath/LANGUAGE_run1_TAB.txt", sep='\t')
procedures = language["Procedure[Block]"].to_numpy()
responseOnsets = language["ResponsePeriod.OnsetTime"].to_numpy()

presentMathStart = language["PresentMathFile.StartTime"].to_numpy()
presentStoryStart = language["PresentStoryFile.StartTime"].to_numpy()

# things to dump
durations = [91472 - 65543., 105633 - 102675]
procedureNames = ["Story", "Response"]


offset = 2

presentMathStart = presentMathStart[offset:]
presentStoryStart = presentStoryStart[offset:]
procedures = procedures[offset:]
responseOnsets = responseOnsets[offset:]

flagMath = False

prevMathTaskDuration = None
prevMathResponseDuration = None
for i, procedure in enumerate(procedures):

    if(procedure == "PresentChangePROC"):
        continue

    if(procedure == "MathProc" or procedure == "DummyProc"):

        taskDuration = responseOnsets[i] - presentMathStart[i]
        procedureNames.append("Math")
        durations.append(taskDuration)


        if(not i == len(procedures)-1):
        
            if(not np.isnan(presentMathStart[i+1])):
                responseDuration = presentMathStart[i+1] - responseOnsets[i]
            else:
                if(not flagMath):
                    responseDuration = presentStoryStart[i+2] - responseOnsets[i]
                    flagMath = True

                else:
                    responseDuration = presentMathStart[i+2] - responseOnsets[i]
            
            if(np.isnan(responseDuration)):
                print(i, procedures[i], flagMath)

            procedureNames.append("Response")
            durations.append(responseDuration)

    elif(procedure == "StoryProc"):

        taskDuration = responseOnsets[i] - presentStoryStart[i]
        procedureNames.append("Story")
        durations.append(taskDuration)


        if(not i == len(procedures)-1):
        
            if(not np.isnan(presentStoryStart[i+1])):
                responseDuration = presentStoryStart[i+1] - responseOnsets[i]
            else:
                responseDuration = presentMathStart[i+2] - responseOnsets[i]

            procedureNames.append("Response")
            durations.append(responseDuration)


durations = np.array(durations)
procedureNames = np.array(procedureNames)