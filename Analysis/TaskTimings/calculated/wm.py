import pandas as pd
import sys
import numpy as np

sys.path.append("../")

wm = None # pd.read_csv("somePath/WM_run1_TAB.txt", sep='\t')
procedures = wm["Procedure[Block]"]
trialTypes = wm["TargetType"]
trialOnsetTimes = wm["ExperimenterWindow.OnsetTime"]
initialOnsets = wm["SyncSlide.OnsetTime"]
initialFinishes = wm["SyncSlide.FinishTime"]



# things to dump
durations = []
procedureNames = []
trialNames = []

restOffset = 4
cueDuration = 2500
fixRestDuration = 15000

# durations
# initial rest
durations.append(initialFinishes[restOffset-1] - initialOnsets[0])
procedureNames.append("REST")
trialNames.append("REST")

restDuration = 0

for i, trialOnset in enumerate(trialOnsetTimes[restOffset:]):

	
	procedure = procedures[i+restOffset]
	trialName = trialTypes[i+restOffset]

	if(np.isnan(trialOnset)):

		if("Cue" in procedure):
			restDuration = cueDuration
		else:
			restDuration = fixRestDuration

		durations.append(restDuration)
		procedureNames.append("REST")
		trialNames.append("REST")
			
	else:
		if(i+restOffset+1 < len(trialOnsetTimes) and not np.isnan(trialOnsetTimes[restOffset+i+1])):
			trialDuration = trialOnsetTimes[i+restOffset+1] - trialOnset

		durations.append(trialDuration)
		procedureNames.append(procedure)
		trialNames.append(trialName)



