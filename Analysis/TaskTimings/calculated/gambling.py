import pandas as pd
import sys
import numpy as np

sys.path.append("../")

gambling = None # pd.read_csv("somePath/GAMBLING_run1_TAB.txt", sep='\t')
procedures = gambling["Procedure[Trial]"]
trialTypes = gambling["TrialType"]
trialOnsetTimes = gambling["ExperimenterWindow.OnsetTime"]
initialOnsets = gambling["SyncSlide.OnsetTime"]
initialFinishes = gambling["SyncSlide.FinishTime"]
oneSecFixationOnsets = gambling["OneSecFixation.OnsetTime"]

fifteenFixationOnsets = gambling["FifteenSecFixation.OnsetTime"]


# things to dump
durations = []
procedureNames = []
trialNames = []

restOffset = 4
fixationDuration = 15000 + 500

# durations
# initial rest
durations.append(initialFinishes[restOffset-1] - initialOnsets[0] + 500)
procedureNames.append("REST")
trialNames.append("REST")

restDuration = 0

for i, trialOnset in enumerate(trialOnsetTimes[restOffset:]):
	onsecOnset = oneSecFixationOnsets[i+restOffset]	
	
	procedure = procedures[i+restOffset]
	trialName = trialTypes[i+restOffset]


	if(np.isnan(trialOnset)):


		if(i+restOffset+1 < len(trialOnsetTimes)):
			restDuration =  fixationDuration

		durations.append(restDuration)
		procedureNames.append("Rest")
		trialNames.append("Rest")
			
	else:

		if(np.isnan(trialOnsetTimes[i+restOffset+1])):
			
			trialDuration = fifteenFixationOnsets[i+restOffset+1] - trialOnset
		else:
		
			trialDuration = trialOnsetTimes[i+restOffset+1] - trialOnset

		durations.append(trialDuration)
		procedureNames.append(procedure)
		trialNames.append(trialName)


durations = np.array(durations)
procedureNames = np.array(procedureNames)
trialNames = np.array(trialNames)