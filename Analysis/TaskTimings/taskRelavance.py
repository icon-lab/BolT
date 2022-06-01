import numpy as np
import torch
import os
import pandas as pd

from PIL import Image
import numpy as np
from matplotlib import cm

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

matplotlib.use('Agg')



all_labeled_relevanceMaps = None # here you need to load relevancymaps averaged across all subjects
all_labeled_inputTokens = None # here you need to load ROI signals averaged across all subjects

task_block_timing = []

TR = 740

def durationsToTaskLabels(durations, labels):

    sampledLabels = []

    carry = 0

    for label, duration in zip(labels, durations):
        
        sampleCount = int((duration - carry) // TR + 1)
        carry = int(TR - (duration - carry) % TR)

        for i in range(sampleCount):
            sampledLabels.append(label)


    sampledTasks = {}

    sampledLabels = np.array(sampledLabels)

    for taskName in np.unique(sampledLabels):
        sampledTasks[taskName] = sampledLabels == taskName

    #print(sampledLabels)
    
    
    return sampledTasks

# these durations are obtained from the scrips in the calculated folder
hardCodedDurations = {
    "GAMBLING" : [ 8486.,  3545.,  3598.,  3546.,  3612.,  3585.,  3572.,  3665.,
        3186., 15500.,  3425.,  3665.,  3626.,  3558.,  3679.,  3585.,
        3745.,  3106., 15500.,  3439.,  3625.,  3665.,  3585.,  3546.,
        3678.,  3639.,  3199., 15500.,  3545.,  3666.,  3571.,  3613.,
        3611.,  3599.,  3585.,  3279., 15500.],

    "WM" : [ 7982.,  2500.,  2559.,  2559.,  2572.,  2572.,  2559.,  2559.,
        2559.,  2559.,  2559.,  2559.,  2500.,  2572.,  2572.,  2573.,
        2572.,  2572.,  2573.,  2572.,  2572.,  2572.,  2572., 15000.,
        2500.,  2559.,  2559.,  2559.,  2559.,  2559.,  2559.,  2559.,
        2559.,  2559.,  2559.,  2500.,  2559.,  2559.,  2559.,  2572.,
        2559.,  2559.,  2559.,  2559.,  2559.,  2559., 15000.,  2500.,
        2573.,  2572.,  2572.,  2573.,  2572.,  2559.,  2572.,  2559.,
        2572.,  2572.,  2500.,  2572.,  2573.,  2572.,  2572.,  2572.,
        2573.,  2572.,  2572.,  2573.,  2573., 15000.,  2500.,  2559.,
        2559.,  2559.,  2559.,  2559.,  2559.,  2559.,  2559.,  2559.,
        2559.,  2500.,  2573.,  2572.,  2572.,  2559.,  2559.,  2572.,
        2573.,  2559.,  2572.,  2572., 15000.],
    
    "MOTOR" : [ 7983., 15127., 15127., 15000., 15114., 15127., 15127., 15000.,
       15127., 15127., 15127., 15128., 15127., 15000.],

    "LANGUAGE" : [25929.,  2958.,  9666.,  3197.,  9704.,  3195.,  9440.,  4576.,
       21933.,  3171., 21046.,  3201., 18497.,  4510.,  9591.,  3212.,
        9330.,  3238.,  9637.,  4536.,  9884.,  3232., 10483.,  3220.,
        9348.,  3296., 10925.],

    "EMOTION" : [7977., 5024., 3012., 2999., 3012., 3012., 3012., 1000., 5011.,
       3012., 3013., 3012., 3012., 3012.,  999., 5012., 3012., 3012.,
       3012., 3012., 3012., 1000., 5011., 3012., 3012., 3012., 3013.,
       3012.,  999., 5012., 3012., 3012., 3012., 3012., 3012., 1000.,
       5011., 3012., 3012., 3012.],

    "SOCIAL" : [ 7774., 23024., 15000., 23011., 15000., 23025., 15000., 23024.,
       15000., 23024., 15000.],

    "RELATIONAL" : [ 7983.,  1999.,  4119.,  4118.,  4118.,  4118.,  1935.,  3318.,
        3319.,  3319.,  3318.,  3318., 16000.,  1994.,  3318.,  3319.,
        3318.,  3319.,  3319.,  1949.,  4119.,  4118.,  4132.,  4132.,
       16000.,  1995.,  3318.,  3332.,  3320.,  3318.,  3318.,  1954.,
        4118.,  4119.,  4118.,  4118., 16000.]
    

}


hardCodedTrialNames = {

    "GAMBLING" : ['Rest', 'Reward', 'Reward', 'Punishment', 'Reward', 'Reward',
       'Reward', 'Neutral', 'Reward', 'Rest', 'Reward', 'Punishment',
       'Punishment', 'Punishment', 'Neutral', 'Punishment', 'Punishment',
       'Punishment', 'Rest', 'Punishment', 'Punishment', 'Punishment',
       'Neutral', 'Punishment', 'Neutral', 'Punishment', 'Punishment',
       'Rest', 'Reward', 'Reward', 'Punishment', 'Reward', 'Punishment',
       'Reward', 'Reward', 'Reward', 'Rest'],

    "WM" : ['Rest', 'Rest', 'nonlure', 'nonlure', 'nonlure', 'target',
       'nonlure', 'nonlure', 'lure', 'nonlure', 'lure', 'target', 'Rest',
       'nonlure', 'target', 'nonlure', 'nonlure', 'lure', 'target',
       'nonlure', 'lure', 'lure', 'nonlure', 'Rest', 'Rest', 'nonlure',
       'nonlure', 'nonlure', 'target', 'lure', 'lure', 'nonlure',
       'nonlure', 'target', 'nonlure', 'Rest', 'target', 'nonlure',
       'nonlure', 'nonlure', 'lure', 'nonlure', 'lure', 'lure', 'target',
       'nonlure', 'Rest', 'Rest', 'nonlure', 'lure', 'nonlure', 'target',
       'nonlure', 'nonlure', 'target', 'nonlure', 'lure', 'lure', 'Rest',
       'nonlure', 'nonlure', 'target', 'nonlure', 'lure', 'target',
       'nonlure', 'lure', 'nonlure', 'nonlure', 'Rest', 'Rest', 'nonlure',
       'nonlure', 'lure', 'nonlure', 'nonlure', 'nonlure', 'lure',
       'target', 'lure', 'target', 'Rest', 'nonlure', 'lure', 'nonlure',
       'nonlure', 'target', 'target', 'nonlure', 'nonlure', 'lure',
       'nonlure', 'Rest'],

    "MOTOR" : ['Rest', 'Left Hand', 'Right Foot', 'Rest', 'Tongue', 'Left Foot',
       'Right Hand', 'Rest', 'Left Hand', 'Tongue', 'Right Foot',
       'Right Hand', 'Left Foot', 'Rest'],

    "SOCIAL" : ['Rest', 'Mental', 'Rest', 'Random', 'Rest', 'Random', 'Rest',
       'Mental', 'Rest', 'Random', 'Rest'],

    "RELATIONAL" : ['Rest', 'Rest', 'Relational', 'Relational', 'Relational',
       'Relational', 'Rest', 'Control', 'Control', 'Control', 'Control',
       'Control', 'Rest', 'Rest', 'Control', 'Control', 'Control',
       'Control', 'Control', 'Rest', 'Relational', 'Relational',
       'Relational', 'Relational', 'Rest', 'Rest', 'Control', 'Control',
       'Control', 'Control', 'Control', 'Rest', 'Relational',
       'Relational', 'Relational', 'Relational', 'Rest'],

    "EMOTION" : ['Rest', 'Rest', 'Shape', 'Shape', 'Shape', 'Shape', 'Shape',
       'Shape', 'Rest', 'Face', 'Face', 'Face', 'Face', 'Face', 'Face',
       'Rest', 'Shape', 'Shape', 'Shape', 'Shape', 'Shape', 'Shape',
       'Rest', 'Face', 'Face', 'Face', 'Face', 'Face', 'Face', 'Rest',
       'Shape', 'Shape', 'Shape', 'Shape', 'Shape', 'Shape', 'Rest',
       'Face', 'Face', 'Face'],


    "LANGUAGE" : ['Story', 'Response', 'Math', 'Response', 'Math', 'Response',
       'Math', 'Response', 'Story', 'Response', 'Story', 'Response',
       'Story', 'Response', 'Math', 'Response', 'Math', 'Response',
       'Math', 'Response', 'Math', 'Response', 'Math', 'Response', 'Math',
       'Response', 'Math'],

}

def arrayToPILImage(array):

    # normalize to 0 and 1
    array += np.abs(np.min(array))
    array /= np.max(array)

    #apply colormap here
    imageArray = cm.magma(array)
    image = Image.fromarray(np.uint8(imageArray * 255))

    return image

labelToName = {
    0 : "EMOTION",
    1 : "GAMBLING",
    2 : "LANGUAGE",
    3 : "MOTOR",
    4 : "RELATIONAL",
    5 : "SOCIAL",
    6 : "WM"
}

""" offsetDict = {
    "EMOTION" : 4,
    "GAMBLING" : 1,
    "LANGUAGE" : 0,
    "MOTOR" : 3,
    "RELATIONAL" : 2,
    "SOCIAL" : 2,
    "WM" : 4
}
 """
offsetDict = {
    "EMOTION" : 0,
    "GAMBLING" : 0,
    "LANGUAGE" : 0,
    "MOTOR" : 0,
    "RELATIONAL" : 0,
    "SOCIAL" : 0,
    "WM" : 0
}

paddingDict = {
    "EMOTION" : 0,
    "GAMBLING" : 0,
    "LANGUAGE" : 0,
    "MOTOR" : 0,
    "RELATIONAL" : 0,
    "SOCIAL" : 0,
    "WM" : 0    
}

widthDict = {
    "EMOTION" : 400,
    "GAMBLING" : 400,
    "LANGUAGE" : 800,
    "MOTOR" : 400,
    "RELATIONAL" : 400,
    "SOCIAL" : 400,
    "WM" : 500
}


taskToColor = {

    # for EMOTION

    "Rest" : [1,0,0],
    "Shape" : [0,1,0],
    "Face" : [0.5,0,0.5],

    # for GAMBLING
    "Task" : [0,1,0],
    "Reward" : [0,1,1],
    "Punishment" : [1,0.8,0],
    "Neutral" : [1,0,1],
    
    # for LANGUAGE
    "Story" : [0,1,0],
    "Math" : [0.5,0,0.5],
    "Response" : [0.5,0.5,0.5],

    # for MOTOR
    "Left Hand" : [0,1,0],
    "Left Foot" : [0.5,0,0.5],
    "Right Hand" : [0.5,0.5,0.5],
    "Right Foot" : [1,0,1],
    "Tongue" : [0,1,1],

    # for RELATIONAL
    "Relational" : [0,1,1],
    "Control" : [1,0,1],


    # for SOCIAL
    "Mental" : [0,1,0],
    "Random" : [0.5,0,0.5],


    # for WORKING MEMORY
    "lure" : [0,1,0],
    "nonlure" : [0,1,1],
    "target" : [1,0,1]

}


averaged_labeled_relevanceMaps = {}

os.makedirs("./TaskTimingPaintings/", exist_ok=True)

for label in all_labeled_relevanceMaps.keys():


    target_relevanceMaps = all_labeled_relevanceMaps[label]
    target_inputTokens = all_labeled_inputTokens[label]
    
    willAverage_relevanceMap = []
    willAverage_inputTokens = []

    maxTimePoint = 0

    timingHeight = 20
    width = widthDict[labelToName[label]]

    for relevanceMap in target_relevanceMaps:
        if(relevanceMap.shape[-1] > maxTimePoint):
            maxTimePoint = relevanceMap.shape[-1]

    for i, relevanceMap in enumerate(target_relevanceMaps):
        inputTokens = target_inputTokens[i]
        if(relevanceMap.shape[-1] != maxTimePoint):
            continue
        willAverage_relevanceMap.append(relevanceMap.mean(axis=0))
        willAverage_inputTokens.append(inputTokens[:maxTimePoint])        

    print("Task {} max length = {}".format(labelToName[label], maxTimePoint))

    willAverage_relevanceMap = np.stack(willAverage_relevanceMap, axis=0).mean(0)
    willAverage_inputTokens = np.stack(willAverage_inputTokens, axis=0).mean(0)

    image = arrayToPILImage(willAverage_inputTokens.T.copy())
    image.save("./TaskTimingPaintings/{}_originalInputTokens.png".format(labelToName[label]))

    weighted_inputTokens = willAverage_relevanceMap[None,:] * willAverage_inputTokens.T
    image = arrayToPILImage(weighted_inputTokens.copy())
    image.save("./TaskTimingPaintings/{}_weightedInputTokens.png".format(labelToName[label]))

    willAverage_relevanceMap = willAverage_relevanceMap[None, :].repeat(timingHeight * 3, axis=0)
    image = arrayToPILImage(willAverage_relevanceMap.copy())
    image.save("./TaskTimingPaintings/{}_averageAttendedPoints.png".format(labelToName[label]))


    taskArray = []

    sampledTasks = durationsToTaskLabels(hardCodedDurations[labelToName[label]], hardCodedTrialNames[labelToName[label]])

    for task in list(sampledTasks.keys()):
        pixel = np.array([taskToColor[task]])
        taskArr = np.array(sampledTasks[task])

        print("label = {}, maxLen = {}, maxRel = {}".format(labelToName[label], len(taskArr), maxTimePoint))
        pixels = taskArr[:maxTimePoint][:, None] * pixel
        
        for i, pixel in enumerate(pixels):

            if(np.sum(pixel) == 0):
                pixels[i] = [1,1,1]

        pixels = pixels[None, :].repeat(timingHeight, axis=0)
        taskArray.append(pixels)
    
    taskArray = np.concatenate(taskArray, axis=0)



    taskImage = Image.fromarray(np.uint8(taskArray * 255))
    taskImage.save("./TaskTimingPaintings/{}_official.png".format(labelToName[label]))

