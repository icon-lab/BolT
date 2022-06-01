from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm

import os

from subjectReader import getSubjects, readSubject


def tokenExtractor(dataset, seed, topK, startK):

    targetDataset = dataset

    foldCount = 5
    if(targetDataset == "abide1"):
        foldCount = 10

    targetFolds = range(foldCount)    


    for targetFold in targetFolds:

        saveFolder_train = "./Analysis/DataExtracted/{}/seed_{}/{}/TRAIN/startK_{}".format(targetDataset, seed, targetFold, startK)
        saveFolder_test = "./Analysis/DataExtracted/{}/seed_{}/{}/TEST/startK_{}".format(targetDataset, seed, targetFold, startK)

        if(os.path.exists(saveFolder_train + "/x_train_static.npy") and os.path.exists(saveFolder_test + "/x_test_static.npy")):
            continue

        os.makedirs(saveFolder_train, exist_ok=True)
        os.makedirs(saveFolder_test, exist_ok=True)


        trainSubjectDirs = getSubjects(targetDataset, seed, targetFold, True)
        testSubjectDirs = getSubjects(targetDataset, seed, targetFold, False)

        train_random_subjIds = []
        test_random_subjIds = []

        x_train_random = []
        x_train_random_relevancyScore = []

        y_train_random = []

        x_test_random = []
        
        x_test_random_relevancyScore = []

        y_test_random = []



        train_static_subjIds = []
        test_static_subjIds = []

        x_train_static = []
        x_train_static_relevancyScore = []

        y_train_static = []

        x_test_static = []
        x_test_static_relevancyScore = []

        y_test_static = []


        print("Extracting train subjects...")

        for subjectDir in tqdm(trainSubjectDirs, ncols=60):

            subjId = subjectDir.split("/")[-1]

            attentionMaps, clsRelevancyMap, label, inputTokens = readSubject(subjectDir)
            attentionMap = attentionMaps[-1].mean(axis=1)
        

            clsRelevancyMap = clsRelevancyMap.mean(axis=0)

            if(startK + topK <= clsRelevancyMap.shape[0]):

                if(startK != 0):
                    target_ind_static = np.argsort(clsRelevancyMap)[-startK - topK: -startK]
                else:
                    target_ind_static = np.argsort(clsRelevancyMap)[-topK:]

            else: 

                target_ind_static = np.argsort(clsRelevancyMap)[0:topK]

            # STATIC
            targetTokens = inputTokens[target_ind_static]
            
            averageRelScore = np.mean(clsRelevancyMap[target_ind_static]) / np.min(clsRelevancyMap)

            for token in targetTokens:
                x_train_static.append(token)
                y_train_static.append(label)
                x_train_static_relevancyScore.append(averageRelScore)
                train_static_subjIds.append(subjId)

            # RANDOM
            randomIdx = np.random.choice(range(len(clsRelevancyMap)), len(targetTokens))
            averageRelScore = np.mean(clsRelevancyMap[randomIdx]) / np.min(clsRelevancyMap)

            for idx in randomIdx:
                x_train_random.append(inputTokens[idx])
                y_train_random.append(label)
                x_train_random_relevancyScore.append(averageRelScore)
                train_random_subjIds.append(subjId)

                    

        print("Extracting test subjects...")
        for subjectDir in tqdm(testSubjectDirs, ncols=60):
            attentionMaps, clsRelevancyMap, label, inputTokens = readSubject(subjectDir)
            attentionMap = attentionMaps[-1].mean(axis=1)

            subjId = subjectDir.split("/")[-1]
            

            clsRelevancyMap = clsRelevancyMap.mean(axis=0)


            if(startK + topK <= clsRelevancyMap.shape[0]):

                if(startK != 0):
                    target_ind_static = np.argsort(clsRelevancyMap)[-startK - topK: -startK]
                else:
                    target_ind_static = np.argsort(clsRelevancyMap)[-topK:]
            else: 
                target_ind_static = np.argsort(clsRelevancyMap)[0:topK]                    

            # STATIC
            targetTokens = inputTokens[target_ind_static]    
            averageRelScore = np.mean(clsRelevancyMap[target_ind_static]) / np.min(clsRelevancyMap)

            for token in targetTokens:
                x_test_static.append(token)
                y_test_static.append(label)
                x_test_static_relevancyScore.append(averageRelScore)
                test_static_subjIds.append(subjId)

            # RANDOM
            randomIdx = np.random.choice(range(len(clsRelevancyMap)), len(targetTokens))
            averageRelScore = np.mean(clsRelevancyMap[randomIdx]) / np.min(clsRelevancyMap)

            for idx in randomIdx:
                x_test_random.append(inputTokens[idx])
                y_test_random.append(label)        
                x_test_random_relevancyScore.append(averageRelScore)
                test_random_subjIds.append(subjId)

                        
        np.save(saveFolder_train + "/x_train_static.npy", x_train_static)
        np.save(saveFolder_train + "/x_train_static_relevancyScore.npy", x_train_static_relevancyScore)
        np.save(saveFolder_train + "/x_train_random.npy", x_train_random)
        np.save(saveFolder_train + "/x_train_random_relevancyScore.npy", x_train_random_relevancyScore)
        np.save(saveFolder_train + "/y_train_static.npy", y_train_static)
        np.save(saveFolder_train + "/y_train_random.npy", y_train_random)                

        np.save(saveFolder_test + "/x_test_static.npy", x_test_static)
        np.save(saveFolder_test + "/x_test_static_relevancyScore.npy", x_test_static_relevancyScore)
        np.save(saveFolder_test + "/x_test_random.npy", x_test_random)
        np.save(saveFolder_test + "/x_test_random_relevancyScore.npy", x_test_random_relevancyScore)
        np.save(saveFolder_test + "/y_test_static.npy", y_test_static)
        np.save(saveFolder_test + "/y_test_random.npy", y_test_random)


        np.save(saveFolder_train + "/train_random_subjIds.npy", train_random_subjIds)
        np.save(saveFolder_test + "/test_random_subjIds.npy", test_random_subjIds)

        np.save(saveFolder_train + "/train_static_subjIds.npy", train_static_subjIds)
        np.save(saveFolder_test + "/test_static_subjIds.npy", test_static_subjIds)


    #clf = LogisticRegression(random_state=0).fit(x_train, y_train)