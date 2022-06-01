from re import sub
import numpy as np
import nilearn as nil
import nilearn.image
import nilearn.datasets
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib
from statistics import mode

from sklearn.linear_model import LogisticRegression


matplotlib.use('Agg')

roi = nil.datasets.fetch_atlas_schaefer_2018()
roiMap = nil.image.load_img(roi["maps"])
roiMap_array = roiMap.get_fdata()    


def getSubjectwiseAccuracy(y_predicts, ys, subjectIds):

    results_subjectWise = {}

    for i, y in enumerate(ys):
        subjectId = subjectIds[i]
        yPredict = y_predicts[i]

        if(not subjectId in results_subjectWise):
            results_subjectWise[subjectId] = {"groundTruth" : y, "predictions" : []}

        results_subjectWise[subjectId]["predictions"].append(yPredict)

    totalSubject = len(results_subjectWise.keys())

    totalCorrectPredicts = 0

    for subjectId in results_subjectWise:
        
        if(mode(results_subjectWise[subjectId]["predictions"]) == results_subjectWise[subjectId]["groundTruth"]):
            totalCorrectPredicts += 1

    return totalCorrectPredicts / totalSubject

def saveImportanceResults(importances, saveFolderName):

    for i, importance in enumerate(importances):


        np.save(saveFolderName + "/importance_{}.npy".format(i), importance)

        plt.figure()
        plt.plot(importance)
        plt.savefig(saveFolderName + "/importance_{}.png".format(i))

        topPercent = 2

        topRois = np.nonzero(importance>np.percentile(importance,100-topPercent))[0]

        # first save for female, ladies first i guess
        os.makedirs(saveFolderName, exist_ok=True)
        for j, roi in enumerate(topRois):
            volume = np.zeros_like(roiMap_array)
            impact = importance[roi]
            volume[roiMap_array==(roi+1)] = 1 # constant coloring of all important rois
            savePrepend = "class_{}_roi_{}_top_{}.nii.gz".format(i, roi+1, len(importance) - sorted(importance.tolist()).index(impact))
            nil.image.new_img_like(roiMap, volume, roiMap.affine).to_filename(saveFolderName + "/" + savePrepend)
            

def generateImportanceFromCoefs(coefs):
    
    importances = []
    
    if(coefs.shape[0] == 1): # binary classification
        coef = coefs[0]
        logOdds = np.exp(coef)

        importance_0 = 1-logOdds
        importance_1 = logOdds-1

        importances.append(importance_0)
        importances.append(importance_1)

    else:

        for coef in coefs:
            logOdds = np.exp(coef)
            importance = logOdds-1

            importances.append(importance)

    return importances

def brainRegressor(dataset, seed, startK):


    targetDataset = dataset


    foldCount = 5
    if(targetDataset == "abide1"):
        foldCount = 10

    targetFolds = range(foldCount)

    relevancyScores = []

    allImportances = []

    train_accuracies_static_subjectWise = []
    test_accuracies_static_subjectWise = []

    train_accuracies_random_subjectWise = []
    test_accuracies_random_subjectWise = []

    train_accuracies_point_subjectWise = []
    test_accuracies_point_subjectWise = []

    train_accuracies_static_tokenWise = []
    test_accuracies_static_tokenWise = []

    train_accuracies_random_tokenWise = []
    test_accuracies_random_tokenWise = []

    train_accuracies_point_tokenWise = []
    test_accuracies_point_tokenWise = []


    for targetFold in targetFolds:

        saveFolder_train = "./Analysis/DataExtracted/{}/seed_{}/{}/TRAIN/startK_{}".format(targetDataset, seed, targetFold, startK)
        saveFolder_test = "./Analysis/DataExtracted/{}/seed_{}/{}/TEST/startK_{}".format(targetDataset, seed, targetFold, startK)
        saveFolder_results = "./Analysis/DataExtracted/{}/seed_{}/{}/RESULTS/startK_{}".format(targetDataset, seed, targetFold, startK) 

        os.makedirs(saveFolder_train, exist_ok=True)
        os.makedirs(saveFolder_test, exist_ok=True)
        os.makedirs(saveFolder_results, exist_ok=True)


        train_static_subjIds = np.load(saveFolder_train + "/train_static_subjIds.npy")
        test_static_subjIds = np.load(saveFolder_test + "/test_static_subjIds.npy")

        train_random_subjIds = np.load(saveFolder_train + "/train_random_subjIds.npy")
        test_random_subjIds = np.load(saveFolder_test + "/test_random_subjIds.npy")



        x_train_static = np.array(np.load(saveFolder_train + "/x_train_static.npy"))
        x_train_static_relevancyScore = np.array(np.load(saveFolder_train + "/x_train_static_relevancyScore.npy"))
        
        x_train_random = np.array(np.load(saveFolder_train + "/x_train_random.npy"))
        x_train_random_relevancyScore = np.array(np.load(saveFolder_train + "/x_train_random_relevancyScore.npy"))


        y_train_static = np.array(np.load(saveFolder_train + "/y_train_static.npy"))
        y_train_random = np.array(np.load(saveFolder_train + "/y_train_random.npy"))    

        x_test_static = np.array(np.load(saveFolder_test + "/x_test_static.npy"))
        x_test_static_relevancyScore = np.array(np.load(saveFolder_test + "/x_test_static_relevancyScore.npy"))

        x_test_random = np.array(np.load(saveFolder_test + "/x_test_random.npy"))
        x_test_random_relevancyScore = np.array(np.load(saveFolder_test + "/x_test_random_relevancyScore.npy"))
    
        y_test_static = np.array(np.load(saveFolder_test + "/y_test_static.npy"))
        y_test_random = np.array(np.load(saveFolder_test + "/y_test_random.npy"))

        # FOR STATIC

        clf_static = LogisticRegression().fit(x_train_static, y_train_static)
        static_train_acc_tokenWise = clf_static.score(x_train_static, y_train_static)
        static_test_acc_tokenWise = clf_static.score(x_test_static, y_test_static)
        
        y_predict_train_static = clf_static.predict(x_train_static)
        y_predict_test_static = clf_static.predict(x_test_static)

        static_train_acc_subjectWise = getSubjectwiseAccuracy(y_predict_train_static, y_train_static, train_static_subjIds)
        static_test_acc_subjectWise = getSubjectwiseAccuracy(y_predict_test_static, y_test_static, test_static_subjIds)        

        print("Rel - Train accuracy token wise : {}, train accuracy subject wise : {}, Test accuracy token wise : {}, test accuracy subject wise : {}".format(static_train_acc_tokenWise, static_train_acc_subjectWise, static_test_acc_tokenWise, static_test_acc_subjectWise))
        train_accuracies_static_subjectWise.append(static_train_acc_subjectWise)
        test_accuracies_static_subjectWise.append(static_test_acc_subjectWise)
        train_accuracies_static_tokenWise.append(static_train_acc_tokenWise)
        test_accuracies_static_tokenWise.append(static_test_acc_tokenWise)
        # FOR RANDOM

        clf_random = LogisticRegression().fit(x_train_random, y_train_random)
        
        random_train_acc_tokenWise = clf_random.score(x_train_random, y_train_random)
        random_test_acc_tokenWise = clf_random.score(x_test_random, y_test_random)    

        y_predict_train_random = clf_random.predict(x_train_random)
        y_predict_test_random = clf_random.predict(x_test_random)

        random_train_acc_subjectWise = getSubjectwiseAccuracy(y_predict_train_random, y_train_random, train_random_subjIds)
        random_test_acc_subjectWise = getSubjectwiseAccuracy(y_predict_test_random, y_test_random, test_random_subjIds)

        print("Random - Train accuracy token wise : {}, train accuracy subject wise : {}, Test accuracy token wise : {}, test accuracy subject wise : {}".format(random_train_acc_tokenWise, random_train_acc_subjectWise, random_test_acc_tokenWise, random_test_acc_subjectWise))
        train_accuracies_random_subjectWise.append(random_train_acc_subjectWise)
        test_accuracies_random_subjectWise.append(random_test_acc_subjectWise)
        train_accuracies_random_tokenWise.append(random_train_acc_tokenWise)
        test_accuracies_random_tokenWise.append(random_test_acc_tokenWise)


        coefs = clf_static.coef_

        importances = generateImportanceFromCoefs(coefs)

        allImportances.append(importances)

        relevancyScores.append(np.mean(x_test_static_relevancyScore))

        # plot and save the importance values
        saveImportanceResults(importances, saveFolder_results)

    # find best rois average across all folds now
    if(len(allImportances) > 1):

        resultMessage = "\n STATIC RESULTS \n \n"
        for i in range(len(train_accuracies_static_subjectWise)):
            resultMessage += "STATIC SUBJECTWISE - Fold : {}, train acc : {}, test acc : {}\n".format(i, train_accuracies_static_subjectWise[i], test_accuracies_static_subjectWise[i])
        resultMessage += "\n STATIC SUBJECTWISE - Total, train acc : {} +- {}, test acc : {} +- {} \n \n".format(np.mean(train_accuracies_static_subjectWise), np.std(train_accuracies_static_subjectWise), np.mean(test_accuracies_static_subjectWise), np.std(test_accuracies_static_subjectWise))
        for i in range(len(train_accuracies_static_tokenWise)):
            resultMessage += "STATIC TOKENWISE - Fold : {}, train acc : {}, test acc : {}\n".format(i, train_accuracies_static_tokenWise[i], test_accuracies_static_tokenWise[i])
        resultMessage += "\n STATIC TOKENWISE - Total, train acc : {} +- {}, test acc : {} +- {} \n \n".format(np.mean(train_accuracies_static_tokenWise), np.std(train_accuracies_static_tokenWise), np.mean(test_accuracies_static_tokenWise), np.std(test_accuracies_static_tokenWise))



        resultMessage += "\n \n \n RANDOM RESULTS \n \n"
        for i in range(len(test_accuracies_random_subjectWise)):
            resultMessage += "RANDOM SUBJECTWISE - Fold : {}, train acc : {}, test acc :  {}\n".format(i, train_accuracies_random_subjectWise[i], test_accuracies_random_subjectWise[i])
        resultMessage += "\n RANDOM SUBJECTWISE - Total, train acc : {} +- {}, test acc : {} +- {} \n \n".format(np.mean(train_accuracies_random_subjectWise), np.std(train_accuracies_random_subjectWise), np.mean(test_accuracies_random_subjectWise), np.std(test_accuracies_random_subjectWise))
        for i in range(len(test_accuracies_random_tokenWise)):
            resultMessage += "RANDOM TOKENWISE - Fold : {}, train acc : {}, test acc :  {}\n".format(i, train_accuracies_random_tokenWise[i], test_accuracies_random_tokenWise[i])
        resultMessage += "\n RANDOM TOKENWISE - Total, train acc : {} +- {}, test acc : {} +- {}".format(np.mean(train_accuracies_random_tokenWise), np.std(train_accuracies_random_tokenWise), np.mean(test_accuracies_random_tokenWise), np.std(test_accuracies_random_tokenWise))
        
        
        saveFolder_results = "./Analysis/DataExtracted/{}/seed_{}/RESULTS/startK_{}".format(targetDataset, seed, startK) 

        # make sure to delete previous results
        shutil.rmtree(saveFolder_results, ignore_errors=True)

        os.makedirs(saveFolder_results, exist_ok=True)

        resultFile = open(saveFolder_results + "/results.txt", 'w')
        resultFile.write(resultMessage)
        resultFile.close()

        allImportances = np.array(allImportances).mean(axis=0)

        averageRelevancies = np.array(relevancyScores)

        saveImportanceResults(allImportances, saveFolder_results)


    return test_accuracies_static_subjectWise, averageRelevancies, allImportances

