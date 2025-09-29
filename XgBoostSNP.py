import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
import multiprocessing
import os
import gc
import math
import numpy as np
import pandas as pd
import seaborn as sns
from svglib.svglib import svg2rlg  # svg graphics

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, auc, RocCurveDisplay, make_scorer, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_validate, StratifiedKFold, GroupKFold, \
    cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer

import xgboost as xgb
from main import normalize, preProcess, readInFile, dataSplit, resultCSV, calcGCcontent, GCcontent, RocDisplay, analyzeTitrationGroups, phredQual, filterTableResults, saveModel, loadModel
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import logging

XGDataFeatures = ["ZScoreGCScore", "ZScoreGTScore", "Theta", "ZScoreClusterSep", "R", "X", "Y", "AlleleAB_x",
     "ZScoreXrawYraw", "45Theta", "90Theta", "absZScoreXY", "45SubArc", "SubArc", "90SubArc", "ZScoreXYMean", "ZScoreXYVariance", "Angle Error"]
labels = ["A A", "A B", "B B"]
testSamples = [7046]
modelName= "XGBoost"
if __name__ == "__main__":

    dateStr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Run {dateStr} {modelName} details.txt"
    def customScore(yTrue, pred):
        global ys
        ys.append(pred)
        negLoss = -log_loss(yTrue, pred)
        return negLoss
    startTime = time.time()
    print("Starting... in main", flush =True)
    manager = multiprocessing.Manager()
    ys = manager.list()
    XGBoostDf = readInFile(2)
    XGBoostDf = preProcess(XGBoostDf)
    XGBoostDf['AlleleAB_x'] = XGBoostDf['AlleleAB_x'].astype('category')
    XGBoostDf['AlleleAB_y'] = XGBoostDf['AlleleAB_y'].astype('category')


    xTrainGroup, xTestGroup, yTrainGroup, yTestGroup, individuals, testSNPs,  titrateGroup, clsWeights = dataSplit(XGBoostDf,0, XGDataFeatures, log_file)
    xgbCf = xgb.XGBClassifier(objective="multi:softprob", enable_categorical=True)
    print(XGDataFeatures)
    gc.collect()
    le = LabelEncoder()
    yTestGroup =le.fit_transform(yTestGroup)
    yTrainGroup = le.fit_transform(yTrainGroup)
    gkf = GroupKFold(n_splits=7)
    cvScore = cross_val_score(xgbCf, xTrainGroup, yTrainGroup, groups=individuals, cv=GroupKFold(), error_score = 'raise')
    parameters = {
        'n_estimators': [1],
        'learning_rate':[0.001],
        'max_depth':[1],
        'gamma':[0],
        'min_child_weight':[0.1],
        'subsample':[1],
    }
    with open(log_file, "w") as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

        #'n_estimators': [1000, 1500],
        #'learning_rate': [0.01, 0.1],
        #'max_depth': [10, 100],
        #'gamma': [0, 1, 5],
        #'min_child_weight': [0, 0.1],
        #'subsample': [0.5, 0.8, 0.9]

    CustomNegLogLoss = make_scorer(customScore, needs_proba=True)
    scorers = {"Accuracy": "accuracy", "CustomNegLog": CustomNegLogLoss, "Roc": "roc_auc_ovo"}
    print("Start gridsearch initialization...", flush=True)
    gS = GridSearchCV(xgbCf, parameters, scoring=scorers, refit= "CustomNegLog", cv=gkf, n_jobs= 7
                      )
    gS.fit(xTrainGroup,yTrainGroup, groups = individuals, sample_weight =clsWeights)
    gsDf = pd.DataFrame(gS.cv_results_)
    print("Done writing results...", flush=True)
    gsDf.to_csv("XGBoostGsResults.CSV", index = False)

    #for individuals in testSamples:
        #os.makedirs(f"(your directory)/{individuals}/{modelName}", exist_ok=True)
        #saveModel(gS, f"(your directory)/{individuals}/{modelName}/{modelName}.pkl")
        # gS = loadModel(f"(your directory)/{individuals}/{modelName}/{modelName}.pkl")
        #gsDf.to_csv(f"/(your directory)/{individuals}/{modelName}/{modelName}GSresults.csv", index=False)
    bestModel = gS.best_estimator_
    #cvResult = cross_val_score(bestModel,xTrainGroup,yTrainGroup,scoring = "neg_log_loss", cv = gkf, groups= individuals)
    #print("predicting best model", flush=True)
    #print("results of CV", cvResult, flush=True)

    yPred = bestModel.predict(xTestGroup)
    yTestGroup= le.inverse_transform(yTestGroup)
    yTrainGroup = le.inverse_transform(yTrainGroup)
    yPred = le.inverse_transform(yPred)
    gsPredProb = list(ys)
    print(gsPredProb)
    print(yTestGroup, yPred)
    accuracies = accuracy_score(yTestGroup, yPred)
    predProb = bestModel.predict_proba(xTestGroup)
    #RocDisplay(yTrainGroup, yTestGroup, predProb, "XGBoost")
    accuracies = np.append(accuracies, accuracies)
    endTime=time.time()
    print("XGBoost accuracy on Test set", accuracies)

    analyzeTitrationGroups(titrateGroup, testSamples, XGDataFeatures, bestModel, modelName, le = le)
    # resultCSV(f"{modelName}",yPredProb,yPred,testSNPs)
    duration = endTime- startTime
    with open(log_file, "a") as f:
        f.write(f"duration {duration}\n")
    plt.show()




'''
xgbCf.fit(xTrainGroup,yTrainGroup)
yPred = xgbCf.predict(xTestGroup)
yPredProb = xgbCf.predict_proba(xTestGroup)
accuracies = accuracy_score(yTestGroup, yPred)
accuracies = np.append(accuracies, accuracies)
print("xgboost",accuracies)
'''
