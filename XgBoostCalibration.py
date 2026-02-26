from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
import os
import gc

import numpy as np
import pandas as pd
from svglib.svglib import svg2rlg  # svg graphics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, auc, RocCurveDisplay, make_scorer, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_validate, StratifiedKFold, GroupKFold, \
    cross_val_score, cross_val_predict, GridSearchCV
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
import pipes
import xgboost as xgb
from main import normalize, preProcess, readInFile, dataSplit, resultCSV, calcGCcontent, GCcontent, RocDisplay, analyzeTitrationGroups, phredQual, filterTableResults, saveModel, loadModel
import time
from sklearn.preprocessing import LabelEncoder
import logging

XGDataFeatures = ["ZScoreGCScore", "ZScoreGTScore", "Theta", "ZScoreClusterSep", "R", "X", "Y", "AlleleAB_x",
     "ZScoreXrawYraw", "45Theta", "90Theta", "absZScoreXY", "45SubArc", "SubArc", "90SubArc", "ZScoreXYMean", "ZScoreXYVariance", "Angle Error"]
labels = ["A A", "A B", "B B"]
testSamples = [7046]
exclusion  = []
downSample = 0
modelName= "CalibratedXGBoost"
dateStr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
omniPath =f"/(replace with your dir)/{dateStr}"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

if __name__ == "__main__":


    log_file = f"Run {dateStr} {modelName} details.txt"
    def customScore(yTrue, pred):
        global ys
        ys.append(pred)
        negLoss = -log_loss(yTrue, pred)
        return negLoss
    def calibrationSet(xTrainGroup,yTrainGroup,individuals):
        from sklearn.utils.class_weight import compute_sample_weight
        indDf=pd.DataFrame({"ExternalSampleID":individuals})

        xTrainGroup= pd.concat([xTrainGroup, yTrainGroup,indDf], axis =1 )
        #equal class 
        xCalTrainGroup = xTrainGroup.groupby("AlleleAB_y").sample(n=1000, replace =True)

        xTrainGroup.drop(xCalTrainGroup.index,inplace=True, axis=0)
        clsWeight = compute_sample_weight(class_weight="balanced",
                                          y=xTrainGroup["AlleleAB_y"])
        individuals= xTrainGroup.ExternalSampleID.tolist()
        yCalTrainGroup = xCalTrainGroup["AlleleAB_y"]
        yTrainGroup = xTrainGroup["AlleleAB_y"]

        xTrainGroup.drop("ExternalSampleID",inplace = True, axis=1)
        xCalTrainGroup.drop("ExternalSampleID", inplace=True, axis=1)
        xTrainGroup.drop("AlleleAB_y", inplace=True, axis=1)
        xCalTrainGroup.drop("AlleleAB_y", inplace=True, axis=1)

        return xTrainGroup, yTrainGroup, xCalTrainGroup, yCalTrainGroup, individuals , clsWeight

    startTime = time.time()
    print("Starting... in main", flush =True)
    manager = multiprocessing.Manager()
    ys = manager.list()
    XGBoostDf = readInFile(1)
    XGBoostDf = preProcess(XGBoostDf)
    XGBoostDf['AlleleAB_x'] = XGBoostDf['AlleleAB_x'].astype('category')
    XGBoostDf['AlleleAB_y'] = XGBoostDf['AlleleAB_y'].astype('category')


    xTrainGroup, xTestGroup, yTrainGroup, yTestGroup, individuals, testSNPs,  titrateGroup, clsWeights,splits  = dataSplit(XGBoostDf,downSample, XGDataFeatures,testSamples, exclusion,log_file)
    #calibration set
    xTrainGroup, yTrainGroup, xCalTrainGroup, yCalTrainGroup, individuals, clsWeights = calibrationSet(xTrainGroup,yTrainGroup,individuals)
    xgbCf = xgb.XGBClassifier(objective="multi:softprob", enable_categorical=True)
    downsample = 0
    gc.collect()
    le = LabelEncoder()
    yTestGroup =le.fit_transform(yTestGroup)
    yTrainGroup = le.fit_transform(yTrainGroup)
    yCalTrainGroup = le.fit_transform(yCalTrainGroup)
    gkf = GroupKFold(n_splits=7)

    cvScore = cross_val_score(xgbCf, xTrainGroup, yTrainGroup, groups=individuals, cv=GroupKFold(), error_score = 'raise')
    parameters = {
        #'n_estimators': [1],
        #'learning_rate':[0.001],
        #'max_depth':[1],
        #'gamma':[0],
        #'min_child_weight':[0.1],
        #'subsample':[1],
        'n_estimators': [1000],
        'learning_rate': [0.01],
        'max_depth': [100],
        'gamma': [1],
        'min_child_weight': [0.1],
        'subsample': [0.8]

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
    gS = GridSearchCV(xgbCf, parameters, scoring=scorers, refit= "CustomNegLog", cv=gkf, n_jobs= 255
                      )
    gS.fit(xTrainGroup,yTrainGroup, groups = individuals, sample_weight =clsWeights)
    gsDf = pd.DataFrame(gS.cv_results_)
    bestModel = gS.best_estimator_

    calClf = CalibratedClassifierCV(bestModel,cv = 'prefit', method="sigmoid")
    calClf.fit(xCalTrainGroup, yCalTrainGroup)

    print("Done writing results...", flush=True)

    for y, i in enumerate(testSamples):
      os.makedirs(f"{omniPath}/{i}/{modelName}", exist_ok=True)
      log_file = f"{omniPath}/{i}/{modelName}/Run {dateStr} {modelName} details.txt"
      with open(log_file, "w") as f:
        for key, value in parameters.items():
          f.write(f"{key}: {value}\n")
          f.write(f"downsampling? {downSample}\n")
      if y == 0:
        saveModel(calClf, f"{omniPath}/{i}/{modelName}/{modelName}.pkl")
        bestModel = loadModel(f"{omniPath}/{i}/{modelName}/{modelName}.pkl")
        gsDf.to_csv(f"{omniPath}/{i}/{modelName}/{modelName}GSresults.csv", index=False)

    #edit out after finish run 2/8/2026      
    #os.makedirs(f"{omniPath}/7046/{modelName}", exist_ok=True)
    #bestModel = loadModel(f"/CalibratedXGBoost/{modelName}F.pkl")
    #bestModel = gS.best_estimator_
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

    calPredProb = calClf.predict_proba(xTestGroup)
    yCalPred = calClf.predict(xTestGroup)
    yCalPred=le.inverse_transform(yCalPred)
    #RocDisplay(yTrainGroup, yTestGroup, predProb, "XGBoost")
    accuracies = np.append(accuracies, accuracies)
    endTime=time.time()
    print("XGBoost accuracy on Test set", accuracies)

    analyzeTitrationGroups(titrateGroup, testSamples, XGDataFeatures, bestModel, modelName, le = le)

    duration = endTime- startTime
    with open(log_file, "a") as f:
        f.write(f"duration {duration}\n")
    
    volume = "All"
    for individuals in testSamples:
      for InputNg in [1,0.5,0.1,0.05,0.01]:
        #resultCSV(f"{modelName}", predProb, yPred, testSNPs, yTestGroup, f"{omniPath}/{individuals}/{modelName}")
        df = pd.read_csv(f"{omniPath}/{individuals}/{modelName}/{modelName}PredictedFor{individuals}At{InputNg}Results.csv")
        df["Correct/Incorrect"] = df["Pred"] == df["Actual"]
        y_test = df["Correct/Incorrect"]
        prob_pos = df[["Prob AA", "Prob AB", "Prob BB"]].max(axis=1)
        true_pos, pred_pos = calibration_curve(y_test, prob_pos, n_bins=5)
        plt.plot(pred_pos,
                 true_pos,
                 marker='o',
                 linewidth=1,
                 label='XGBoost')
        plt.savefig(f"{omniPath}/{individuals}/{modelName}/{modelName} {InputNg} Calibration plot.png")


'''
xgbCf.fit(xTrainGroup,yTrainGroup)
yPred = xgbCf.predict(xTestGroup)
yPredProb = xgbCf.predict_proba(xTestGroup)
accuracies = accuracy_score(yTestGroup, yPred)
accuracies = np.append(accuracies, accuracies)
print("xgboost",accuracies)
'''



