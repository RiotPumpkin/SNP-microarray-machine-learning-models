import os
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import multiprocessing
import tensorflow as tf
import keras_tuner as kt
import keras.src.optimizers.optimizer
from keras.utils import to_categorical
from keras.datasets import boston_housing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import math

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import losses
from keras import constraints
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from keras import models
from keras import Sequential, layers
from keras.callbacks import EarlyStopping
from keras.constraints import MaxNorm
from keras.optimizers import Adam
from keras import backend as K

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import time
import main
from sklearn.model_selection import train_test_split, KFold, cross_validate, StratifiedKFold, GroupKFold, \
    cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, auc, RocCurveDisplay, make_scorer, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
start = time.time()

seedValue = 42

def buildModel(dropoutRate, weightConstraint,hiddenLayer, l1, l2, learnRate, neurons, neurons2, negative_slope):
#def buildModel(dropoutRate, weightConstraint,hiddenLayer, l1, l2, learnRate, neurons, neurons2):
    model = Sequential()
    model.add(keras.Input(shape=(20,)))
    '''
    model.add(Dropout(dropoutRate))
    model.add(Dense(neurons,
                    kernel_constraint=MaxNorm(weightConstraint),
                    kernel_regularizer=l1_l2(l1=l1, l2=l2),
                    bias_regularizer=l1_l2(l1=l1, l2=l2)))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=negative_slope))
    '''
    for i in range(hiddenLayer):
        model.add(Dropout(dropoutRate))
        model.add(layers.Dense(neurons2, activation='leaky_relu',
                               kernel_constraint=MaxNorm(weightConstraint),
                               kernel_regularizer=l1_l2(l1=l1, l2=l2),
                               bias_regularizer=l1_l2(l1=l1, l2=l2)))
        model.add(tf.keras.layers.LeakyReLU(negative_slope=negative_slope))

    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    return model



NeuralNetFeatures = ["ZScoreGCScore", "ZScoreGTScore", "ZScoreClusterSep", "R", "X", "Y", "AlleleAB_x_AA", "Theta",
                     "AlleleAB_x_BB", "AlleleAB_x_AB",
                     "ZScoreXrawYraw", "45Theta", "90Theta", "absZScoreXY", "45SubArc", "SubArc", "90SubArc",
                     "ZScoreXYMean", "ZScoreXYVariance", "Angle Error"]
labels = ["A A", "A B", "B B"]
modelName = "Neural Network"
testSamples = [7046]
if __name__ == "__main__":
    def customScore(yTrue, pred):
        global ys
        ys.append(pred)
        negLoss = -log_loss(yTrue, pred)
        return negLoss

    # load dataset
    dateStr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"Run {dateStr} {modelName} details.txt"

    manager = multiprocessing.Manager()
    ys = manager.list()
    print("Loading File...", flush=True)
    NeuralSNPData = main.readInFile(2)
    CustomNegLogLoss = make_scorer(customScore, needs_proba=True)
    scorers = {"Accuracy": "accuracy", "CustomNegLog": CustomNegLogLoss}
    es = EarlyStopping(monitor=CustomNegLogLoss, mode = 'max', verbose = 1)
    end = time.time()
    print("Done loading", end - start, flush=True)
    print("Starting Preprocessing", flush=True)
    NeuralSNPData = main.preProcess(NeuralSNPData)

    print("Done...", end - start, flush=True)
    end = time.time()
    print("DataSplit and label encoding...", end - start, flush=True)
    xTrainGroup, xTestGroup, yTrainGroup, yTestGroup, individuals, testSNPs, titrateGroup, clsWeight = main.dataSplit(
        NeuralSNPData, 1,
        NeuralNetFeatures, log_file)
    yTestGroupRoc = yTestGroup
    yTrainGroupRoc = yTrainGroup
    le = LabelEncoder()
    yTestGroup = le.fit_transform(yTestGroup)
    yTrainGroup = le.fit_transform(yTrainGroup)
    print("Done dataSplit and label encoding...", end - start, flush=True)

    end = time.time()
    print("creating model...", end - start, flush=True)

    model = KerasClassifier(model=buildModel)

    epochs = [50]
    batches = [50]
    dropout = [0, 0.2]
    weightConstraint = [3]
    learnRate = [0.001, 0.01]
    neurons2 = [10, 20]
    neurons = [1, 5, 10, 20]
    hiddenLayer = [0, 1]
    l1 = [0.01]
    l2 = [0.01]
    negative_slope = [0.1]
    paramGrid = dict(epochs=epochs,
                     batch_size=batches,
                     optimizer__learnRate=learnRate,
                     model__dropoutRate=dropout,
                     model__weightConstraint=weightConstraint,
                     model__neurons = neurons,
                     model__neurons2 = neurons2,
                     model__learnRate=learnRate,
                     model__hiddenLayer=hiddenLayer,
                     model__l1=l1,
                     model__l2=l2,
                     model__negative_slope=negative_slope
                     )
    with open(log_file, "a") as f:
        for key, value in paramGrid.items():
            f.write(f"{key}: {value}\n")
    #main.resultCSV("NeuralNet",predProb,yPred,testSNPs,yTestGroup)
    gkf = GroupKFold(n_splits=6)
    end = time.time()
    print("GS initiation...", end - start, flush=True)
    gS = GridSearchCV(estimator=model, param_grid=paramGrid, scoring=scorers, refit="CustomNegLog", cv=gkf, n_jobs=-1,
                      verbose=2, pre_dispatch='2*n_jobs')
    print("finished GS...", end - start, flush=True)
    print("Searching Gs...", end - start, flush=True)
    gSResult = gS.fit(xTrainGroup, yTrainGroup, groups=individuals, callbacks = [es])
    end = time.time()

    print("Best Scores GS", gSResult.best_score_, flush=True)
    gsDf = pd.DataFrame(gS.cv_results_)
    gsDf.to_csv(f"{modelName}GSresults.csv",
                index=False)
    print("Predicting...", end - start, flush=True)
    main.saveModel(gS, "NNSNPsIllumina.pkl")
    #for individuals in testSamples:
        #main.saveModel(gS, f"(your directory)/{individuals}/{modelName}/{modelName}.pkl")
        #gsDf.to_csv(f"(your directory)/{individuals}/{modelName}/{modelName}GSresults.csv", index=False)
    bestModel = gS.best_estimator_
    cvResult = cross_val_score(bestModel, xTrainGroup, yTrainGroup, scoring="neg_log_loss", cv=gkf, groups=individuals)
    print("results", cvResult, flush=True)

    yPred = bestModel.predict(xTestGroup)
    end = time.time()
    print("Inversing...", end - start, flush=True)
    yTestGroup = le.inverse_transform(yTestGroup)
    yPred = le.inverse_transform(yPred)
    end = time.time()
    print("Finished...", end - start, flush=True)
    print(yTestGroup, yPred)
    end = time.time()
    print("Calculating accuracy...", end - start, flush=True)
    accuracies = accuracy_score(yTestGroup, yPred)
    print(accuracies, flush=True)
    predProb = bestModel.predict_proba(xTestGroup)
    #main.RocDisplay(yTrainGroupRoc, yTestGroupRoc, predProb, "Neural Network Test Group")
    accuracies = np.append(accuracies, accuracies)
    print("Finished predicting", end - start, flush=True)
    print(accuracies)
    #NetCm = confusion_matrix(yTestGroup, yPred, labels=labels)
    #ConfusionMatrixDisplay(confusion_matrix=NetCm, display_labels=labels).plot()
    main.analyzeTitrationGroups(titrateGroup, testSamples, NeuralNetFeatures, bestModel, modelName, le=le)
    end = time.time()
    duration = end- start
    with open(log_file, "w") as f:
        f.write(f"duration {duration}\n")
    #main.resultCSV(f"{modelName}",yPredProb,yPred,testSNPs)
    print("Finished", duration, flush=True)
