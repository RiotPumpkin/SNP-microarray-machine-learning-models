import matplotlib.pyplot as plt
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
import os
import gc
import math
import numpy as np
import pandas as pd
from sklearn.utils import resample, class_weight
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, auc, RocCurveDisplay, log_loss, \
    make_scorer, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_validate, StratifiedKFold, GroupKFold, \
    cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer

import time
import sys


import statsmodels.api as sm
import logging
# test GGPLOT
from plotnine.data import huron

# scatter plot matrix SPLOMB
# marginal distributions
# summary statistics visualization
# z-score plot actual value x11 forwarding, winscp

pi = math.pi
accuracies = np.array([])
cmPlots = {}
cmMatrix = []
plotTitle = ["ng1Table", "ng0_5Table", "ng0_1Table", "ng0_05Table", "ng0_01Table"]
# classifierTitle = ["DummyClassifier", "Random Forest", "Logistic Regression", "Adaboost", "NeuralNet"]

samplesID = [7013, 7035, 7046, 7437, 13047, 7028, 13053]
s = np.array(["Sample7013", "Sample7035", "Sample7046", "Sample7437", "Sample13047", "Sample13050", "Sample7028",
              "Sample13053"])
normalizeFeatures = ["GCScore", "GTScore", "ClusterSep", "XrawYraw", "XYMean", "XYVariance"]
testSamples = [7046]
# trainSamples = [7035, 7437, 13047, 7028, 13050, 7013]
ng1Table = pd.DataFrame()
ng0_5Table = pd.DataFrame()
ng0_1Table = pd.DataFrame()
ng0_05Table = pd.DataFrame()
ng0_01Table = pd.DataFrame()
volumeTables = [ng1Table, ng0_5Table, ng0_1Table, ng0_05Table, ng0_01Table]


# scikit opt hyperparameters definitions logistics regression
# LR hyperparameters solver, penalty, max_iter, C (regularization strength), tol, fit_intercept,
# intercept_scaling, class_weight, random_state, multi_class, verbose, warm_start, and l1_ratio

# standard error
def printError(*args, **kwargs):
    sys.stderr = open('err.txt', 'w')
    print(*args, file=sys.stderr, **kwargs)


# null classifier
def nullClf(test, answer):
    null = DummyClassifier(strategy="most_frequent")
    null.fit(testGroup, yTestGroup)
    pred = null.predict(test)
    accuracies = accuracy_score(yTestGroup, pred)
    nullProbability = null.predict_proba(test)
    print("Dummy accuracy", accuracies)


# creates tables based on the input amount of DNA
# data preprocessingf
def filterTableByVolumeJoin(volumeTable, truthTable, NgVol):
    # volume = volume[["AlleleAB", "Chr", "Position", "SNPName", "DNAinputNG"]]
    joinedTable = pd.merge(volume, truthTable, on=["Chr", "Position", "SNPName"], how="left")
    # drop alleles AB with -- in joinedDt
    # joinedTable = joinedTable[joinedTable.AlleleAB_x != "- -"]
    return joinedTable


# creates plots of confusion matrix using facets
'''
# AA BB AB  0 1 2
def displayCMPlot(dfVolume, option):
    labels = ["A A", "A B", "B B"]
    samples = [7013, 7035, 7046, 7437, 13047, 13050, 7028, 13053]
    inputNg = [1, 0.5, 0.1, 0.05, 0.01]
    #cm = confusion_matrix(dfVolume["AlleleAB_x"], dfVolume["AlleleAB_y"], labels=labels)
    #disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    for i in samples:
        Group = dfVolume[dfVolume["ExternalSampleID"] == i]
        print("titrateGroup1", Group["ExternalSampleID"])
        for y in inputNg:
            inputNgGroup = Group[Group["DNAinputNG"] == y]
            xTitrateGroup = inputNgGroup["AlleleAB_x"]
            yTitrateGroup = inputNgGroup["AlleleAB_y"]
            accuracies = accuracy_score(yTitrateGroup, xTitrateGroup)
            print(accuracies)
            cm = confusion_matrix(yTitrateGroup, xTitrateGroup, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
            disp.ax_.set_title(f"Sample ID {i}, input Ng {y}")


    return disp
    # disp.subplot(1, 5, index)
    # plt.title(plotTitle[index])
'''


def displayCMPlot(inputNgGroup, volume, individuals,concordanceFilter, filter=0, model="GenomeStudio", prediction="Predicted"):
    labels = ["A A", "A B", "B B"]
    # cm = confusion_matrix(dfVolume["AlleleAB_x"], dfVolume["AlleleAB_y"], labels=labels)
    # disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    labelX = inputNgGroup[prediction]
    labelY = inputNgGroup["AlleleAB_y"]
    accuracies = accuracy_score(labelY, labelX)
    print(accuracies)
    cm = confusion_matrix(labelY, labelX, labels=labels)
    dfCm = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix in dataframe", dfCm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
    disp.ax_.set_title(f"{volume} Ng, {model}, {concordanceFilter},Phred filter: {filter}, Sample ID {individuals}")
    os.makedirs(f"(your directory)\\{individuals}\\{model}\\{volume}", exist_ok = True)
    #os.makedirs(f"/home/ac0539/snp_chip_data/Omni/{individual}/{model}/{volume}/", exist_ok=True)
    plt.savefig(
        f'(your directory)\\{individuals}\\{model}\\{volume}\\{volume}Ng ConfusionMatrix {model} Phred {filter} {concordanceFilter} No Nulls.png')


    return disp


# normalize values with a zscore
def normalize(df, columnName, saveScaler):
    if saveScaler == 1:
        scaler = StandardScaler()
    else:
        scaler = joblib.load('ZScoreScaler.bin')
    if columnName in ("XYMean", "XYVariance"):
        # Apply normalization within each ExternalSampleID group
        df[f"ZScore{columnName}"] = df.groupby("ExternalSampleID")[columnName].transform(
            lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        )
    else:
        # Apply normalization within each ExternalSampleID and DNAinputNG group
        df[f"ZScore{columnName}"] = df.groupby(["ExternalSampleID", "DNAinputNG"])[columnName].transform(
            lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        )
        # creates bin file saving parameters of normalization

    if saveScaler == 1:
        joblib.dump(scaler, 'ZScoreScaler.bin', compress=True)

    return df


def readInFile(option):

    #unix os
    if os.path.isfile("(your directory)/preProcessedDf.pkl"):
        print("Found pickled file", flush=True)
        dF = pd.read_pickle("(your directory)/preProcessedDf.pkl")

        return dF
    else:
        print("Reading", flush=True)
        currentPath = (os.path.dirname(__file__))
        path = os.path.join(currentPath, "../(your directory)/(your file containing SNPs).csv")
        dF = pd.read_csv(path, sep="\t")
        end = time.time()
        print("done", start - end, flush=True)
        dF.isnull().values.any()
        #dF.to_pickle("//preProcessedDf.pkl")
        return dF
    elif option == 2:

        # reading in the SNP files and combining into one large dataframe called joinedDt
        dF = pd.read_csv('(your directory where csv si stored)',
                                 sep="\t")

        #dF.to_csv("ToyDataset.csv", index = False)
        #sys.exit("Break the code")

        return dF


def preProcess(dF):

    if os.path.isfile("(your directory)/preProcessedDf.pkl"):
        print("Pickled file already exists", flush=True)
        dF = pd.read_pickle("(your directory)/preProcessedDf.pkl")

        return dF
    else:
        # Matching source strand with chr position index name etc...
        dF.drop(dF[dF["AlleleAB"] == "- -"].index, inplace=True)
        dF["XrawYraw"] = dF.XRaw + dF.YRaw
        dF = dF[dF["Chr"] != "X"]
        dF = dF[dF["Chr"] != "XY"]
        print("Chr", dF["Chr"].unique(), flush=True)
        # joinedDt["ZScoreXY"] = joinedDt.groupby("DNAinputNG")["Xraw+Yraw"].transform(z_score)
        # adding variance and intensity
        # calculate theta distance polar coordinates?
        dF.loc[dF['AlleleAB'] == 'A A', 'Angle Error'] = dF.R * dF['Theta']
        dF.loc[dF['AlleleAB'] == 'B B', 'Angle Error'] = dF.R * (pi / 2 - dF["Theta"])
        dF.loc[dF['AlleleAB'] == 'A B', 'Angle Error'] = dF.R * (pi / 4 - dF["Theta"])
        dF["45Theta"] = (pi / 2 - dF["Theta"])
        dF["90Theta"] = (pi / 4 - dF["Theta"])
        dF["SubArc"] = dF.R * (dF["Theta"])
        dF["45SubArc"] = dF.R * (pi / 2 - dF["Theta"])
        dF["90SubArc"] = dF.R * (pi / 4 - dF["Theta"])

        # mean signal intensities

        dF["XYMean"] = dF.groupby(["ExternalSampleID", "DNAinputNG"])['XrawYraw'].transform("mean")
        dF["XYVariance"] = dF.groupby(["ExternalSampleID", "DNAinputNG"])['XrawYraw'].transform("var")

        print("Var", dF["XYVariance"], "Mean", dF["XYMean"], flush=True)
        print("dF including XYmean, Var", dF, flush=True)

        truthTable = dF[(dF["DNAinputNG"] == 50)]

        truthTable = truthTable[
            ['AlleleAB', 'Chr', "Position", "SNPName", "Relationship", "Family", "ExternalSampleID"]]
        dF.drop(dF[(dF["DNAinputNG"] == 50)].index, inplace=True)

        dF = pd.merge(dF, truthTable, on=["Chr", "Position", "SNPName", "Relationship", "Family", "ExternalSampleID"],
                      how="left", indicator= True)
        # dF.to_csv("MERGEDDF.csv", index = False)

        print("Columns of merged Dataframe,", dF, flush=True)

        '''
        for i, Ng in enumerate(inputNg):
            volumeTables[i] = dF[(dF["DNAinputNG"] == Ng)]
            truthTable = truthTable[['AlleleAB', 'Chr', "Position", "SNPName"]]
            volumeTables[i] = volumeTables[i].reset_index(drop=True)
            volumeTables[i] = pd.merge(volumeTables[i], truthTable, on=["Chr", "Position", "SNPName"], how="left")
            volumeTables[i] = volumeTables[i].dropna(subset=['AlleleAB_y'])
        '''

        # processedDf = pd.concat(volumeTables, axis=0)
        # processedDf.reset_index(drop=True, inplace=True)

        dF.drop(dF[dF["AlleleAB_y"] == "- -"].index, inplace=True)
        dF.drop(dF[dF["AlleleAB_x"] == "- -"].index, inplace=True)
        dF = dF[~dF.isin(["- -"]).any(axis=1)]
        dF = dF.dropna(subset=['AlleleAB_y'])
        dF.reset_index(drop=True, inplace=True)
        print("Preprocessing Df info", dF.info(), flush=True)
        print("Preprocessing Df mem usage", dF.memory_usage(), flush=True)
        # processedDf.dropna(inplace=True)
        # processedDf.drop(processedDf[processedDf["AlleleAB_y"] == "- -"].index, inplace=True)
        # processedDf.drop(processedDf[processedDf["AlleleAB_x"] == "- -"].index, inplace=True)
        dF["Correct/Incorrect"] = dF["AlleleAB_x"] == dF["AlleleAB_y"]
        start = time.time()
        print("Starting One hot encoding", flush=True)
        dF['AlleleAB_x'] = dF['AlleleAB_x'].astype('category')
        dF['AlleleNum'] = dF['AlleleAB_x'].cat.codes
        enc = OneHotEncoder()

        encData = pd.DataFrame(enc.fit_transform(dF[['AlleleAB_x']]).toarray())
        encData.columns = enc.get_feature_names_out(['AlleleAB_x'])
        columns = encData.columns
        newColumn = [name.replace(' ', '') for name in columns]
        encData.columns = newColumn
        encData.reset_index(drop=True, inplace=True)
        dF.reset_index(drop=True, inplace=True)
        dF = pd.concat([dF, encData], axis=1)
        end = time.time()
        # dF.to_pickle("(your directory)/preProcessedDf.pkl")
        for i in normalizeFeatures:
            normalize(dF, i, 1)
        dF["absZScoreXY"] = dF.ZScoreXrawYraw.abs()

        #df2 = pd.read_csv('..\\InfiniumOmni5-4v1-2_A1.csv',
        #                 sep=",", skiprows=7)
        #print(df2.info())
        #dF = pd.merge(dF, df2[['Name', 'Chr', 'AlleleA_ProbeSeq']],
        #              left_on=['SNPName', 'Chr'],
        #              right_on=['Name', 'Chr'],
        #              how='left')

        #dF = calcGCcontent(dF)
        # DF3=dF[dF["GCcontent"].isna()]
        # DF3.to_csv("NULLVALUES.csv")

        return dF


def graphInCorGc(df):
    # graphing distributions of incorrect/correct of GCScores
    plt.figure()
    correct = df[df["Correct/Incorrect"] == True]
    incorrect = df[df["Correct/Incorrect"] == False]
    # correct.groupby("DNAinputNG")["GCScore"].plot(kind = "hist", alpha=0.5, legend=True)
    incorrect.groupby("DNAinputNG")["GCScore"].plot(kind="hist", alpha=0.5, legend=True)

    # plt.hist([correct["GCScore"], incorrect["GCScore"]], alpha=0.5, label=["correct", "incorrect"])
    plt.legend(loc='upper right')
    plt.show()


# data partitioning into train and test data
# groupKfold
def dataSplit(data, downsample, wantedFeatures, log_file):
    from sklearn.utils.class_weight import  compute_sample_weight
    # individuals for learning
    groups = data[~data["ExternalSampleID"].isin(testSamples)]
    test = data[data["ExternalSampleID"].isin(testSamples)]
    # graphing inc/cor distribution of GC scores across different titration
    graphInCorGc(test)

    groups.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    # individuals for test
    with open(log_file, "a") as f:
        f.write(f"downsampling? {downsample}\n")

    if downsample == 1:
        clsWeight =  compute_sample_weight(class_weight= "balanced", y = groups["AlleleAB_y"] )
        correctDf = groups[groups["Correct/Incorrect"] == True]
        incorrectDf = groups[groups["Correct/Incorrect"] == False]

        print("correct Len", len(correctDf), "incorrect len", len(incorrectDf))
        #incorrectDf = resample(groups, random_state = 42, n_samples = 15000000, replace=True)
        #downsampleCorrectDf = resample(correctDf, random_state = 42, n_samples=len(incorrectDf))
        #add in sampling by individuals too
        incorrectDf = incorrectDf.groupby(["DNAinputNG","AlleleAB_y"]).sample(n=50, replace =True)
        downsampleCorrectDf = correctDf.groupby(["DNAinputNG","AlleleAB_y"]).sample(n=50, replace = True)
        print("Len of Df", len(incorrectDf), len(downsampleCorrectDf), flush=True)
        print("Rndpulled from incorrectDf", incorrectDf.sample(n=20), flush=True)
        incorrect20 = incorrectDf.sample(n=200)
        #incorrect20.to_csv("pulled.csv")
        groups = pd.concat([incorrectDf, downsampleCorrectDf])
        clsWeight= None
    # yTestGroup and testGroup are 3 individuals
    else:
        clsWeight =  compute_sample_weight(class_weight="balanced",
                                         y=groups["AlleleAB_y"])

    groups.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    yTrain = groups["AlleleAB_y"]
    yTest = test["AlleleAB_y"]
    # gcScore= test[test["AlleleAB_y"] =="A B"]

    gcScore = test["GCScore"]
    gcPred = test["AlleleAB_x"]
    gcTest = test["AlleleAB_y"]
    SNPinfo = test[["SNPName", "Chr", "Position", "SNPIndex"]]
    xTest = test[wantedFeatures]
    xTrain = groups[wantedFeatures]
    titrateGroup = test
    individuals = groups.ExternalSampleID.tolist()

    return xTrain, xTest, yTrain, yTest, individuals, SNPinfo, titrateGroup, clsWeight


def resultCSV(algoName, predictedProbability, pred, testDf, volume = "All"):
    results = pd.DataFrame(index=None,
                           columns=["Algorithm", "SNP name", "Chr", "SNPIndex", "Prob AA", "Prob AB", "Prob BB", "Pred",
                                    "Actual"])
    results.Pred = pred
    results.Actual = yTestGroup
    results.Algorithm = algoName
    results[["Prob AA", "Prob AB", "Prob BB"]] = predictedProbability
    results[["SNP name", "Chr", "Position", "SNPIndex"]] = testDf[["SNPName", "Chr", "Position", "SNPIndex"]]
    results.to_csv(f"{algoName} {volume}PredictedResults.csv", index=False)
    #results.to_csv(f".csv", index=False)


# originalCm = confusion_matrix(testingDf["AlleleAB_y"], testingDf["AlleleAB_x"], labels=labels)
# ConfusionMatrixDisplay(confusion_matrix=originalCm, display_labels=labels).plot()
def GCcontent(seq):
    if isinstance(seq, str):
        seq = seq.upper()
        g = seq.count("G")
        c = seq.count("C")
        GC = g + c
        GCContent = (GC / len(seq)) * 100
        return GCContent
    else:
        return None


def calcGCcontent(df):
    df["GCcontent"] = df["AlleleA_ProbeSeq"].apply(GCcontent)
    return df


from sklearn.preprocessing import LabelBinarizer


def RocDisplay(y_train, y_test, y_score, modelUsed, threshold=0, volume=0):
    import matplotlib.pyplot as plt
    from scipy import interpolate
    from sklearn.metrics import RocCurveDisplay

    n_classes = 3

    if not all(isinstance(i, np.ndarray) for i in [y_train, y_test]):
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    if not all(isinstance(i, np.ndarray) for i in [y_score]):
        y_score = y_score.to_numpy()

    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    ###

    y_onehot_test.shape  # (n_samples, n_classes)

    target_names = ["A A", "A B", "B B"]
    class_of_interest = "A B"
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
    print("class ID", class_id)
    # %%
    display = RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="darkorange",
        plot_chance_level=True,
    )

    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="One-vs-Rest ROC curves:\n AA vs (AB & BB)",
    )

    # %%
    # %%
    # ROC curve using micro-averaged OvR
    # ----------------------------------
    #
    # Micro-averaging aggregates the contributions from all the classes (using
    # :func:`numpy.ravel`) to compute the average metrics as follows:
    #
    # :math:`TPR=\frac{\sum_{c}TP_c}{\sum_{c}(TP_c + FN_c)}` ;
    #
    # :math:`FPR=\frac{\sum_{c}FP_c}{\sum_{c}(FP_c + TN_c)}` .
    #
    # We can briefly demo the effect of :func:`numpy.ravel`:

    print(f"y_score:\n{y_score[0:2, :]}")
    print()
    print(f"y_score.ravel():\n{y_score[0:2, :].ravel()}")

    # %%
    # In a multi-class classification setup with highly imbalanced classes,
    # micro-averaging is preferable over macro-averaging. In such cases, one can
    # alternatively use a weighted macro-averaging, not demoed here.

    display = RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_score.ravel(),
        name="micro-average OvR",
        color="darkorange",
        plot_chance_level=True,
    )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Micro-averaged One-vs-Rest\nReceiver Operating Characteristic",
    )

    # %%
    # In the case where the main interest is not the plot but the ROC-AUC score
    # itself, we can reproduce the value shown in the plot using
    # :class:`~sklearn.metrics.roc_auc_score`.

    from sklearn.metrics import roc_auc_score

    micro_roc_auc_ovr = roc_auc_score(
        y_test,
        y_score,
        multi_class="ovr",
        average="micro",
    )

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

    # %%
    # This is equivalent to computing the ROC curve with
    # :class:`~sklearn.metrics.roc_curve` and then the area under the curve with
    # :class:`~sklearn.metrics.auc` for the raveled true and predicted classes.

    from sklearn.metrics import auc, roc_curve

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc, thresholds = dict(), dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

    # %%
    # .. note:: By default, the computation of the ROC curve adds a single point at
    #     the maximal false positive rate by using linear interpolation and the
    #     McClish correction [:doi:`Analyzing a portion of the ROC curve Med Decis
    #     Making. 1989 Jul-Sep; 9(3):190-5.<10.1177/0272989x8900900307>`].
    #
    # ROC curve using the OvR macro-average
    # -------------------------------------
    #
    # Obtaining the macro-average requires computing the metric independently for
    # each class and then taking the average over them, hence treating all classes
    # equally a priori. We first aggregate the true/false positive rates per class:

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_onehot_test[:, i], y_score[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all - curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    class_of_interest = "A B"

    # %%
    # Plot all OvR ROC curves together
    # --------------------------------

    from itertools import cycle

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    for y in range(0, 3):
        f = fpr[y]
        t = tpr[y]
        th = thresholds[y]
        thresholdsLength = len(th)
        colorMap = plt.get_cmap('jet', thresholdsLength)
        '''
        texts = [ax.text(f[i] - 0.03, t[i] + 0.005, str(th[i])[:5], fontdict={'size': 10},
                    color=colorMap(i*y / thresholdsLength)) for i in range(0, thresholdsLength, 5000)];
        adjust_text(texts, expand=(1.2, 2), # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
            arrowprops=dict(arrowstyle='->', color='red') # ensure the labeling is clear by adding arrows
            );
        '''
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{modelUsed} Phred {threshold}, {volume} Ng\n Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass",
    )

    import pickle

    tprs = np.zeros(shape=(3, 3))
    for i in range(n_classes):
        tprintrp = interpolate.interp1d(thresholds[i], tpr[i])
        fprintrp = interpolate.interp1d(thresholds[i], fpr[i])

        print("TPR for thresholds,", i, tprintrp([0.999, 0.99, 0.9]), type(tprintrp))

        tprs[i] = tprintrp([0.999, 0.99, 0.9])

    tprDf = pd.DataFrame(tprs, columns=[0.999, 0.99, 0.9])
    tprDf.to_csv("tprs.csv")
    # %%
    # One-vs-One multiclass ROC
    # =========================
    #
    # The One-vs-One (OvO) multiclass strategy consists in fitting one classifier
    # per class pair. Since it requires to train `n_classes` * (`n_classes` - 1) / 2
    # classifiers, this method is usually slower than One-vs-Rest due to its
    # O(`n_classes` ^2) complexity.
    #
    # In this section, we demonstrate the macro-averaged AUC using the OvO scheme
    # for the 3 possible combinations in the :ref:`iris_dataset`: "setosa" vs
    # "versicolor", "versicolor" vs "virginica" and  "virginica" vs "setosa". Notice
    # that micro-averaging is not defined for the OvO scheme.
    #
    # ROC curve using the OvO macro-average
    # -------------------------------------
    #
    # In the OvO scheme, the first step is to identify all possible unique
    # combinations of pairs. The computation of scores is done by treating one of
    # the elements in a given pair as the positive class and the other element as
    # the negative class, then re-computing the score by inversing the roles and
    # taking the mean of both scores.

    from itertools import combinations

    pair_list = list(combinations(np.unique(y_test), 2))

    # %%
    pair_scores = []
    mean_tpr = dict()

    for ix, (label_a, label_b) in enumerate(pair_list):
        a_mask = y_test == label_a
        b_mask = y_test == label_b
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
        idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]

        fpr_a, tpr_a, _ = roc_curve(a_true, y_score[ab_mask, idx_a])

        fpr_b, tpr_b, _ = roc_curve(b_true, y_score[ab_mask, idx_b])

        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_score = auc(fpr_grid, mean_tpr[ix])
        pair_scores.append(mean_score)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.plot(
            fpr_grid,
            mean_tpr[ix],
            label=f"Mean {label_a} vs {label_b} (AUC = {mean_score :.2f})",
            linestyle=":",
            linewidth=4,
        )
        RocCurveDisplay.from_predictions(
            a_true,
            y_score[ab_mask, idx_a],
            ax=ax,
            name=f"{label_a} as positive class",
        )
        RocCurveDisplay.from_predictions(
            b_true,
            y_score[ab_mask, idx_b],
            ax=ax,
            name=f"{label_b} as positive class",
            plot_chance_level=True,
        )
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"{target_names[idx_a]} vs {label_b} ROC curves",
        )

    print(f"Macro-averaged One-vs-One ROC AUC score:\n{np.average(pair_scores):.2f}")

    # %%
    # One can also assert that the macro-average we computed "by hand" is equivalent
    # to the implemented `average="macro"` option of the
    # :class:`~sklearn.metrics.roc_auc_score` function.

    macro_roc_auc_ovo = roc_auc_score(
        y_test,
        y_score,
        multi_class="ovo",
        average="macro",
    )

    print(f"Macro-averaged One-vs-One ROC AUC score:\n{macro_roc_auc_ovo:.2f}")

    # %%
    # Plot all OvO ROC curves together
    # --------------------------------

    ovo_tpr = np.zeros_like(fpr_grid)

    fig, ax = plt.subplots(figsize=(6, 6))
    for ix, (label_a, label_b) in enumerate(pair_list):
        ovo_tpr += mean_tpr[ix]
        ax.plot(
            fpr_grid,
            mean_tpr[ix],
            label=f"Mean {label_a} vs {label_b} (AUC = {pair_scores[ix]:.2f})",
        )

    ovo_tpr /= sum(1 for pair in enumerate(pair_list))

    ax.plot(
        fpr_grid,
        ovo_tpr,
        label=f"One-vs-One macro-average (AUC = {macro_roc_auc_ovo:.2f})",
        linestyle=":",
        linewidth=4,
    )
    ax.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Extension of Receiver Operating Characteristic\nto One-vs-One multiclass",
        aspect="equal",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
    )
    #for y in testSamples:
        #for i in plt.get_fignums():
            #plt.figure(i).savefig(
                #f"C:\\Users\\High Sierra\\Documents\\PhD\\SNPMicroArray\\src\\Omni\\{y}\\" + f"{modelUsed} " + str(
                    #i) + '.png')
    plt.show()

    # The OvO strategy is recommended if the user is mainly interested in correctly
    # identifying a particular class or subset of classes, whereas evaluating the
    # global performance of a classifier can still be summarized via a given
    # averaging strategy.
    #
    # Micro-averaged OvR ROC is dominated by the more frequent class, since the
    # counts are pooled. The macro-averaged alternative better reflects the
    # statistics of the less frequent classes, and then is more appropriate when
    # performance on all the classes is deemed equally important.


'''
def crsValRoc(cv,clf, individuals):

    fig, ax = plt.subplots(figsize = (6,6))
    for fold, (train, test) enumerate in cv.split(x,y):
        clf.fit(x[train],y[train])
        cvRoc= RocDisplay().from_estimator(clf, x[test],y[test], name = f"ROC fold {fold}", alpha =0.5, ax= ax, plot_chance_level = (fold == n_splits - 1))
'''

def saveModel(model, modelName):
    joblib.dump(model, modelName)


def loadModel(modelName):
    model = joblib.load(modelName)
    return model


def thresholdViewer(filterProb, thresholds):
    filteredProb = np.where(filterProb > thresholds, 1, 0)
    return filteredProb

def phredQual(yPredProb, df, modelUsed, threshold, volume,individual, evenSample, conFilter):
    #calculate percentage heterozygosity
    classCount = df["AlleleAB_y"].value_counts()
    classPercentage =  classCount/ classCount.sum() * 100
    print("percentages", classPercentage)
    with open('classPercentage.txt', 'w') as f:
        for className, percentage in classPercentage.items():
            f.write(f'{className}: {percentage:.2f}%\n')
    f.close()

    PredProbDf = pd.DataFrame(yPredProb, columns=["A A", "A B", "B B"])
    PhredPredDf = -10 * np.log(1 - PredProbDf)
    PhredPredDf.rename(columns={"A A": "Phred A A", "A B": "Phred A B", "B B": "Phred B B"}, errors="raise",
                       inplace=True)
    PhredPredDf["Max Prob"] = PhredPredDf[["Phred A A", "Phred A B", "Phred B B"]].max(axis=1)
    PhredPredDf["Threshold"] = threshold
    df = pd.concat([df, PhredPredDf], axis=1)
    df = pd.concat([df, PredProbDf], axis=1)
    df.reset_index(drop=True, inplace=True)
    print("Max Prob", df["Max Prob"])
    if conFilter == "withConcordance":
        FilteredPhredPredDf = df[df["Max Prob"] >= threshold]
        FilteredPhredPredDf = concordanceFilter(FilteredPhredPredDf)
        # FilteredPhredPredDf.to_csv(f"FILTEREDPHREDPPED{volume}{threshold}.csv")
        count = len(FilteredPhredPredDf)
        print("Number of SNPs above threshold for concordance", count)
        FilteredGCScoreDf = gcScoreFilter(df, count)
    else:
        FilteredPhredPredDf = df[df["Max Prob"] >= threshold]
        # FilteredPhredPredDf.to_csv(f"FILTEREDPHREDPPED{volume}{threshold}.csv")
        #count = (df['Max Prob'] > threshold).sum()
        count = len(FilteredPhredPredDf)
        print("Number of SNPs above threshold", count)
        FilteredGCScoreDf = gcScoreFilter(df, count)
    #even samples choosing minimium counts with equal amounts of classes for both filtering based on phred scores and on ranking GC scores
    if evenSample == 1:
        minCount = FilteredPhredPredDf["AlleleAB_y"].value_counts().min()
        FilteredPhredPredDf=FilteredPhredPredDf.groupby("AlleleAB_y").sample(n=minCount)
        minCount = FilteredGCScoreDf["AlleleAB_y"].value_counts().min()
        FilteredGCScoreDf = FilteredGCScoreDf.groupby("AlleleAB_y").sample(n=minCount)
        classCount = FilteredPhredPredDf["AlleleAB_y"].value_counts()
        classPercentage = classCount / classCount.sum() * 100
        print("percentages", classPercentage)
        with open('classPercentage.txt', 'w') as f:
            for className, percentage in classPercentage.items():
                f.write(f'{className}: {percentage:.2f}%\n')
        f.close()

    # FilteredPhredPredDf = FilteredPhredPredDf.drop(columns = ["Max value"])
    # draw ROC Curve

    predProb = FilteredPhredPredDf[["A A", "A B", "B B"]]

    actualLabels = FilteredPhredPredDf["AlleleAB_y"]
    trainingLabel = FilteredPhredPredDf["AlleleAB_x"]

    # Roc Curve
    #RocDisplay(actualLabels, trainingLabel, predProb, modelUsed, threshold, volume)
    return FilteredPhredPredDf, FilteredGCScoreDf

def gcScoreFilter(df, cutoff):
    df.sort_values(by=["GCScore"], ascending=False, inplace=True)
    print("Sorted Columns", df)
    df = df.head(cutoff)
    print("Filtered Df based on GC Score...", df)
    return df

def filterTableResults(gcFilterDf, phredFilteredDf, inputDf, filterAccuracy, volume, modelName, thr, individuals, concordanceFilter):

    predictions = {f"{volume}Ng Genome studio": inputDf["AlleleAB_x"], f"{volume}Ng {modelName}": inputDf["Predicted"]}
    for label in predictions.keys():
        if label ==f"{volume}Ng Genome studio":
            actualLabel = gcFilterDf["AlleleAB_y"]
            #GenomeStudio "predictions"
            predLabel = gcFilterDf["AlleleAB_x"]
            displayCMPlot(gcFilterDf, volume, individuals, concordanceFilter,filter=thr, prediction="AlleleAB_x")
            report = pd.DataFrame(
                classification_report(actualLabel, predLabel, labels=["A A", "A B", "B B"], output_dict=True))
            report["Threshold"] = thr
            report["Model"] = f"{volume}Ng Genome studio"
        else:
            actualLabel = phredFilteredDf["AlleleAB_y"]
            predLabel = phredFilteredDf["Predicted"]
            displayCMPlot(phredFilteredDf, volume, individuals, concordanceFilter,thr, modelName, "Predicted")
            report2 = pd.DataFrame(
                classification_report(actualLabel, predLabel, labels=["A A", "A B", "B B"], output_dict=True))
            report2["Threshold"] = thr
            report2["Model"] = f"{volume}Ng {modelName}"
        # Compute and store accuracies
        accuracy = accuracy_score(actualLabel, predLabel)
        colName = f"{label}"
        filterAccuracy.loc[thr, colName] = accuracy
        filterAccuracy.loc[thr, "Threshold"] = thr

        print(f"{label} {concordanceFilter}Classification Report", classification_report(actualLabel, predLabel, labels = ["A A", "A B", "B B"]))
        print(f"{concordanceFilter} Filtered Accuracy ", filterAccuracy)
    report = pd.concat([report, report2], axis = 1)

    return report
    filterAccuracy.to_csv(
        f"(your directory)\\{individuals}\\{modelName}\\{volume}\\{volume}Ng {selectedModel}FilteredAccuracies.csv")
    #filterAccuracy.to_csv(
        #f"/home/ac0539/snp_chip_data/Omni/{individual}/{modelName}/{volume}/{volume}Ng {modelName}FilteredAccuracies.csv")
def analyzeTitrationGroups(titrateGroup, testSamples, dataFeatures, bestModel, modelName, labels = ["A A", "A B", "B B"] , inputNg=[1, 0.5, 0.1, 0.05, 0.01], le = None):
    phredDataframes = []
    gcDataFrames = []
    for i in testSamples:
        # Filter data for the specific sample
        Group = titrateGroup[titrateGroup["ExternalSampleID"] == i]

        for y in inputNg:
            # Filter data for the specific DNA input quantity
            inputNgGroup = Group[Group["DNAinputNG"] == y]
            inputNgGroup.reset_index(drop=True, inplace=True)

            # Extract features and labels for prediction
            xTitrateGroup = inputNgGroup[dataFeatures]
            yTitrateGroup = inputNgGroup["AlleleAB_y"]

            # Make predictions
            pred = bestModel.predict(xTitrateGroup)
            if not le == None :
                pred = le.inverse_transform(pred)
            inputNgGroup["Predicted"] = pred
            predProb = bestModel.predict_proba(xTitrateGroup)


            # Apply Phred score filtering
            for name in ["withoutConcordance", "withConcordance"]:
                # Prepare DataFrame for filtering results
                filterAccuracy = pd.DataFrame(columns=[f"{y}Ng Genome studio",
                                                       f"{y}Ng {modelName}", "Threshold"])
                resultsTable = pd.DataFrame()
                for x in [0]:
                    phredFilteredDf, gcFilterDf = phredQual(predProb, inputNgGroup, f"{modelName}", x, y, i, 0, f"{name}")
                    phredDataframes.append(phredFilteredDf)
                    gcDataFrames.append(gcFilterDf)
                    results= filterTableResults(gcFilterDf, phredFilteredDf, inputNgGroup, filterAccuracy, y, f"{modelName}",
                                       x, i, name)
                    resultsTable = pd.concat([resultsTable, results])
                # Generate confusion matrix
                resultsTable.to_csv(f"(your directory)\\{i}\\{modelName}\\{y}\\{i}{modelName}{name}SummaryStatisticTable {y}Ng.csv",index =False)
            #resultsTable.to_csv(f"(your directory)/{i}/{modelName}/{y}/{i}_{modelName}_{name}_SummaryStatisticTable_{y}Ng.csv",index =False)
def concordanceFilter(predFile):
    conCordpredFile = predFile[predFile["AlleleAB_x"] == predFile["Predicted"]]
    return conCordpredFile


if __name__ == "__main__":
    path = "(your directory)"
    start = time.time()
    dateStr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelName = "Logistic regression"
    log_file = f"Run {dateStr} {modelName} details.txt"
    # machine learning component
    # log likelihood
    def customScore(yTrue, pred):
        global ys
        ys.append(pred)
        negLoss = -log_loss(yTrue, pred)
        return negLoss
    import multiprocessing
    CustomNegLogLoss = make_scorer(customScore, needs_proba=True)
    scorers = {"Accuracy": "accuracy", "CustomNegLog": CustomNegLogLoss, "Roc": "roc_auc_ovo"}
    manager = multiprocessing.Manager()

    print("Reading file...", flush=True)
    joinedDt = readInFile(2)
    print(joinedDt)
    end = time.time()
    print("Done loading", start - end, flush=True)
    joinedDt.isnull().values.any()

    # LABELS NEED TO UPDATED WITH THE TABLES!!
    # one hot encoding
    # working with ng1Table for now...
    # GAM2
    # GLIMPSE 2 pdf lecture
    print("Starting Preprocessing", flush=True)
    start = time.time()
    testingDf = preProcess(joinedDt)
    end = time.time()
    print("finished preprocessing", start - end, flush=True)

    labels = ["A A", "A B", "B B"]
    inputNg = [1, 0.5, 0.1, 0.05, 0.01]

    for i in testSamples:
        Group = testingDf[testingDf["ExternalSampleID"] == i]
        print(f"externalSampleID {i}", Group["ExternalSampleID"])
        genomeResults = Group["AlleleAB_x"]
        genomeAnswers = Group["AlleleAB_y"]
        accuracies = accuracy_score(genomeAnswers, genomeResults)
        print("Accuracy of GenomeStudio overall", accuracies)
        Cm = confusion_matrix(genomeAnswers, genomeResults, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=Cm, display_labels=labels).plot()
        disp.ax_.set_title(f"Genome Studio Total")
        for y in inputNg:
            inputNgGroup = Group[Group["DNAinputNG"] == y]
            xTitrateGroup = inputNgGroup["AlleleAB_x"]
            yTitrateGroup = inputNgGroup["AlleleAB_y"]
            accuracies = accuracy_score(yTitrateGroup, xTitrateGroup)
            accuracies = np.append(accuracies, accuracies)
            print("Genomestudio accuracy each input ng", accuracies)
            LogCm = confusion_matrix(yTitrateGroup, xTitrateGroup, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=LogCm, display_labels=labels).plot()
            disp.ax_.set_title(f"Genome Studio Sample ID {i}, input Ng {y}")
            #plt.savefig(f"../{i}GenomeStudioConfusionMatrix{y}.png")

    '''
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey='row')
    for i, cm in enumerate(volumeTables):
        disp = displayCMPlot(cm)
        disp.plot(ax=axes[i], xticks_rotation=45)
        disp.ax_.set_title(plotTitle[i])
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        # plt.savefig('../output/ConfusionMatrix.png')
    '''
    # ng0_01Table = ng0_01Table.drop(ng0_01Table[ng0_01Table["AlleleAB_y"] == "- -"].index)
    # ng0_01Table = ng0_01Table.drop(ng0_01Table[ng0_01Table["AlleleAB_x"] == "- -"].index)
    dataFeatures = ["ZScoreGCScore", "ZScoreGTScore", "ZScoreClusterSep", "R", "X", "Y", "AlleleAB_x_AA", "Theta",
                    "AlleleAB_x_BB", "AlleleAB_x_AB",
                    "ZScoreXrawYraw", "45Theta", "90Theta", "absZScoreXY", "45SubArc", "SubArc", "90SubArc",
                    "ZScoreXYMean", "ZScoreXYVariance", "Angle Error"]

    # testingDf.to_pickle("processedDf.pkl")
    # parentDirectory = os.path.dirname(os.path.dirname(__file__))
    # path = os.path.join(parentDirectory, "scratch/processedDf.pkl")
    # testingDf.to_pickle(path)

    print("Start Data split", flush=True)
    start = time.time()
    xTrainGroup, testGroup, yTrainGroup, yTestGroup, individuals, testSNPs,  titrateGroup, clsWeights = dataSplit(
        testingDf, 1, dataFeatures, log_file)

    end = time.time()
    print("finished data split", start - end, flush=True)
    # x is data, y is labels
    # xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.3)


    gc.collect()

    startTime = time.time()
    print("startTime", flush=True)
    print("instantiating multilog", startTime - time.time(), flush=True)
    multiLog = LogisticRegression(multi_class='multinomial',
                                  solver='saga', class_weight= "balanced"
                                  )  # change the solver #parameters first go to turn off regularization
    print("finished multiLog", startTime - time.time(), flush=True)
    # k fold 5
    # gridsearch for hyperparameter tuning
    gkf = GroupKFold(n_splits=7)
    # scikit applies l2 automatically
    parameters = {
        'penalty': ["None", "l1", "l2", "elasticnet"],
        # 'C': np.logspace(-4,4,20),
        'C': [120],
        'l1_ratio': [0.5, 0.1],
        'max_iter': [10000]
    }
    with open(log_file, "w") as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

    print("instantiating gridSearch", flush=True)
    ys = manager.list()
    gS = GridSearchCV(multiLog, parameters, scoring=scorers, refit="CustomNegLog", cv=gkf, n_jobs=7)
    print("finished gs instantiating", startTime - time.time(), flush=True)
    print("Grid search", flush=True)
    results = gS.fit(xTrainGroup, yTrainGroup, groups=individuals,sample_weight =clsWeights)

    gsPredProb = list(ys)
    print(gsPredProb, flush=True)
    print("Writing results of grid search", startTime - time.time(), flush=True)
    #
    gsDf = pd.DataFrame(gS.cv_results_)
    saveModel(gS, "testModelSave.pkl")
    gS = loadModel("testModelSave.pkl")

    bestModel = gS.best_estimator_
    #for individuals in testSamples:
        #os.makedirs(f"/home/ac0539/snp_chip_data/Omni/{individual}/{modelName}/", exist_ok=True)
        #saveModel(gS, f"/home/ac0539/snp_chip_data/Omni/{individuals}/{modelName}/{modelName}.pkl")
        # gS = loadModel(f"/home/ac0539/snp_chip_data/Omni/{individuals}/{modelName}/{modelName}.pkl")
        #gsDf.to_csv(f"/home/ac0539/snp_chip_data/Omni/{individuals}/{modelName}/{modelName}GSresults.csv", index=False)
    # cvResult =cross_val_score(bestModel,xTrainGroup,yTrainGroup,scoring = "neg_log_loss", cv = gkf, groups= individuals)
    # print("Y Train Columns group", yTrainGroup, flush = True)
    # print("predicting best model", flush=True)
    # print("results", cvResult, flush=True)

    yPred = bestModel.predict(testGroup)
    yPredProb = bestModel.predict_proba(testGroup)
    #RocDisplay(yTrainGroup, yTestGroup, yPredProb, "Logistic Regression Test Group")

    accuracies = accuracy_score(yTestGroup, yPred)
    accuracies = np.append(accuracies, accuracies)
    # confusion matrix
    LogCm = confusion_matrix(yTestGroup, yPred, labels=labels)
    ConfusionMatrixDisplay(confusion_matrix=LogCm, display_labels=labels).plot()
    # coefficients and intercept of the logistic regression model
    print("printing coefficients", flush=True)
    coefficients = pd.DataFrame(bestModel.coef_, columns=bestModel.feature_names_in_)
    coefficients["intercepts"] = bestModel.intercept_
    coefficients = coefficients.transpose()
    coefficients.columns = [bestModel.classes_]
    print(coefficients)
    coefficients.to_csv("LogisticRegressionCoefficient.csv")
    plt.title("LogCM All input ng accuracy")
    print("Log accuracy", accuracies)

    print("Log Likelihood", yPredProb)
    # predProb = pd.DataFrame(yPredProb, columns=model.classes_)
    # predProb.to_csv("LrPredProb.csv")
    # null classifier
    # nullClf(testGroup,yTestGroup)

    analyzeTitrationGroups(titrateGroup, testSamples, dataFeatures, bestModel,modelName)
    endTime = time.time()
    duration = endTime - startTime
    with open(log_file, "a") as f:
        f.write(f"duration {duration}\n")
    #resultCSV(f"{modelName}",yPredProb,yPred,testSNPs)
    print(duration, flush=True)

    plt.close()

    # plt.show()
