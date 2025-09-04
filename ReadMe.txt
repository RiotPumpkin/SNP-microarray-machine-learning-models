# Classifiers for Illumina Omni5-4

## Three machine learning classifiers

* Multinomial logistic regression (MLR)
* Xgboost
* Neural Networks

## Inputs
* Uses Genomestudio manifest files. 

## Instructions
Training and testing are included in the same script.
A toy dataset labeled testSnps.csv is included for testing. The script was created for a titration series [50ng, 1ng, 0.5ng, 0.1ng, 0.05ng, 0.01ng]
There are three pyscripts. Main.py, XgBoostSNP.py, NeuralNetworkSNPmicroarray.py
Main.py contains the MLR model, XgBooost.py contains the XgBoost model and NeuralNetworkSNPmicroarray.py contains the neural network model.

The manifest file feature names must be edited to not include spaces. Allele-1-AB and Allele-2-AB needs to be combined together to create a column named AlleleAB with values "A A", "A B", or "B B". Any non-autosomal chromosomes needs to be removed. Any invalid entries such as nulls, "- -". The ground truth which in this case the 50ng was concatenated with each titration series matched using SNP ID, Chr, Fa , and ?
The RMLR is in the main.py file along with the def. XGBoost.py and NeuralNetworkSNPMicoarry.py imports def from main.py. Make sure to specify the filepath to your files. SampleIDs must be set to the samples from the input.

The features used to train the models are "ZScoreGCScore", "ZScoreGTScore", "ZScoreClusterSep", "R", "X", "Y", "AlleleAB_x_AA", "Theta",
                    "AlleleAB_x_BB", "AlleleAB_x_AB",
                    "ZScoreXrawYraw", "45Theta", "90Theta", "absZScoreXY", "45SubArc", "SubArc", "90SubArc",
                    "ZScoreXYMean", "ZScoreXYVariance", "Angle Error"
set the flag in downSample in dataSplpit to 0 or 1 for downSampling of sample. If the flag is set to 1 an equal amount of incorrect and correct genotypes are called. If the flag is set to 0 the full training set is used and with equal weight balanced for all three classes.
The evenSample flag in predQual can be set to 0 or 1. 1 means there is an equal amount values reported for all three classes in the summaryStatistic file. 0 means no even sampling is used.
Adjust the hyperparmeters in the gridsearch to what is appropriate for your data. 

Load the testSnps.csv by changing the directory to where your toy dataset is saved.
Edit the (your directory) in the script to where you want to save your files. 

If you want to run a pickled model use the ""
## Outputs
Several files will be created containing the results of the model will be created. 
Folders with corresponding model and input
A .csv file containing the coefficients for RMLR called LogisticRegressionCoefficient.csv
A .pkl of the trained modeled will be created.
A .csv file containing summary statistics called (phred score) (model name) (concordance filter type)SummaryStatisticTable 
A .txt file called run (date run)(your model). txt  containing how the script ran for and whether the sample was downsampled. 
