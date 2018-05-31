import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.context import SparkContext
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

sc = SparkContext('local', 'test')

# Load and parse the data
def parsePoint(line, ind):
    arr = line.split(',')
    values = [float(x) for x in arr]
    return LabeledPoint(values[ind], values[4:])

# Remove rows with Nan
def filterRows(row):
    arr = row.split(',')
    if 'nan' in arr:
        return False
    return True

def trainAndTest(parsedData):
    # split input into features and labels
    features = np.array(parsedData.map(lambda lp: lp.features.toArray()).collect())
    labels = np.array(parsedData.map(lambda lp: lp.label).collect())
    # split data into training and test data
    features, testfeatures, labels, testlabels = train_test_split(features, labels, test_size = 0.2, random_state = 0)
    # run Lasso regression to fid the train data
    lm = linear_model.Lasso(alpha=0.01)
    lm.fit(features, labels)
    # calculate predictions on train data & calculate Mean Squared Error & R2 score
    y_pred = lm.predict(features)
    print("Mean Squared Error = ", sklearn.metrics.mean_squared_error(labels, y_pred))
    print("R2 score = ", sklearn.metrics.r2_score(labels, y_pred))
    # calculate predictions on test data & calculate Mean Squared Error & R2 score
    y_pred_test = lm.predict(testfeatures)
    print("Mean Squared Error = ", sklearn.metrics.mean_squared_error(testlabels, y_pred_test))
    print("R2 score = ", sklearn.metrics.r2_score(testlabels, y_pred_test))
    # plot graph of prediction vs ground truth
    plt.scatter(labels, y_pred, color='black')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.show()
    plt.savefig('result.png')

# load the prepared dataset
data = sc.textFile("data/testFeatLabs.csv")
printStat = ['Below High School Education level', 'High School Education level', 'Some College Education level', 'Bachelors Degree and above Education Level' ]

# run Lasso regression to predict the value for each class
for ind in range(4):
    print('\nTraining for', printStat[ind], ':')
    parsedData = data.filter(filterRows).map(lambda x: parsePoint(x, ind))
    trainAndTest(parsedData)