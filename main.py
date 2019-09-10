import random
import sys
from CondensedKNearestNeighbors import CondensedKNearestNeighbors
from CrossValidation import kFoldCrossValidation
import eColiProcess
import firesProcess
from kNearestNeighbors import kNearestNeighbors
import machineProcess
import segmentProcess

"""
This is a main method to call all of the other scripts, takes one of the four data sets for this assignment and calls either kNearestNeighbors or condensedKNN
"""

if 'ecoli' in sys.argv[1].lower():
    dataset = eColiProcess.process(sys.argv[1])

elif 'forestfires' in sys.argv[1].lower():
    dataset = firesProcess.process(sys.argv[1])

elif 'machine' in sys.argv[1].lower():
    dataset = machineProcess.process(sys.argv[1])

elif 'segment' in sys.argv[1].lower():
    dataset = segmentProcess.process(sys.argv[1])

if sys.argv[2].lower() == "classify":  # if it's a classification set, needs to go to CrossValidation to go through stratification
    crossFolds = kFoldCrossValidation(dataset)
    train = []
    accuracy = []
    if sys.argv[3].lower() == "knn":  # go to kNearestNeighbors without condensing
        for i in range(len(crossFolds)):
            train = []
            for crossFold in crossFolds[:i] + crossFolds[i + 1:]:  # use all crossfolds but the test one as training set
                train.extend(crossFold)
            knn = kNearestNeighbors(train, 2)  # calling knn is place to change k value
            mistakes = 0
            for obs in crossFolds[i]:  # use other crossfold for testing
                if knn.classify(obs) != obs.classifier:  # if the knn classifies incorrectly, list as mistake
                    mistakes += 1
            accuracy.append((len(crossFolds[i]) - mistakes) / len(crossFolds[i]))  # get the accuracies of each cross-validation run
    elif sys.argv[3].lower() == "condensed":
        for i in range(len(crossFolds)):  # same all regular knn but go to condensed KNN and only use that data to classify
            train = []
            for crossFold in crossFolds[:i] + crossFolds[i + 1:]:
                train.extend(crossFold)
            knn = CondensedKNearestNeighbors(train, 2)
            mistakes = 0
            for obs in crossFolds[i]:
                if knn.classify(obs) != obs.classifier:
                    mistakes += 1
            accuracy.append((len(crossFolds[i]) - mistakes) / len(crossFolds[i]))
    print("Average accuracy over five-fold cross-validation:")
    print(sum(accuracy) / len(accuracy))  # return the average accuracy of the five runs

elif sys.argv[2].lower() == "regression":  # just put the regression cross-validation in here since it isn't as complicated
    random.shuffle(dataset)  # randomly shuffle the data
    crossFolds = [[] for i in range(5)]
    index = 0
    meanSquaredError = []

    while index < len(dataset):  # add one Observation to each crossfold until the dataset has all been distributed
        for crossFold in crossFolds:
            if index >= len(dataset):
                break
            else:
                crossFold.append(dataset[index])
                index += 1

    for i in range(len(crossFolds)):  # doesn't use condensed so just goes through and runs knn on each testing and training combination
        train = []
        for crossFold in crossFolds[:i] + crossFolds[i + 1:]:
            train.extend(crossFold)
        knn = kNearestNeighbors(train, 2)  # change k if needed
        MSE = 0
        for obs in crossFolds[i]:  # find mean squared error for each testing and training set
            MSE += ((obs.classifier - knn.regression(obs)) ** 2 / len(crossFolds[i]))
        meanSquaredError.append(MSE)

    avgMeanSquaredError = sum(meanSquaredError) / len(meanSquaredError)

    print("Average mean-squared error over five-fold cross-validation:")
    print(avgMeanSquaredError)  # return average mean-squared error for the cross-validated set
