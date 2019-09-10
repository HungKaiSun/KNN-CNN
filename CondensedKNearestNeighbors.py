import random
from kNearestNeighbors import kNearestNeighbors


class CondensedKNearestNeighbors(kNearestNeighbors):
    """
    This takes a set of training data and adds relevant samples one by one until no more are added. 
    It then creates a kNearestNeighbors object using just those samples.
    """

    def __init__(self, train, k=1):
        """
        @param train The full training set to be cut down
        @param k=1 A k value, with the default set to 1
        """

        samples = []  # sample Observations
        randomSample = train.pop(random.randint(0, len(train) - 1))
        samples.append(randomSample)  # Choose a random starting Observation

        numberOfSamples = len(samples)
        while True:  # continue until no more samples are added
            minSampleDistance = 99999  # set initial distance high to always improve at the beginning
            closestSample = None  # there is no closest potential sample at the start
            for index, observation in enumerate(train):
                for sample in samples:
                    sampleDistance = observation.calcDistance(sample)
                    if sampleDistance < minSampleDistance:
                        minSampleDistance = sampleDistance
                        closestSample = sample
                if closestSample.classifier == observation.classifier:  # do not consider for reduction, these are the same class and the closest sample will represent it
                    continue
                else:  # current observation is a different class from its closest sample, add this to the reduced dataset
                    samples.append(train.pop(index))
            if len(samples) == numberOfSamples:  # no new samples on this pass, exit the while loop
                break
            numberOfSamples = len(samples)  # update the number of samples to check against the next run
        print("Number of samples selected: " + str(len(samples)))

        super().__init__(samples, k)  # initialize a kNearestNeigbors object with these samples as training set
