class kNearestNeighbors:
    """
    This class takes a training set and then can be used to find the nearest neighbors between the training set and a test set
    """

    def __init__(self, train, k=1):
        """
        @param train The training set (in our case, the 80% of the data for each round of cross-validation)
        @param k=1 The value for k, with default set at 1
        Constructs new kNearestNeighbors object with a training set and the value of k
        """
        self.train = train
        self.k = k

    def findAllDistance(self, testPoint):
        """
        @param testPoint An Observation taken from the test set
        Using the training set, one Observation from the test set is passed in and the distance from it to every point in the training set is found
        """
        neighbors = {}
        nearestNeighbors = []

        for point in self.train:  # for each point in the training set, find the distance to the testPoint
            neighbors[point] = point.calcDistance(testPoint)

        for i in range(self.k):  # go through k times and find the k nearest neighbors to the test point
            nearestNeighbor = min(neighbors, key=neighbors.get)
            nearestNeighbors.append(nearestNeighbor)
            neighbors.pop(nearestNeighbor)

        return nearestNeighbors

    def classify(self, testPoint):
        """
        @param testPoint
        If the dataset is a classification set, find the most common class in the set of k nearest neighbors and assign that class to the test point
        """
        classCount = {}
        for neighbor in self.findAllDistance(testPoint):
            classCount[neighbor.classifier] = classCount.get(neighbor.classifier, 0) + 1  # go through nearest neighbors and count which class neighbors belong to

        classification = max(classCount, key=classCount.get)  # return which class the majority of the nearest neighbors belong to
        return classification

    def regression(self, testPoint):
        """
        @param testPoint
        If the dataset is a regression set, find the mean of the regression values and apply it to the test point
        """
        total = 0
        nearestNeighbors = self.findAllDistance(testPoint)

        for neighbor in nearestNeighbors:
            total += float(neighbor.classifier)

        mean = total / len(nearestNeighbors)
        return mean
