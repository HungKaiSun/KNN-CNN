class Observation:
    """
    This class creates an Observation instance, which is an object with a Class/Label and a set of Features associated.
    I decided to go through and set up a more object-oriented approach since I realized we are going to keep using observations, 
    classes and features for the assignments.
    """

    def __init__(self, classifier, features):
        """
        @param classifier The class/label or regression value for an observation
        @param features A list of Feature objects that represent the values of each feature in the observation
        The init method instantiates the Observation, giving it a class and a set of Features.
        """
        if features is None:  # each observation needs features, but a testing set with unknown class can be used
            raise ValueError("Features cannot be null")
        self.features = features
        self.classifier = classifier

    def calcDistance(self, other):
        """
        @param other This is the second point/Observation from which the distance to this Observation should be calculated 
        @return distance Returns the total distance between the two Observations over all features
        This method determines which distance metric should be used for each feature and calculates the distance, then returns the total distance between
        the two observations. 
        """
        distance = 0
        for feat1, feat2 in zip(self.features, other.features):  # go through each feature and calculate the distance between the two
            metric = feat1.getDistMetric()  # found in the Categorical/ContinuousFeature classes
            distance += metric(feat1, feat2)  # add all distances together
        return distance
