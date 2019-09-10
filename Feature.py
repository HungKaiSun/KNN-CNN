class Feature():

    def __init__(self, type, value):
        """
        @param type Categorical or Continuous, passed in from the CategoricalFeature and ContinuousFeature classes
        @param value The numerical or string value of the feature 
        Instantiates a new feature with the type (categorical or continuous) and the value of the feature
        """
        self._type = type
        self._value = value

    def isComparable(self, otherFeature):
        """
        @param otherFeature: Feature class to determine comparability with 
        Check to see if this feature is comparable with another feature
        """
        return self._type == otherFeature._type

    def getDistMetric(self):
        """
        If the Feature is Continuous or Categorical, its distance metric is returned
        """

        def noDistMetric():
            """
            Raises an error if the Feature type is not Continuous or Categorical
            """
            raise NotImplementedError("Distance metric is not supported on feature type")
            return noDistMetric
