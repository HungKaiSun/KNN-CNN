from Feature import Feature


class CategoricalFeature(Feature):
    """
    This class calls the parent Feature class and returns the type and distance metric for a categorical feature
    """

    def __init__(self, value):
        """
        Instantiates a new Feature of "categorical" type
        """
        super().__init__('categorical', value)

    def getDistMetric(self):
        """
        @return feat1.value == feat2.value
        This is for binary values or something like date or day, where there isn't really an origin value
        """
        return lambda feat1, feat2: int(feat1._value == feat2._value)  # takes the value for each Feature and returns whether it is the same or different
