import warnings
from random import random
from NEAT.Attributes import FloatAttribute, BoolAttribute, StringAttribute

class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """
    def __init__(self, Key):
        self.Key = Key

    def __str__(self):
        attrib = ['Key'] + [a.Name for a in self.GeneAttributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, Other):
        assert isinstance(self.Key, type(Other.Key)), "Cannot compare keys {0!r} and {1!r}".format(self.Key, Other.Key)
        return self.Key < Other.Key

    @classmethod
    def ParseConfig(cls, Config, ParameterDictionary):
        pass

    @classmethod
    def GetConfigParameters(cls):
        return [Attribute.GetConfigParameters() for Attribute in cls.GeneAttributes]

    def InitAttributes(self, Config):
        for Attribute in self.GeneAttributes:
            setattr(self, Attribute.Name, Attribute.InitValue(Config))

    def Mutate(self, Config):
        for Attribute in self.GeneAttributes:
            Value = getattr(self, Attribute.Name)
            NewAttributeValue = Attribute.MutateValue(Value, Config)
            setattr(self, Attribute.Name, NewAttributeValue)

    def Copy(self):
        NewGene = self.__class__(self.Key)
        for Attribute in self.GeneAttributes:
            setattr(NewGene, Attribute.Name, getattr(self, Attribute.Name))
        return NewGene

    def Crossover(self, Other):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.Key == Other.Key
        NewGene = self.__class__(self.Key)
        for Attribute in self.GeneAttributes:
            AttributeParent = self if (random() > 0.5) else Other
            setattr(NewGene, Attribute.Name, getattr(AttributeParent, Attribute.Name))
        return NewGene

class DefaultNodeGene(BaseGene):
    GeneAttributes = [FloatAttribute('Bias'),
                      FloatAttribute('Response'),
                      StringAttribute('Activation', Options='sigmoid'),
                      StringAttribute('Aggregation', Options='sum')]

    def __init__(self, Key):
        assert isinstance(Key, int), "DefaultNodeGene key must be an int, not {!r}".format(Key)
        BaseGene.__init__(self, Key)

    def Distance(self, Other, Config):
        GeneDistance = abs(self.Bias - Other.Bias) + abs(self.Response - Other.Response)
        if self.Activation != Other.Activation:
            GeneDistance += 1.0
        if self.Aggregation != Other.Aggregation:
            GeneDistance += 1.0
        return GeneDistance * Config.CompatibilityWeightCoefficient

# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    GeneAttributes = [FloatAttribute('Weight'),
                        BoolAttribute('bEnabled')]

    def __init__(self, Key):
        assert isinstance(Key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(Key)
        BaseGene.__init__(self, Key)

    def Distance(self, Other, Config):
        GeneDistance = abs(self.Weight - Other.Weight)
        if self.bEnabled != Other.bEnabled:
            GeneDistance += 1.0
        return GeneDistance * Config.CompatibilityWeightCoefficient