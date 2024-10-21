from NEAT.Config import ConfigParameter

from random import choice, gauss, random, uniform


class BaseAttribute(object):
    """Superclass for the type-specialized attribute subclasses, used by genes."""
    def __init__(self, Name, **DefaultDictionary):
        self.Name = Name
        for ItemName, DefaultValue in DefaultDictionary.items():
            self.ConfigItems[ItemName] = [self.ConfigItems[ItemName][0], DefaultValue]
        for ItemName in self.ConfigItems:
            setattr(self, ItemName + "Name", self.ConfigItemName(ItemName))

    def ConfigItemName(self, ConfigItemBaseName):
        return "{0}{1}".format(self.Name, ConfigItemBaseName)

    def GetConfigParameters(self):
        return [ConfigParameter(self.ConfigItemName(ItemName), self.ConfigItems[ItemName][0], self.ConfigItems[ItemName][1]) for ItemName in self.ConfigItems]

class FloatAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    ConfigItems = {"InitMean": [float, None],
                     "InitStandardDev": [float, None],
                     "InitType": [str, 'gaussian'],
                     "ReplaceRate": [float, None],
                     "MutateRate": [float, None],
                     "MutatePower": [float, None],
                     "MaxValue": [float, None],
                     "MinValue": [float, None]}

    def Clamp(self, Value, Config):
        MinValue = getattr(Config, self.MinValueName)
        MaxValue = getattr(Config, self.MaxValueName)
        return max(min(Value, MaxValue), MinValue)

    def InitValue(self, Config):
        MeanValue = getattr(Config, self.InitMeanName)
        StandardDeviation = getattr(Config, self.InitStandardDevName)
        InitType = getattr(Config, self.InitTypeName).lower()
        if ('gauss' in InitType) or ('normal' in InitType):
            return self.Clamp(gauss(MeanValue, StandardDeviation), Config)
        if 'uniform' in InitType:
            MinValue = max(getattr(Config, self.MinValueName), (MeanValue - (2 * StandardDeviation)))
            MaxValue = min(getattr(Config, self.MaxValueName), (MeanValue + (2 * StandardDeviation)))
            return uniform(MinValue, MaxValue)
        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(Config, self.InitTypeName), self.InitTypeName))

    def MutateValue(self, Value, Config):
        if random() < getattr(Config, self.MutateRateName):
            MutatePower = getattr(Config, self.MutatePowerName)
            return self.Clamp(Value + gauss(0.0, MutatePower), Config)
        if random() < getattr(Config, self.ReplaceRateName):
            return self.InitValue(Config)
        return Value

    def Validate(self, Config):  # pragma: no cover
        pass

class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    ConfigItems = \
    {
        "Default": [str, None],
        "MutateRate": [float, None],
        "MutateTrueRate": [float, 0.0],
        "MutateFalseRate": [float, 0.0]
    }

    def InitValue(self, Config):
        Default = str(getattr(Config, self.DefaultName)).lower()
        if Default in ('1', 'on', 'yes', 'true'):
            return True
        elif Default in ('0', 'off', 'no', 'false'):
            return False
        elif Default in ('random', 'none'):
            return bool(random() < 0.5)
        raise RuntimeError("Unknown default value {!r} for {!s}".format(Default, self.name))

    def MutateValue(self, Value, Config):
        MutateRate = getattr(Config, self.MutateRateName)
        MutateRate += getattr(Config, self.MutateFalseRateName) if Value else getattr(Config, self.MutateTrueRateName)
        if (MutateRate > 0) and (random() < MutateRate):
            return False if Value else True
        return Value

    def Validate(self, Config):  # pragma: no cover
        pass

class StringAttribute(BaseAttribute):
    """
    Class for string attributes such as the aggregation function of a node,
    which are selected from a list of options.
    """
    ConfigItems = \
    {
        "Default": [str, 'random'],
        "Options": [list, None],
        "MutateRate": [float, None]
    }

    def InitValue(self, Config):
        Default = getattr(Config, self.DefaultName)
        if Default.lower() in ('none', 'random'):
            Options = getattr(Config, self.OptionsName)
            return choice(Options)
        return Default

    def MutateValue(self, Value, Config):
        MutateRate = getattr(Config, self.MutateRateName)
        if MutateRate > 0 and random() < MutateRate:
            NewValue = Value
            Options = getattr(Config, self.OptionsName)
            while NewValue == Value:
                NewValue = choice(Options)
            return NewValue
        return Value

    def Validate(self, Config):  # pragma: no cover
        pass