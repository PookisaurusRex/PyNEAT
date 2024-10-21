from __future__ import print_function

import os
import warnings
from configparser import ConfigParser

class ConfigParameter(object):
    """Contains information about one configuration item."""
    def __init__(self, Name, DataType, DefaultValue=None):
        self.Name = Name
        self.DataType = DataType
        self.DefaultValue = DefaultValue

    def __repr__(self):
        if self.DefaultValue is None:
            return "ConfigParameter({!r}, {!r})".format(self.Name, self.DataType)
        return "ConfigParameter({!r}, {!r}, {!r})".format(self.Name, self.DataType, self.DefaultValue)

    def Parse(self, Section, Config):
        if (int == self.DataType):
            return Config.getint(Section, self.Name)
        if (bool == self.DataType):
            return Config.getboolean(Section, self.Name)
        if (float == self.DataType):
            return Config.getfloat(Section, self.Name)
        if (list == self.DataType):
            Intermediate = Config.get(Section, self.Name)
            return Intermediate.split(" ")
        if (str == self.DataType):
            return Config.get(Section, self.Name)
        raise RuntimeError("Unexpected configuration type: " + repr(self.DataType))

    def Interpret(self, ConfigDict):
        """
        Converts the config_parser output into the proper type,
        supplies defaults if available and needed, and checks for some errors.
        """
        Value = ConfigDict.get(self.Name.lower())
        if Value is None:
            if self.DefaultValue is None:
                raise RuntimeError('Missing configuration item: ' + self.Name)
            else:
                warnings.warn("Using default {!r} for '{!s}'".format(self.DefaultValue, self.Name), DeprecationWarning)
                if (str != self.DataType) and isinstance(self.DefaultValue, self.DataType):
                    return self.DefaultValue
                else:
                    Value = self.DefaultValue
        try:
            if str == self.DataType:
                return str(Value)
            if int == self.DataType:
                return int(Value)
            if bool == self.DataType:
                if Value.lower() == "true":
                    return True
                elif Value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.Name + " must be True or False")
            if float == self.DataType:
                return float(Value)
            if list == self.DataType:
                return Value.split(" ")
        except Exception:
            raise RuntimeError("Error interpreting Config item '{}' with value {!r} and type {}".format(self.Name, Value, self.DataType))
        raise RuntimeError("Unexpected configuration type: " + repr(self.DataType))

    def Format(self, Value):
        if list == self.DataType:
            return " ".join(Value)
        return str(Value)

def WritePrettyParameters(File, Config, Parameters):
    ParameterNames = [Parameter.Name for Parameter in Parameters]
    LongestName = max(len(Name) for Name in ParameterNames)
    ParameterNames.sort()
    Parameters = dict((Parameter.Name, Parameter) for Parameter in Parameters)
    for Name in ParameterNames:
        Parameter = Parameters[Name]
        File.write('{} = {}\n'.format(Parameter.Name.ljust(LongestName), Parameter.Format(getattr(Config, Parameter.Name))))

class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""
    pass

class DefaultClassConfig(object):
    """
    Replaces at least some boilerplate configuration code
    for reproduction, species_set, and stagnation classes.
    """
    def __init__(self, ParameterDictionary, ParameterList):
        self.Parameters = ParameterList
        ParameterListNames = []
        for Parameter in ParameterList:
            setattr(self, Parameter.Name, Parameter.Interpret(ParameterDictionary))
            ParameterListNames.append(Parameter.Name.lower())
        UnknownList = [ParameterName for ParameterName in ParameterDictionary if ParameterName not in ParameterListNames]
        if UnknownList:
            if len(UnknownList) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" + "\n\t".join(UnknownList))
            raise UnknownConfigItemError("Unknown configuration item {!s}".format(UnknownList[0]))

    @classmethod
    def WriteConfig(cls, File, Config):
        WritePrettyParameters(File, Config, Config.Parameters)

class Config(object):
    """A simple container for user-configurable parameters of NEAT."""
    NEATParameters = [ConfigParameter('PopulationSize', int),
                ConfigParameter('MinimumGeneration', int),
                ConfigParameter('FitnessCriterion', str),
                ConfigParameter("FitnessGroup", str, "population"),
                ConfigParameter('FitnessThreshold', float),
                ConfigParameter('bResetOnExtinction', bool),
                ConfigParameter('bDisableFitnessTermination', bool, False)]

    def __init__(self, GenomeType, ReproductionType, SpeciesSetType, StagnationType, Filename):
        # Check that the provided types have the required methods.
        assert hasattr(GenomeType, 'ParseConfig')
        assert hasattr(ReproductionType, 'ParseConfig')
        assert hasattr(SpeciesSetType, 'ParseConfig')
        assert hasattr(StagnationType, 'ParseConfig')

        self.GenomeType = GenomeType
        self.ReproductionType = ReproductionType
        self.SpeciesSetType = SpeciesSetType
        self.StagnationType = StagnationType

        if not os.path.isfile(Filename):
            raise Exception('No such Config file: ' + os.path.abspath(Filename))

        Parameters = ConfigParser()
        with open(Filename) as File:
            if hasattr(Parameters, 'read_file'):
                Parameters.read_file(File)
            else:
                Parameters.readfp(File)

        # NEAT configuration
        if not Parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        ParameterListNames = []
        for Parameter in self.NEATParameters:
            if Parameter.DefaultValue is None:
                setattr(self, Parameter.Name, Parameter.Parse('NEAT', Parameters))
            else:
                try:
                    setattr(self, Parameter.Name, Parameter.Parse('NEAT', Parameters))
                except Exception:
                    setattr(self, Parameter.Name, Parameter.DefaultValue)
                    warnings.warn("Using default {!r} for '{!s}'".format(Parameter.DefaultValue, Parameter.Name), DeprecationWarning)
            ParameterListNames.append(Parameter.Name.lower())
        ParameterDictionary = dict(Parameters.items('NEAT'))
        UnknownParameterList = [Parameter for Parameter in ParameterDictionary if Parameter not in ParameterListNames]
        if UnknownParameterList:
            if len(UnknownParameterList) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(UnknownParameterList))
            raise UnknownConfigItemError("Unknown (section 'NEAT') configuration item {!s}".format(UnknownParameterList[0]))

        # Parse type sections.
        GenoneDictionary = dict(Parameters.items(GenomeType.__name__))
        self.GenomeConfig = GenomeType.ParseConfig(GenoneDictionary)

        SpeciesSetDictionary = dict(Parameters.items(SpeciesSetType.__name__))
        self.SpeciesSetConfig = SpeciesSetType.ParseConfig(SpeciesSetDictionary)

        StagnationDictionary = dict(Parameters.items(StagnationType.__name__))
        self.StagnationConfig = StagnationType.ParseConfig(StagnationDictionary)

        ReproductionDictionary = dict(Parameters.items(ReproductionType.__name__))
        self.ReproductionConfig = ReproductionType.ParseConfig(ReproductionDictionary)

    def Save(self, Filename):
        with open(Filename, 'w') as File:
            File.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            File.write('# or the experiment itself.  This is the only required section.\n')
            File.write('[NEAT]\n')
            WritePrettyParameters(File, self, self.NEATParameters)

            File.write('\n[{0}]\n'.format(self.GenomeType.__name__))
            self.GenomeType.WriteConfig(File, self.GenomeConfig)

            File.write('\n[{0}]\n'.format(self.SpeciesSetType.__name__))
            self.SpeciesSetType.WriteConfig(File, self.SpeciesSetConfig)

            File.write('\n[{0}]\n'.format(self.StagnationType.__name__))
            self.StagnationType.WriteConfig(File, self.StagnationConfig)

            File.write('\n[{0}]\n'.format(self.ReproductionType.__name__))
            self.ReproductionType.WriteConfig(File, self.ReproductionConfig)