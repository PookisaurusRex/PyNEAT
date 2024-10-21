"""Keeps track of whether species are making progress and helps remove ones that are not."""
import sys

from NEAT.Config import ConfigParameter, DefaultClassConfig
from NEAT.MathUtils import StatisticalFunctions

# TODO: Add a method for the user to change the "is stagnant" computation.


class DefaultStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""
    @classmethod
    def ParseConfig(cls, ParameterDictionary):
        return DefaultClassConfig(ParameterDictionary,
                                  [ConfigParameter('SpeciesFitnessFunction', str, 'mean'),
                                   ConfigParameter('MaxStagnation', int, 15),
                                   ConfigParameter('SpeciesElitism', int, 0),
                                   ConfigParameter('bIgnoreStagnationAtMax', bool, True),
                                   ConfigParameter('bEnforceMinimumFitness', bool, True),
                                   ConfigParameter('MinimumFitnessThreshold', float, 0.0)])

    def __init__(self, Config, Reporters):
        # pylint: disable=super-init-not-called
        self.StagnationConfig = Config
        self.SpeciesFitnessFunction = StatisticalFunctions.get(Config.SpeciesFitnessFunction)
        if self.SpeciesFitnessFunction is None:
            raise RuntimeError("Unexpected species fitness func: {0!r}".format(Config.SpeciesFitnessFunction))
        self.Reporters = Reporters

    def Update(self, SpeciesSet, Generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """
        SpeciesData = []
        for SpeciesID, Species in SpeciesSet.Species.items():
            MaxHistoricalFitness = max(Species.FitnessHistory) if Species.FitnessHistory else -sys.float_info.max
            Species.Fitness = self.SpeciesFitnessFunction(Species.GetFitnesses())
            Species.FitnessHistory.append(Species.Fitness)
            Species.AdjustedFitness = None
            if self.StagnationConfig.bEnforceMinimumFitness and Species.Fitness <= self.StagnationConfig.MinimumFitnessThreshold:
                pass
            elif (MaxHistoricalFitness is None) or (Species.Fitness > MaxHistoricalFitness) or (self.StagnationConfig.bIgnoreStagnationAtMax and (Species.Fitness >= MaxHistoricalFitness)):
                Species.LastImproved = Generation
            SpeciesData.append((SpeciesID, Species))
        # Sort in ascending fitness order.
        SpeciesData.sort(key=lambda Species: Species[1].Fitness) # Sorted into ascending fitness order; less fit species stagnate first.
        Result = []
        SpeciesFitnesses = []
        NumNonStagnant = len(SpeciesData)
        for Index, (SpeciesID, Species) in enumerate(SpeciesData):
            bStagnant = False
            if (NumNonStagnant > self.StagnationConfig.SpeciesElitism):
                bStagnant = (Generation - Species.LastImproved > self.StagnationConfig.MaxStagnation)
            if (len(SpeciesData) - Index <= self.StagnationConfig.SpeciesElitism):
                bStagnant = False
            if bStagnant:
                NumNonStagnant -= 1
            Species.bStagnant = bStagnant
            Result.append((SpeciesID, Species))
            SpeciesFitnesses.append(Species.Fitness)
        return Result
