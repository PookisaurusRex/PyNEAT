"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division

import math
import numpy as np
from copy import copy
from itertools import count
from random import choice, random, shuffle
from NEAT.Config import ConfigParameter, DefaultClassConfig
from Benchmarking import PerfTimer

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def ParseConfig(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('Elitism', int, 1),
                                   ConfigParameter('SurvivalThreshold', float, 0.2),
                                   ConfigParameter('CrossoverChance', float, 0.8),
                                   ConfigParameter('CrossSpeciesCrossoverChance', float, 0.05),
                                   ConfigParameter('MutateChance', float, 1.0),
                                   ConfigParameter('MinSpeciesSize', int, 2),
                                   ConfigParameter('AdjustedFitType', str, 'normalized'),
                                   ConfigParameter('MinAdjustedFitnessRange', float, 1.0),])

    def __init__(self, Config, Reporters, Stagnation):
        # pylint: disable=super-init-not-called
        self.ReproductionConfig = Config
        self.Reporters = Reporters
        self.GenomeIndexer = count(1)
        self.Stagnation = Stagnation
        self.Ancestors = {}

    def CreateNew(self, GenomeType, GenomeConfig, NumGenomes):
        NewGenomes = {}
        for i in range(NumGenomes):
            Key = next(self.GenomeIndexer)
            Genome = GenomeType(Key)
            Genome.ConfigureNew(GenomeConfig)
            NewGenomes[Key] = Genome
            self.Ancestors[Key] = tuple()
        return NewGenomes

    @staticmethod
    def ComputeSpawn(AdjustedFitness, PreviousSizes, PopulationSize, MinSpeciesSize):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        AdjustedFitnessSum = sum(AdjustedFitness)
        SpawnAmounts = []
        for AdjustedFitness, PreviousSize in zip(AdjustedFitness, PreviousSizes):
            if AdjustedFitnessSum > 0:
                Size = max(MinSpeciesSize, AdjustedFitness / AdjustedFitnessSum * PopulationSize)
            else:
                Size = MinSpeciesSize
            Difference = (Size - PreviousSize) * 0.5
            Count = int(round(Difference))
            Spawn = PreviousSize
            if abs(Count) > 0:
                Spawn += Count
            elif Difference > 0:
                Spawn += 1
            elif Difference < 0:
                Spawn -= 1
            SpawnAmounts.append(Spawn)
        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        TotalSpawn = sum(SpawnAmounts)
        Normalized = PopulationSize / TotalSpawn
        SpawnAmounts = [max(MinSpeciesSize, int(round(NumberSpawn * Normalized))) for NumberSpawn in SpawnAmounts]
        return SpawnAmounts

    def Reproduce(self, Config, SpeciesSet, PopulationSize, Generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        RemainingSpecies = [Species for SpeciesID, Species in SpeciesSet.Species.items() if not Species.bStagnant]
        if not RemainingSpecies:
            SpeciesSet.Species = {}
            return {} # was []

        #BotReproductionTimer = PerfTimer('BotReproduction')
        OldPopulation = [(GenomeID, Genome) for SpeciesID, SingleSpecies in SpeciesSet.Species.items() for GenomeID, Genome in SingleSpecies.Members.items()]
        AdjustedFitnesses = [SingleSpecies.AdjustedFitness for SingleSpecies in RemainingSpecies if SingleSpecies.AdjustedFitness is not None]
        AverageAdjustedFitness = np.mean(AdjustedFitnesses) # type: float
        self.Reporters.Info("Average adjusted fitness: {:.3f}".format(AverageAdjustedFitness))
        # Compute the number of new members for each species in the new generation.
        PreviousSizes = [len(SingleSpecies.Members) for SingleSpecies in RemainingSpecies]
        MinSpeciesSize = self.ReproductionConfig.MinSpeciesSize
        MinSpeciesSize = max(MinSpeciesSize, self.ReproductionConfig.Elitism)
        SpawnAmounts = self.ComputeSpawn(AdjustedFitnesses, PreviousSizes, PopulationSize, MinSpeciesSize)
        NewPopulation = {}
        SpeciesSet.Species = {}
        for SpawnQuantity, SingleSpecies in zip(SpawnAmounts, RemainingSpecies):
            # If elitism is enabled, each species always at least gets to retain its elites.
            SpawnQuantity = max(SpawnQuantity, self.ReproductionConfig.Elitism)
            assert SpawnQuantity > 0
            # The species has at least one member for the next generation, so retain it.
            OldMembers = list(SingleSpecies.Members.items())
            SingleSpecies.Members = {}
            SpeciesSet.Species[SingleSpecies.Key] = SingleSpecies
            # Sort members in order of descending fitness.
            OldMembers.sort(reverse=True, key=lambda Genome: Genome[1].Fitness)
            # Transfer elites to new generation.
            if self.ReproductionConfig.Elitism > 0:
                for GenomeID, Genome in OldMembers[:self.ReproductionConfig.Elitism]:
                    NewPopulation[GenomeID] = Genome
                    SpawnQuantity -= 1
            if SpawnQuantity <= 0:
                continue
            # Only use the survival threshold fraction to use as parents for the next generation.
            ReproductionCutoffIndex = int(math.ceil(self.ReproductionConfig.SurvivalThreshold * len(OldMembers)))
            # Use at least two parents no matter what the threshold fraction result is.
            ReproductionCutoffIndex = max(ReproductionCutoffIndex, 2)
            OldMembers = OldMembers[:ReproductionCutoffIndex]
            # Randomly choose parents and produce the number of offspring allotted to the species.
            while SpawnQuantity > 0:
                SpawnQuantity -= 1
                ChildGenome = None
                ChildGenomeID = next(self.GenomeIndexer)

                if (random() <= self.ReproductionConfig.CrossoverChance): 
                    ParentID1, Parent1 = choice(OldMembers)
                    ParentID2, Parent2 = choice(OldPopulation) if (random() <= self.ReproductionConfig.CrossSpeciesCrossoverChance) else choice(OldMembers)
                    ChildGenome = Config.GenomeType(ChildGenomeID)
                    ChildGenome.ConfigureCrossover(Parent1, Parent2, Config.GenomeConfig)
                    self.Ancestors[ChildGenomeID] = (ParentID1, ParentID2)
                else:
                    CloneParentID, CloneParent = choice(OldMembers)
                    ChildGenome = Config.GenomeType(ChildGenomeID)
                    ChildGenome.ConfigureClone(CloneParent, Config.GenomeConfig)
                    self.Ancestors[ChildGenomeID] = (CloneParentID, CloneParentID)
                
                if (random() <= self.ReproductionConfig.MutateChance):
                        ChildGenome.Mutate(Config.GenomeConfig)
                NewPopulation[ChildGenomeID] = ChildGenome
        #BotReproductionTimer.Stop()
        return NewPopulation