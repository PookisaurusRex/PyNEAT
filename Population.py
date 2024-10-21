"""Implements the core evolution algorithm."""
from __future__ import print_function

import numpy as np

from NEAT.Reports import ReporterSet
from NEAT.MathUtils import mean

import types
import NEAT.AdjustFitness
from itertools import count

from Benchmarking import PerfTimer


class CompleteExtinctionException(Exception):
    pass

class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """
    def __init__(self, Config, InitialPopulation=None, InitialSpeciesSet=None, InitialGeneration=None):
        self.Reporters = ReporterSet()
        self.Config = Config
        self.Stagnation = Config.StagnationType(Config.StagnationConfig, self.Reporters)
        self.Reproduction = Config.ReproductionType(Config.ReproductionConfig, self.Reporters, self.Stagnation)
        if Config.FitnessGroup == "population":
            self.FitnessGroup = "population"
        elif Config.FitnessGroup == "species":
            self.FitnessGroup = "species"
        else:
            raise RuntimeError("Unexpected FitnessGroup: {0!r}".format(Config.FitnessGroup))
        if Config.FitnessCriterion == 'max':
            self.FitnessCriterion = np.nanmax
        elif Config.FitnessCriterion == 'min':
            self.FitnessCriterion = np.nanmin
        elif Config.FitnessCriterion == 'mean':
            self.FitnessCriterion = np.nanmean
        elif Config.FitnessCriterion == 'median':
            self.FitnessCriterion = np.nanmedian
        elif not Config.bDisableFitnessTermination:
            raise RuntimeError("Unexpected FitnessCriterion: {0!r}".format(Config.FitnessCriterion))
        self.Population = self.Reproduction.CreateNew(Config.GenomeType, Config.GenomeConfig, Config.PopulationSize) if InitialPopulation is None else InitialPopulation
        self.SpeciesSet = Config.SpeciesSetType(Config.SpeciesSetConfig, self.Reporters) if InitialSpeciesSet is None else InitialSpeciesSet
        if InitialPopulation is not None:
            self.Reproduction.GenomeIndexer = count(len(InitialPopulation.items()))

        self.Generation = 0 if InitialGeneration is None else InitialGeneration
        self.SpeciesSet.Speciate(Config, self.Population, self.Generation)
        self.BestOfAllTime = None

    def AddReporter(self, Reporter):
        self.Reporters.Add(Reporter)

    def RemoveReporter(self, Reporter):
        self.Reporters.Remove(Reporter)

    def Run(self, FitnessFunction, MaxNumGenerations=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.
        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.
        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.
        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.
        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        if self.Config.bDisableFitnessTermination and (MaxNumGenerations is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")
        CurrentGeneration = 0
        while MaxNumGenerations is None or CurrentGeneration < MaxNumGenerations:
            CurrentGeneration += 1
            self.Reporters.StartGeneration(self.Generation)
            
            # Evaluate all genomes using the user-provided function.
            FitnessFunction(list(self.Population.items()), self.Config)
            
            # Gather and report statistics.
            BestOfGeneration = None
            for Genome in self.Population.values():
                if Genome.Fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(Genome.Key))
                if BestOfGeneration is None or Genome.Fitness > BestOfGeneration.Fitness:
                    BestOfGeneration = Genome
            self.Reporters.PostEvaluate(self.Config, self.Population, self.SpeciesSet, BestOfGeneration)
            
            # Track the best genome ever seen.
            if self.BestOfAllTime is None or BestOfGeneration.Fitness > self.BestOfAllTime.Fitness:
                self.BestOfAllTime = BestOfGeneration

            bFoundSolution = False
            # Population-based FitnessCriterion termination should be handled prior to reproduction to avoid Genomes with 
            if (not self.Config.bDisableFitnessTermination) and (self.Config.MinimumGeneration <= 0 or CurrentGeneration > self.Config.MinimumGeneration):
                if (self.FitnessGroup == "population"):
                    # End if the fitness threshold is reached.
                    EligibleFitnesses = [Genome.Fitness for Genome in self.Population.values() if (Genome.Fitness is not None and Genome.Fitness > 0.0)]
                    print("Population-wide FitnessCriterion")
                elif (self.FitnessGroup == "species") :
                    #if (self.SpeciesSet.GetSpecies(self.BestOfAllTime.Key) is not None):
                    #    print("Species-wide FitnessCriterion - BestOfAllTime")
                    #    EligibleFitnesses = [Genome.Fitness for Genome in self.SpeciesSet.GetSpecies(self.BestOfAllTime.Key).Members.values() if (Genome.Fitness is not None and Genome.Fitness > 0.0)]
                    if (self.SpeciesSet.GetSpecies(BestOfGeneration.Key) is not None):
                        print("Species-wide FitnessCriterion - BestOfGeneration")
                        EligibleFitnesses = [Genome.Fitness for Genome in self.SpeciesSet.GetSpecies(BestOfGeneration.Key).Members.values() if (Genome.Fitness is not None and Genome.Fitness > 0.0)]
                    else:
                        print("Species-wide FitnessCriterion - No eligible species!!")
                        EligibleFitnesses = []
                if len(EligibleFitnesses) > 0:
                    ClosestFitnessToThreshold = self.FitnessCriterion(EligibleFitnesses)
                    print("ClosestFitnessToThreshold: {}".format(ClosestFitnessToThreshold))
                    if ClosestFitnessToThreshold >= self.Config.FitnessThreshold:
                        #self.Reporters.FoundSolution(self.Config, self.Generation, BestOfGeneration)
                        self.Reporters.FoundSolution(self.Config, self.Generation, self.BestOfAllTime)
                        bFoundSolution = True

            self.CalculateAdjustFitnesses()
            self.Reporters.EndGeneration(self.Config, self.Population, self.SpeciesSet)

            if bFoundSolution:
                break

            # Create the next generation from the current generation.
            self.Population = self.Reproduction.Reproduce(self.Config, self.SpeciesSet, self.Config.PopulationSize, self.Generation)

            # Check for complete extinction.
            if not self.SpeciesSet.Species:
                self.Reporters.CompleteExtinction()
                # If requested by the user, create a completely new population, otherwise raise an exception.
                if self.Config.bResetOnExtinction:
                    self.Population = self.Reproduction.CreateNew(self.Config.GenomeType, self.Config.GenomeConfig, self.Config.PopulationSize)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.SpeciesSet.Speciate(self.Config, self.Population, self.Generation)

            self.Generation = CurrentGeneration
        if self.Config.bDisableFitnessTermination:
            self.Reporters.FoundSolution(self.Config, self.Generation, self.BestGenome)
        return self.BestOfAllTime


Population.CalculateAdjustFitnesses = NEAT.AdjustFitness.CalculateAdjustFitnesses
