"""
Implementation of reporter classes, which are triggered on particular events. Reporters
are generally intended to  provide information to the user, store checkpoints, etc.
"""
from __future__ import division, print_function

import time
import numpy as np
from NEAT.MathUtils import StatisticalFunctions


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    def __init__(self):
        self.Reporters = []

    def Add(self, Reporter):
        self.Reporters.append(Reporter)

    def Remove(self, Reporter):
        self.Reporters.remove(Reporter)

    def StartGeneration(self, Generation):
        for Reporter in self.Reporters:
            Reporter.StartGeneration(Generation)

    def EndGeneration(self, Config, Population, SpeciesSet):
        for Reporter in self.Reporters:
            Reporter.EndGeneration(Config, Population, SpeciesSet)

    def PostEvaluate(self, Config, Population, Species, BestGenome):
        for Reporter in self.Reporters:
            Reporter.PostEvaluate(Config, Population, Species, BestGenome)

    def PostReproduction(self, Config, Population, Species):
        for Reporter in self.Reporters:
            Reporter.PostReproduction(Config, Population, Species)

    def CompleteExtinction(self):
        for Reporter in self.Reporters:
            Reporter.CompleteExtinction()

    def FoundSolution(self, Config, Generation, BestGenome):
        for Reporter in self.Reporters:
            Reporter.FoundSolution(Config, Generation, BestGenome)

    def SpeciesStagnant(self, SpeciesID, Species):
        for Reporter in self.Reporters:
            Reporter.SpeciesStagnant(SpeciesID, Species)

    def Info(self, Message):
        for Reporter in self.Reporters:
            Reporter.Info(Message)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def StartGeneration(self, Generation):
        pass

    def EndGeneration(self, Config, Population, SpeciesSet):
        pass

    def PostEvaluate(self, Config, Population, Species, BestGenome):
        pass

    def PostReproduction(self, Config, Population, Species):
        pass

    def CompleteExtinction(self):
        pass

    def FoundSolution(self, Config, Generation, BestGenome):
        pass

    def SpeciesStagnant(self, SpeciesID, Species):
        pass

    def Info(self, Message):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, bShowSpeciesDetails):
        self.bShowSpeciesDetails = bShowSpeciesDetails
        self.Generation = None
        self.GenerationStartTime = None
        self.GenerationTimes = []
        self.NumExtinctions = 0

    def StartGeneration(self, Generation):
        self.Generation = Generation
        print('\n ****** Running generation {0} ****** \n'.format(Generation))
        self.GenerationStartTime = time.time()

    def EndGeneration(self, Config, Population, SpeciesSet):
        ng = len(Population)
        ns = len(SpeciesSet.Species)
        if self.bShowSpeciesDetails:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            print("   ID   age  size  minfitness  maxfitness  fitness  adj fit  stag")
            print("  ====  ===  ====  ==========  ==========  =======  =======  ====")
            for SpeciesID in sorted(SpeciesSet.Species):
                SingleSpecies = SpeciesSet.Species[SpeciesID]
                Age = self.Generation - SingleSpecies.Created
                NumMembers = len(SingleSpecies.Members)
                SpeciesFitnesses = np.array(SingleSpecies.GetFitnesses())
                SpeciesLength = len(SpeciesFitnesses)
                bInvalidFitness = (SpeciesLength is None or SpeciesLength is 0 or np.all(np.isnan(SpeciesFitnesses)))
                MinFitness = "--" if bInvalidFitness else "{:.1f}".format(np.min(SpeciesFitnesses))
                MaxFitness = "--" if bInvalidFitness else "{:.1f}".format(np.max(SpeciesFitnesses))
                Fitness = "--" if bInvalidFitness else "{:.1f}".format(np.mean(SpeciesFitnesses))
                AdjustedFitness = "--" if SingleSpecies.AdjustedFitness is None else "{:.4f}".format(SingleSpecies.AdjustedFitness)
                Stagnation = self.Generation - SingleSpecies.LastImproved
                print("  {: >4}  {: >3}  {: >4}  {: >10}  {: >10}  {: >7}  {: >7}  {: >4}".format(SpeciesID, Age, NumMembers, MinFitness, MaxFitness, Fitness, AdjustedFitness, Stagnation))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))
        Elapsed = time.time() - self.GenerationStartTime
        self.GenerationTimes.append(Elapsed)
        self.GenerationTimes = self.GenerationTimes[-10:]
        Average = sum(self.GenerationTimes) / len(self.GenerationTimes)
        print('Total extinctions: {0:d}'.format(self.NumExtinctions))
        if len(self.GenerationTimes) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(Elapsed, Average))
        else:
            print("Generation time: {0:.3f} sec".format(Elapsed))

    def PostEvaluate(self, Config, Population, Species, BestGenome):
        # pylint: disable=no-self-use
        FitnessValues = [Genome.Fitness for Genome in Population.values()]
        FitnessMean = np.mean(FitnessValues)
        FitnessMedian = np.median(FitnessValues)
        FitnessStandardDev = np.std(FitnessValues)
        BestSpeciesID = Species.GetSpeciesID(BestGenome.Key)
        print('Population\'s average fitness: {:3.5f} median: {:3.5f} stdev: {:3.5f}'.format(FitnessMean, FitnessMedian, FitnessStandardDev))
        print('Best fitness: {0:3.5f} - size: {1!r} - Species {2} - id {3}'.format(BestGenome.Fitness, BestGenome.Size(), BestSpeciesID, BestGenome.Key))
        #print(FitnessValues)
        print('------------------------------------------')

    def CompleteExtinction(self):
        self.NumExtinctions += 1
        print('All species extinct.')

    def FoundSolution(self, Config, Generation, BestGenome):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(self.Generation, BestGenome.Size()))

    def SpeciesStagnant(self, SpeciesID, Species):
        if self.bShowSpeciesDetails:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(SpeciesID, len(Species.Members)))

    def Info(self, Message):
        print(Message)