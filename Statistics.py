"""
Gathers (via the reporting interface) and provides (to callers and/or a file)
the most-fit genomes and information on genome/species fitness and species sizes.
"""
import copy
import csv
import numpy as np
from NEAT.Reports import BaseReporter

# TODO: Make a version of this reporter that doesn't continually increase memory usage.
# (Maybe periodically write blocks of history to disk, or log stats in a database?)
class StatisticsReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """
    def __init__(self):
        BaseReporter.__init__(self)
        self.MostFitGenomes = []
        self.GenerationStatistics = []

    def PostEvaluate(self, Config, Population, Species, BestGenome):
        self.MostFitGenomes.append(copy.deepcopy(BestGenome))
        # Store the fitnesses of the members of each currently active species.
        SpeciesStats = {}
        for SpeciesID, SingleSpecies in Species.Species.items():
            SpeciesStats[SpeciesID] = dict((GenomeKey, Genome.Fitness) for GenomeKey, Genome in SingleSpecies.Members.items())
        self.GenerationStatistics.append(SpeciesStats)

    def GetFitnessStat(self, Function):
        Stat = []
        for Stats in self.GenerationStatistics:
            Scores = []
            for SpeciesStats in Stats.values():
                Scores.extend(SpeciesStats.values())
            Stat.append(Function(Scores))
        return Stat

    def GetFitnessMean(self):
        """Get the per-generation mean fitness."""
        return self.GetFitnessStat(np.mean)

    def GetFitnessStandardDeviation(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.GetFitnessStat(np.std)

    def GetFitnessMedian(self):
        """Get the per-generation median fitness."""
        return self.GetFitnessStat(np.median)

    def BestUniqueGenomes(self, NumberToReturn):
        """Returns the most n fit genomes, with no duplication."""
        BestUnique = {}
        for FitGenome in self.MostFitGenomes:
            BestUnique[FitGenome.Key] = FitGenome
        BestUniqueList = list(BestUnique.values())
        def Key(Genome):
            return Genome.Fitness
        return sorted(BestUniqueList, key=Key, reverse=True)[:NumberToReturn]

    def BestGenomes(self, NumberToReturn):
        """Returns the n most fit genomes ever seen."""
        def Key(Genome):
            return Genome.Fitness
        return sorted(self.MostFitGenomes, key=Key, reverse=True)[:NumberToReturn]

    def BestGenome(self):
        """Returns the most fit genome ever seen."""
        return self.BestGenomes(1)[0]

    def Save(self):
        self.SaveGenomeFitness()
        self.SaveSpeciesCount()
        self.SaveSpeciesFitness()

    def SaveGenomeFitness(self, Delimiter=' ', Filename='FitnessHistory.csv'):
        """ Saves the population's best and average fitness. """
        with open(Filename, 'w') as File:
            Writer = csv.writer(File, delimiter=Delimiter)
            BestFitness = [Genome.Fitness for Genome in self.MostFitGenomes]
            MeanFitness = self.GetFitnessMean()
            for Best, Mean in zip(BestFitness, MeanFitness):
                Writer.writerow([Best, Mean])

    def SaveSpeciesCount(self, Delimiter=' ', Filename='Speciation.csv'):
        """ Log speciation throughout evolution. """
        with open(Filename, 'w') as File:
            Writer = csv.writer(File, delimiter=Delimiter)
            for SpeciesSize in self.GetSpeciesSizes():
                Writer.writerow(SpeciesSize)

    def SaveSpeciesFitness(self, Delimiter=' ', NullValue='NA', Filename='SpeciesFitness.csv'):
        """ Log species' average fitness throughout evolution. """
        with open(filename, 'w') as File:
            Writer = csv.writer(File, delimiter=delimiter)
            for SpeciesFitness in self.GetSpeciesFitness(NullValue):
                Writer.writerow(SpeciesFitness)

    def GetSpeciesSizes(self):
        AllSpecies = set()
        for GenerationStats in self.GenerationStatistics:
            AllSpecies = AllSpecies.union(GenerationStats.keys())
        MaxSpecies = max(AllSpecies)
        SpeciesCounts = []
        for GenerationStats in self.GenerationStatistics:
            Species = [len(GenerationStats.get(SpeciesID, [])) for SpeciesID in range(1, MaxSpecies + 1)]
            SpeciesCounts.append(Species)
        return SpeciesCounts

    def GetSpeciesFitness(self, NullValue=''):
        AllSpecies = set()
        for GenerationStats in self.GenerationStatistics:
            AllSpecies = AllSpecies.union(GenerationStats.keys())
        MaxSpecies = max(AllSpecies)
        SpeciesFitness = []
        for GenerationStats in self.GenerationStatistics:
            MemberFitnesses = [GenerationStats.get(sid, []) for sid in range(1, max_species + 1)]
            Fitness = []
            for MemberFitness in MemberFitnesses:
                if MemberFitness:
                    Fitness.append(mean(MemberFitness))
                else:
                    Fitness.append(NullValue)
            SpeciesFitness.append(Fitness)
        return SpeciesFitness
