"""Divides the population into species based on genomic distances."""
from itertools import count

import numpy as np
from NEAT.Config import ConfigParameter, DefaultClassConfig

class Species(object):
    def __init__(self, Key, Generation):
        self.Key = Key
        self.Created = Generation
        self.LastImproved = Generation
        self.Representative = None
        self.Members = {}
        self.Fitness = None
        self.AdjustedFitness = None
        self.FitnessHistory = []
        self.bStagnant = False

    def Update(self, Representative, Members):
        self.Representative = Representative
        self.Members = Members

    def GetFitnesses(self):
        return [Genome.Fitness for Genome in self.Members.values()]
        #return [Genome.Fitness for Genome in self.Members.values() if Genome.Fitness is not None]


class GenomeDistanceCache(object):
    def __init__(self, Config):
        self.Distances = {}
        self.Config = Config
        self.Hits = 0
        self.Misses = 0

    def __call__(self, Genome1, Genome2):
        GenomeKey1 = Genome1.Key
        GenomeKey2 = Genome2.Key
        Distance = self.Distances.get((GenomeKey1, GenomeKey2))
        if Distance is None:
            # Distance is not already computed.
            Distance = Genome1.Distance(Genome2, self.Config)
            self.Distances[GenomeKey1, GenomeKey2] = Distance
            self.Distances[GenomeKey2, GenomeKey1] = Distance
            self.Misses += 1
        else:
            self.Hits += 1
        return Distance

    def ClearCache(self):
        self.Distances = {}
        self.Hits = 0
        self.Misses = 0


class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, Config, Reporters):
        # pylint: disable=super-init-not-called
        self.SpeciesSetConfig = Config
        self.Reporters = Reporters
        self.Indexer = count(1)
        self.Species = {}
        self.GenomeToSpecies = {}

    @classmethod
    def ParseConfig(cls, ParameterDictionary):
        return DefaultClassConfig(ParameterDictionary, [ConfigParameter('CompatibilityThreshold', float),
                                                        ConfigParameter('bRepresentedByBestFit', bool, True)])

    def Speciate(self, Config, Population, Generation):
        """
        Place genomes into species by genetic similarity.
        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(Population, dict)
        CompatibilityThreshold = self.SpeciesSetConfig.CompatibilityThreshold
        # Find the best representatives for each existing species.
        Unspeciated = set(Population)
        Distances = GenomeDistanceCache(Config.GenomeConfig)
        NewRepresentatives = {}
        NewMembers = {}
        for SpeciesID, SingleSpecies in self.Species.items():
            Candidates = []
            for GenomeID in Unspeciated:
                Genome = Population[GenomeID]
                Distance = Distances(SingleSpecies.Representative, Genome)
                Candidates.append((Distance, Genome))
            # The new representative is the genome closest to the current representative.
            if Config.SpeciesSetConfig.bRepresentedByBestFit:
                ValidCandidates = [CandidateGenome for Distance, CandidateGenome in Candidates if CandidateGenome.Fitness is not None]
                if ValidCandidates:
                    NewRepresentative = max(ValidCandidates, key=lambda Genome: Genome.Fitness if Genome.Fitness is not None else 0.0)
                else:
                    SmallestDistance, NewRepresentative = min(Candidates, key=lambda Genome: Genome[0])
            else:
                SmallestDistance, NewRepresentative = min(Candidates, key=lambda Genome: Genome[0])
            NewRepresentativeID = NewRepresentative.Key
            NewRepresentatives[SpeciesID] = NewRepresentativeID
            NewMembers[SpeciesID] = [NewRepresentativeID]
            Unspeciated.remove(NewRepresentativeID)
        # Partition population into species based on genetic similarity.
        while Unspeciated:
            GenomeID = Unspeciated.pop()
            Genome = Population[GenomeID]
            # Find the species with the most similar representative.
            Candidates = []
            for SpeciesID, RepresentativeID in NewRepresentatives.items():
                Representative = Population[RepresentativeID]
                Distance = Distances(Representative, Genome)
                if Distance < CompatibilityThreshold:
                    Candidates.append((Distance, SpeciesID))
            if Candidates:
                SmallestDistance, SpeciesID = min(Candidates, key=lambda x: x[0])
                NewMembers[SpeciesID].append(GenomeID)
            else:
                # No species is similar enough, create a new species, using this genome as its representative.
                SpeciesID = next(self.Indexer)
                NewRepresentatives[SpeciesID] = GenomeID
                NewMembers[SpeciesID] = [GenomeID]
        # Update species collection based on new speciation.
        self.GenomeToSpecies = {}
        for SpeciesID, RepresentativeID in NewRepresentatives.items():
            SingleSpecies = self.Species.get(SpeciesID)
            if SingleSpecies is None:
                SingleSpecies = Species(SpeciesID, Generation)
                self.Species[SpeciesID] = SingleSpecies
            Members = NewMembers[SpeciesID]
            for GenomeID in Members:
                self.GenomeToSpecies[GenomeID] = SpeciesID
            MemberDictionary = dict((GenomeID, Population[GenomeID]) for GenomeID in Members)
            SingleSpecies.Update(Population[RepresentativeID], MemberDictionary)
        DistanceValues = Distances.Distances.values()
        GenomeDistanceMean = np.mean(list(DistanceValues))
        GenomeDistanceStandardDev = np.std(list(DistanceValues))
        self.Reporters.Info('Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(GenomeDistanceMean, GenomeDistanceStandardDev))
        bVerbose = False
        if bVerbose:
            for GenomeID in set(Population):
                print('Genome({}): Species={}'.format(GenomeID, self.GetSpeciesID(GenomeID)))


    def GetSpeciesID(self, IndividualID):
        return self.GenomeToSpecies[IndividualID] if IndividualID in self.GenomeToSpecies else None

    def GetSpecies(self, IndividualID):
        SpeciesID = self.GetSpeciesID(IndividualID)
        return self.Species[SpeciesID] if SpeciesID in self.Species else None