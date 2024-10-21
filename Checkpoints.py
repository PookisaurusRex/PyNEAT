"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random
import time
import FileIO

import pickle

from NEAT.Population import Population
from NEAT.Reports import BaseReporter


class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self, GenerationInterval=100, TimeIntervalSeconds=300, FilenamePrefix='NEAT-Checkpoint-'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.
        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.GenerationInterval = GenerationInterval
        self.TimeIntervalSeconds = TimeIntervalSeconds
        self.FilenamePrefix = FilenamePrefix
        self.CurrentGeneration = None
        self.LastGenerationCheckpoint = -1
        self.LastTimeCheckpoint = time.time()

    def StartGeneration(self, Generation):
        self.CurrentGeneration = Generation

    def EndGeneration(self, Config, Population, SpeciesSet):
        bNeedToCheckpoint = False
        if self.TimeIntervalSeconds is not None:
            DeltaTime = time.time() - self.LastTimeCheckpoint
            if DeltaTime >= self.TimeIntervalSeconds:
                bNeedToCheckpoint = True
        if (bNeedToCheckpoint is False) and (self.GenerationInterval is not None):
            DeltaGenerations = self.CurrentGeneration - self.LastGenerationCheckpoint
            if DeltaGenerations >= self.GenerationInterval:
                bNeedToCheckpoint = True
        if bNeedToCheckpoint:
            self.SaveCheckpoint(Config, Population, SpeciesSet, self.CurrentGeneration)
            self.LastGenerationCheckpoint = self.CurrentGeneration
            self.LastTimeCheckpoint = time.time()

    def SaveCheckpoint(self, Config, Population, SpeciesSet, Generation):
        """ Save the current simulation state. """
        Filename = '{0}{1}'.format(self.FilenamePrefix, Generation)
        print("Saving checkpoint to {0}".format(Filename))
        CheckpointsPath = FileIO.JoinPaths(FileIO.GetParentPath(), 'Checkpoints')
        if not FileIO.DoesPathExist(CheckpointsPath):
            FileIO.MakePath(CheckpointsPath)
        Filename = FileIO.JoinPaths(CheckpointsPath, Filename)
        with gzip.open(Filename, 'w', compresslevel=5) as File:
            Data = (Generation, Config, Population, SpeciesSet, random.getstate())
            pickle.dump(Data, File, protocol=pickle.HIGHEST_PROTOCOL)
        PopulationMembers = list(Population.values())
        PopulationMembers.sort(reverse=True, key=lambda Genome: Genome.Fitness)
        BestGenome = PopulationMembers[0]
        FileIO.Pickle(BestGenome, Filename+'-Best')

    @staticmethod
    def RestoreCheckpoint(Filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(Filename) as File:
            Generation, Config, Population, SpeciesSet, RandomState = pickle.load(File)
            random.setstate(RandomState)
            return Population(Config, (Population, SpeciesSet, Generation))