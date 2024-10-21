import tensorflow as tf

class BaseGenome(object):
    def __str__(self) -> str:
        raise NotImplementedError("Should implement __str__()")

    def Visualize(self, bDisplayView: bool, RenderFilePath: str):
        """
        Display rendered genome or save rendered genome to specified path or do both
        :param bDisplayView: flag if rendered genome should be displayed
        :param RenderFilePath: File path string, specifying where the genome render should be saved
        """
        raise NotImplementedError("Should implement visualize()")

    def GetModel(self) -> tf.keras.Model:
        """
        :return: Tensorflow model phenotype translation of the genome genotype
        """
        raise NotImplementedError("Should implement GetModel()")

    def GetGenotype(self) -> dict:
        """
        :return: genome genotype dict with the keys being the gene-ids and the values being the genes
        """
        raise NotImplementedError("Should implement GetGenotype()")

    def GetID(self) -> int:
        raise NotImplementedError("Should implement GetID()")

    def GetFitness(self) -> float:
        raise NotImplementedError("Should implement GetFitness()")

    def SetFitness(self, Fitness: float):
        raise NotImplementedError("Should implement SetFitness()")

class BaseEncoding(object):
    def CreateGenome(self, Genotype) -> (int, BaseGenome):
        """
        Create genome based on the supplied genotype, with continuous genome-id for each newly created genome
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :return: tuple of continuous genome-id and created genome
        """
        raise NotImplementedError("Should implement CreateGenome()")

class BaseNeuroevolutionAlgorithm(object):
    def InitializePopulation(self, Population, InitialPopulationSize: int, InputShape: tuple, NumOutput: int):
        """
        Initialize the population according the algorithms specifications to the size 'InitialPopulationSize'. The phenotypes
        of the genomes should accept inputs of the shape 'InputShape' and have 'NumOutput' nodes in their output layer
        :param Population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param InitialPopulationSize: int, amount of genomes to be initialized and added to the population
        :param input_shape: tuple, shape of the input vector for the NN model to be created
        :param num_output: int, number of nodes in the output layer of the NN model to be created
        """
        raise NotImplementedError("Should implement InitializePopulation()")

    def EvolvePopulation(self, Population, bFixedPopulationSize: bool):
        """
        Evolve the population according to the algorithms specifications.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param bFixedPopulationSize: bool flag, indicating if the size of the population size is allowed to change
        """
        raise NotImplementedError("Should implement EvolvePopulation()")

    def EvaluatePopulation(self, Population, GenomeEvaluationFunction):
        """
        Evaluate the population according to the GenomeEvaluationFunction for each genome in the population.
        :param Population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param GenomeEvaluationFunction: callable method that takes a genome as input and returns the fitness score
        """
        raise NotImplementedError("Should implement EvaluatePopulation()")

    def SummarizePopulation(self, Population):
        """
        Output a summary of the population, giving a concise overview of the status of the population.
        :param Population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        """
        raise NotImplementedError("Should implement SummarizePopulation()")