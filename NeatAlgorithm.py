import numpy as np
import tensorflow as tf
from absl import logging
from copy import deepcopy
from random import choice, random, randint, shuffle
from NEAT.BaseAlgorithm import BaseEncoding, BaseGenome, BaseNeuroevolutionAlgorithm

class DirectEncodingConnection(object):
    def __init__(self, GeneID, NodeIn, NodeOut, Weight):
        self.GeneID = GeneID
        self.NodeIn = NodeIn
        self.NodeOut = NodeOut
        self.Weight = Weight
        self.bEnabled = True

    def SetEnabled(self, bEnabled: bool):
        self.bEnabled = bEnabled

    def serialize(self) -> dict:
        return { 'GeneType': 'DirectEncodingConnection', 'NodeIn': self.NodeIn, 'NodeOut': self.NodeOut, 'Weight': str(self.Weight), 'bEnabled': self.bEnabled }

    def Deserialize(SerializedEncoding: dict) -> DirectEncodingConnection:
        DeserializedEncoding = DirectEncodingConnection(SerializedEncoding["GeneID"], SerializedEncoding["NodeIn"], SerializedEncoding["NodeOut"], float(SerializedEncoding["Weight"]))
        return DeserializedEncoding

class DirectEncodingNode(object):
    def __init__(self, GeneID, Node, Bias, Activation):
        self.GeneID = GeneID
        self.Node = Node
        self.Bias = Bias
        self.Activation = Activation

    def serialize(self) -> dict:
        return { 'GeneType': 'DirectEncodingNode', 'GeneID': self.GeneID, 'Node': self.Node, 'Bias': str(self.Bias), 'Activation': tf.keras.activations.serialize(self.Activation) }

    def Deserialize(SerializedEncoding: dict) -> DirectEncodingNode:
        return DirectEncodingNode(SerializedEncoding["GeneID"], SerializedEncoding["Node"], SerializedEncoding["Bias"], SerializedEncoding["Activation"])

class DirectEncodingGenome(BaseGenome):
    """
    Implementation of a DirectEncoding genome as employed by NE-algorithms like NEAT. DirectEncoding genomes have each
    connection, connection-weight, node, node-bias, etc of the corresponding Tensorflow model explicitely encoded in
    their genotype, which is made up of DirectEncoding genes. Upon creation does the DirectEncoding genome immediately
    create the phenotype Tensorflow model based on the genotype.
    """
    def __init__(self, GenomeID, Genotype, bTrainable: bool, dtype, bRunEager):
        """
        Set ID and genotype of genome to the supplied parameters, set the default fitness value of the genome to 0 and
        create the Tensorflow model phenotype using the supplied Genotype, bTrainable, dtype and run_eagerly parameters
        and save it as the model.
        """
        self.GenomeID = GenomeID
        self.Genotype = Genotype
        self.Fitness = 0
        self.Model = DirectEncodingModelTrainable(Genotype, dtype, bRunEager) if bTrainable else DirectEncodingModelNontrainable(Genotype, dtype)

    def __str__(self) -> str:
        return "DirectEncodingGenome || ID: {:>4} || Fitness: {:>8} || Gene Count: {:>4}".format(self.GenomeID, self.fitness, len(self.Genotype))

    def Visualize(self, bDisplay=True, Filename=None, RenderDirectoryPath=None):
        """
        Display rendered genome or save rendered genome to specified path or do both
        :param bDisplay: flag if rendered genome should be displayed
        :param filename: string of filename of the visualization render, excluding the extension
        :param render_dir_path: string of directory path, specifying where the genome render should be saved
        """
        visualize_genome(self.GetID(), self.GetGenotype(), self.GetTopologyLevels(), bDisplay, Filename, RenderDirectoryPath)

    def serialize(self) -> dict:
        """
        Shallow serializes genome and returns it as a dict. Serialization is shallow as only genome characteristics are
        serialized that are required to recreate genome from scratch - but no internal states(fitness, dtype, etc).
        :return: dict; dict containing the shallow serialization of the genome
        """
        return {'GenotypeType':'DirectEncodingGenome', 'Genotype':[Gene.serialize() for Gene in self.Genotype.values()]}

    def GetModel(self) -> typing.Union[DirectEncodingModelTrainable, DirectEncodingModelNontrainable]:
        """
        :return: Tensorflow model phenotype translation of the genome genotype
        """
        return self.Model

    def GetGenotype(self) -> dict:
        """
        :return: genome genotype dict with the keys being the gene-ids and the values being the genes
        """
        return self.Genotype

    def GetTopologyLevels(self) -> [set]:
        """
        :return: list of topologically sorted sets of nodes. Each list element contains the set of nodes that have to be
                 precomputed before the next list element set of nodes can be computed.
        """
        return self.Model.TopologyLevels

    def GetGeneIDs(self) -> []:
        """
        :return: list of gene-ids contained in the genome genotype
        """
        return self.Genotype.keys()

    def GetID(self) -> int:
        return self.GenomeID

    def GetFitness(self) -> float:
        return self.Fitness

    def SetFitness(self, Fitness):
        self.Fitness = Fitness

class DirectEncoding(BaseEncoding):
    """
    Factory Wrapper for DirectEncoding genomes, providing unique continuous gene- and genome-ids for created genes and
    genomes as well as genomes created with the supplied parameters for trainable, dtype and run_eagerly
    """
    def __init__(self, bTrainable: bool, dtype=tf.float32, bRunEager: bool=False):
        self.bTrainable = bTrainable
        self.dtype = dtype
        self.bRunEager = bRunEager
        self.LogParameters()
        self.NewestGeneID = 0
        self.NewestGenomeID = 0
        self.GeneToGeneMap = dict()

    def LogParameters(self):
        logging.debug("Direct Encoding parameter: bTrainable = {}".format(self.bTrainable))
        logging.debug("Direct Encoding parameter: bRunEager = {}".format(self.bRunEager))
        logging.debug("Direct Encoding parameter: dtype = {}".format(self.dtype))

    def CreateGeneConnection(self, ConnectionIn, ConnectionOut, ConnectionWeight) -> (int, DirectEncodingConnection):
        """
        Create DirectEncoding connection gene with unique continuous gene-id based on the supplied (ConnectionIn, ConnectionOut)
        tuple. Uniqueness disregards ConnectionWeight, meaning that identical gene_ids with different conn_weights can exist.
        :param ConnectionIn: node (usually int) the connection is originating from
        :param ConnectionOut: node (usually int) the connection is ending in
        :param ConnectionWeight: weight (usually float or np.float) of the connection
        :return: tuple of unique gene-id and created DirectEncoding connection gene
        """
        GeneKey = (ConnectionIn, ConnectionOut)
        if GeneKey in self.GeneToGeneMap:
            GeneID = self.GeneToGeneMap[GeneKey]
        else:
            self.NewestGeneID += 1
            self.GeneToGeneMap[GeneKey] = self.NewestGeneID
            GeneID = self.NewestGeneID

        return GeneID, DirectEncodingConnection(GeneID, ConnectionIn, ConnectionOut, ConnectionWeight)

    def CreateGeneNode(self, Node, Bias, Activation) -> (int, DirectEncodingNode):
        """
        Create DirectEncoding node gene with unique continuous gene-id based on the supplied node. Uniqueness disregards
        bias and activation, meaning that identical gene_ids with different bias and activation can exist.
        :param node: node (usually int) the gene represents
        :param bias: bias weight (usually float or np.float) of the node
        :param activation: Tensorflow activation function of the node
        :return: tuple of unique gene-id and created DirectEncoding node gene
        """
        GeneKey = (Node,)
        if GeneKey in self.GeneToGeneMap:
            GeneKey = self.GeneToGeneMap[GeneKey]
        else:
            self.GeneID += 1
            self.GeneToGeneMap[GeneKey] = self.GeneID
            GeneID = self.GeneID

        return GeneID, DirectEncodingNode(GeneID, Node, Bias, Activation)

    def CreateGenome(self, Genotype) -> (int, DirectEncodingGenome):
        """
        Create DirectEncoding genome with continuous genome-id for each newly created genome
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :return: tuple of continuous genome-id and created DirectEncoding genome
        """
        self.NewestGenomeID += 1
        return self.NewestGenomeID, DirectEncodingGenome(GenomeID=self.NewestGenomeID, Genotype=Genotype, bTrainable=self.bTrainable, dtype=self.dtype, bRunEager=self.bRunEager)

class NeuroEvolutionAugmentingTopologies(BaseNeuroevolutionAlgorithm):
    """
    Implementation of Kenneth O'Stanleys and Risto Miikkulainen's algorithm 'Neuroevolution of Augmenting Topologies'
    (NEAT) [1,2] for the Tensorflow-Neuroevolution framework.
    [1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
    [2] http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf
    """

    def __init__(self, Config, dtype=tf.float32, bRunEager=False):
        self.ReproductionFraction = None
        self.CrossoverProbability = None
        self.MutationWeightsProbability = None
        self.MutationAddConnectionProbability = None
        self.MutationAddNodeProbability = None
        self.MutationWeightsFraction = None
        self.MutationWeightsMean = None
        self.MutationWeightsStandardDeviation = None
        self.GeneExcessDistanceScalar = None
        self.GeneDisjointDistanceScalar = None
        self.GeneMatchingDistanceScalar = None
        self.ActivationHidden = None
        self.ActivationOutput = None
        self.SpeciesElitism = None
        self.SpeciesMaximumStagnation = None
        self.SpeciesClustering = None
        self.ReadConfigParameters(Config)
        self.LogParameters()

        # Ensure that the probabilities for crossover and mutations add up to 100% (one or the other always happens)
        SummedProbabilities = self.CrossoverProbability + self.MutationAddConnectionProbability + self.MutationAddNodeProbability
        self.CrossoverProbability = self.CrossoverProbability / SummedProbabilities
        self.MutationAddConnectionProbability = self.MutationAddConnectionProbability / SummedProbabilities
        self.MutationAddNodeProbability = self.MutationAddNodeProbability / SummedProbabilities
        assert self.CrossoverProbability + self.MutationAddConnectionProbability + self.MutationAddNodeProbability == 1.0

        # Use DirectEncoding with the supplied parameters for NEAT. Set trainable to False as NEAT is training/evolving the weights itself
        self.Encoding = DirectEncoding(bTrainable=False, dtype=dtype, bRunEager=bRunEager)

        # Initialize species containers. Start with species_id_counter set to 1 as the population initialization will
        # assign all newly initialized genomes to species 1, as defined per NEAT.
        self.NewestSpeciesID = 1
        self.SpeciesAssignment = dict()
        self.SpeciesAverageFitnessHistory = dict()

        # Initialize implementation specific dicts, keeping track of added nodes and the adjusted fitness of genomes,
        # required for determining alloted offspring of each species.
        self.NodeCounter = None
        self.AddNodeHistory = dict()
        self.GenomesAdjustedFitness = dict()

    def ReadConfigParameters(self, Config):
        """
        Read the class parameters supplied via the Config file
        :param Config: ConfigParser Object which has processed the supplied configuration
        """
        SectionName = 'NEAT' if Config.has_section('NEAT') else 'NE_ALGORITHM'
        self.ReproductionFraction = Config.getfloat(SectionName, 'ReproductionFraction')
        self.CrossoverProbability = Config.getfloat(SectionName, 'CrossoverProbability')
        self.MutationWeightsProbability = Config.getfloat(SectionName, 'MutationWeightsProbability')
        self.MutationAddConnectionProbability = Config.getfloat(SectionName, 'MutationAddConnectionProbability')
        self.MutationAddNodeProbability = Config.getfloat(SectionName, 'MutationAddNodeProbability')
        self.MutationWeightsFraction = Config.getfloat(SectionName, 'MutationWeightsFraction')
        self.MutationWeightsMean = Config.getfloat(SectionName, 'MutationWeightsMean')
        self.MutationWeightsStandardDeviation = Config.getfloat(SectionName, 'MutationWeightsStandardDeviation')
        self.GeneExcessDistanceScalar = Config.getfloat(SectionName, 'GeneExcessDistanceScalar')
        self.GeneDisjointDistanceScalar = Config.getfloat(SectionName, 'GeneDisjointDistanceScalar')
        self.GeneMatchingDistanceScalar = Config.getfloat(SectionName, 'GeneMatchingDistanceScalar')
        self.SpeciesElitism = Config.getint(SectionName, 'SpeciesElitism')
        self.SpeciesMaximumStagnation = Config.get(SectionName, 'SpeciesMaximumStagnation')
        self.SpeciesClustering = Config.get(SectionName, 'SpeciesClustering')
        self.ActivationHidden = Config.get(SectionName, 'ActivationHidden')
        self.ActivationOutput = Config.get(SectionName, 'ActivationOutput')

        if ',' in self.SpeciesClustering:
            SpeciesClusteringAlg = self.SpeciesClustering[:self.SpeciesClustering.find(',')]
            SpeciesClusteringVal = float(self.SpeciesClustering[self.SpeciesClustering.find(',') + 2:])
            self.SpeciesClustering = (SpeciesClusteringAlg, SpeciesClusteringVal)

        if ',' in self.SpeciesMaximumStagnation:
            MaxStagnationDuration = int(self.SpeciesMaximumStagnation[:self.SpeciesMaximumStagnation.find(',')])
            MaxStagnationPercent = float(self.SpeciesMaximumStagnation[self.SpeciesMaximumStagnation.find(',') + 2:])
            self.SpeciesMaximumStagnation = (MaxStagnationDuration, MaxStagnationPercent)

        self.ActivationHidden = tf.keras.activations.deserialize(self.ActivationHidden)
        self.ActivationOutput = tf.keras.activations.deserialize(self.ActivationOutput)

    def LogParameters(self):
        logging.debug("NEAT algorithm Config: ReproductionFraction = {}".format(self.ReproductionFraction))
        logging.debug("NEAT algorithm Config: CrossoverProbability = {}".format(self.CrossoverProbability))
        logging.debug("NEAT algorithm Config: MutationWeightsProbability = {}".format(self.MutationWeightsProbability))
        logging.debug("NEAT algorithm Config: MutationAddConnectionProbability = {}".format(self.MutationAddConnectionProbability))
        logging.debug("NEAT algorithm Config: MutationAddNodeProbability = {}".format(self.MutationAddNodeProbability))
        logging.debug("NEAT algorithm Config: MutationWeightsFraction = {}".format(self.MutationWeightsFraction))
        logging.debug("NEAT algorithm Config: MutationWeightsMean = {}".format(self.MutationWeightsMean))
        logging.debug("NEAT algorithm Config: MutationWeightsStandardDeviation = {}".format(self.MutationWeightsStandardDeviation))
        logging.debug("NEAT algorithm Config: GeneExcessDistanceScalar = {}".format(self.GeneExcessDistanceScalar))
        logging.debug("NEAT algorithm Config: GeneDisjointDistanceScalar = {}".format(self.GeneDisjointDistanceScalar))
        logging.debug("NEAT algorithm Config: GeneMatchingDistanceScalar = {}".format(self.GeneMatchingDistanceScalar))
        logging.debug("NEAT algorithm Config: SpeciesElitism = {}".format(self.SpeciesElitism))
        logging.debug("NEAT algorithm Config: SpeciesMaximumStagnation = {}".format(self.SpeciesMaximumStagnation))
        logging.debug("NEAT algorithm Config: SpeciesClustering = {}".format(self.SpeciesClustering))
        logging.debug("NEAT algorithm Config: ActivationHidden = {}".format(self.ActivationHidden))
        logging.debug("NEAT algorithm Config: ActivationOutput = {}".format(self.ActivationOutput))

    def InitializePopulation(self, Population, InitialPopulationSize: int, InputShape: tuple, NumOutput: int):
        """
        Initialize the population with DirectEncoding genomes created according to the NEATs specification of minimal,
        fully-connected topologies (not hidden nodes, all inputs are connected to all outputs). The initial connection
        weights are randomized in the sense of them being mutated once (which means the addition of a value from a
        random normal distribution with cfg specified mean and stddev), while the node biases are all initialized to 0.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param initial_pop_size: int, amount of genomes to be initialized and added to the population
        :param input_shape: tuple, shape of the input vector for the NN model to be created
        :param num_output: int, number of nodes in the output layer of the NN model to be created
        """
        InputDimensions = len(InputShape)
        if InputDimensions > 1:
            NumInput = 1;
            for SingleInputDimension in InputShape:
                NumInput *= SingleInputDimension
        elif InputDimensions == 1:
            NumInput = InputShape[0]
        else:
            raise NotImplementedError("InputShape has zero dimensions")

        if NumInput > 0:
            # Create species 1, as population initialization assigns first created genome to this standard species
            self.SpeciesAssignment[self.NewestSpeciesID] = None
            self.SpeciesAverageFitnessHistory[self.NewestSpeciesID] = []

            for PopulationSize in range(InitialPopulationSize):
                Genotype = dict()
                for ConnectionInput in range(1, NumInput + 1):
                    for ConnectionOutput in range(NumInput + 1, NumInput + NumOutput + 1):
                        # Create initial connection weight as random value from normal distribution with mean and stddev
                        # as configured in the cfg, effectively setting the weight to 0 and mutating it once.
                        ConnectionWeight = np.random.normal(loc=self.MutationWeightsMean, scale=self.MutationWeightsStandardDeviation)
                        GeneID, GeneConnection = self.encoding.create_gene_connection(ConnectionInput, ConnectionOutput, ConnectionWeight)
                        Genotype[GeneID] = GeneConnection
                for Node in range(NumInput + 1, NumInput + NumOutput + 1):
                    # As each node created in this initialization is a node of the output layer, assign the output
                    # activation to all nodes.
                    GeneID, GeneNode = self.encoding.create_gene_node(Node, 0, self.ActivationOutput)
                    Genotype[GeneID] = GeneNode

                # Set node counter to initialized nodes
                self.NodeCounter = NumInput + NumOutput

                GenomeID, NewGenome = self.encoding.create_genome(Genotype)
                Population.add_genome(GenomeID, NewGenome)
                if self.SpeciesAssignment[self.NewestSpeciesID] is None:
                    self.SpeciesAssignment[self.NewestSpeciesID] = [GenomeID]
        else:
            raise NotImplementedError("NumInput has a length of zero")

    def EvolvePopulation(self, Population, bPopulationSizeFixed: bool):
        """
        Evolve the population by first removing stagnating species and then creating mutations (crossover, weight-
        mutation, add conn mutation, add node mutation) within the existing species, which in turn completely replace
        the old generation (except for the species champions).
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param bPopulationSizeFixed: bool flag, indicating if the size of the population can be different after the evolution
                               of the current generation is complete
        """
        # Assert that pop size is fixed as NEAT does not operate on dynamic pop sizes
        assert bPopulationSizeFixed

        # Determine order in which to go through species and evaluate if species should be removed or not. Evaluate
        # this stagnation from the least to the most fit species, judged on its avg fitness.
        SortedSpeciesIDs = sorted(self.SpeciesAssignment.keys(), key=lambda x: self.SpeciesAverageFitnessHistory[x][-1])
        MaxStagnationDuration = self.SpeciesMaximumStagnation[0]
        NonStagnationImprov = self.SpeciesMaximumStagnation[1]

        # Remove stagnating species
        for SpeciesID in SortedSpeciesIDs:
            # Break stagnation evaluation if less species present than configured in 'SpeciesElitism' cfg parameter
            if len(self.SpeciesAssignment) <= self.SpeciesElitism:
                break
            # Only consider stagnation evaluation if species existed for at least 'MaxStagnationDuration' generations
            AverageFitnessHistory = self.SpeciesAverageFitnessHistory[SpeciesID]
            if len(AverageFitnessHistory) >= MaxStagnationDuration:
                AverageFitnessOverStagnantDuration = sum(AverageFitnessHistory[-MaxStagnationDuration:]) / MaxStagnationDuration
                NonStagnationFitness = AverageFitnessHistory[-MaxStagnationDuration] * NonStagnationImprov
                if AverageFitnessOverStagnantDuration < NonStagnationFitness:
                    # Species stagnating. Remove it.
                    logging.debug("Removing species {} as stagnating for {} generations...".format(SpeciesID, MaxStagnationDuration))
                    for GenomeID in self.SpeciesAssignment[SpeciesID]:
                        Population.DeleteGenome(GenomeID)
                        del self.GenomesAdjustedFitness[GenomeID]
                    del self.SpeciesAssignment[SpeciesID]
                    del self.SpeciesAverageFitnessHistory[SpeciesID]

        # Predetermine variables required for the creation of new genomes
        MutationWeightsThreshold = self.CrossoverProbability + self.MutationWeightsProbability
        MutationAddConnectionThreshold = MutationWeightsThreshold + self.MutationAddConnectionProbability
        PopulationSize = Population.GetPopulationSize()
        SpeciesAdjustedFitnessAverage = dict()
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            AdjustedFitnessAverage = 0
            for SpeciesGenomeID in SpeciesGenomeIDs:
                AdjustedFitnessAverage += self.GenomesAdjustedFitness[SpeciesGenomeID]
            SpeciesAdjustedFitnessAverage[SpeciesID] = AdjustedFitnessAverage
        TotalAdjustedFitnessAverage = sum(SpeciesAdjustedFitnessAverage.values())

        # Calculate alloted offspring for each species such that size of population stays constant and genome elitism
        # is 1, as specified by NEAT. This is necessary as the NEAT formula for allotted offspring allows to increase
        # the population, though the NEAT algorithm is specified on a fixed population size. To counteract this will the
        # species with the most/least allotted offsprign add/subtract one potential genome from its allotted size.
        AllotedOffspring = dict()
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            SpeciesOffspring = round(SpeciesAdjustedFitnessAverage[SpeciesID] * PopulationSize / TotalAdjustedFitnessAverage) - 1
            if SpeciesOffspring < 0:
                SpeciesOffspring = 0
            AllotedOffspring[SpeciesID] = SpeciesOffspring

        ResultingPopulationSize = sum(AllotedOffspring.values()) + len(self.SpeciesAssignment)
        while ResultingPopulationSize > PopulationSize:
            MostOffspringID = max(AllotedOffspring, key=AllotedOffspring.get)
            AllotedOffspring[MostOffspringID] -= 1
            ResultingPopulationSize = sum(AllotedOffspring.values()) + len(self.SpeciesAssignment)
        while ResultingPopulationSize < PopulationSize:
            LeastOffspringID = min(AllotedOffspring, key=AllotedOffspring.get)
            AllotedOffspring[LeastOffspringID] += 1
            ResultingPopulationSize = sum(AllotedOffspring.values()) + len(self.SpeciesAssignment)

        # Create new genomes through evolution
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            # Determine fraction of population suitable to be a parent according to 'ReproductionFraction' cfg parameter
            ReproductionCutoffID = int(self.ReproductionFraction * len(SpeciesGenomeIDs))
            if ReproductionCutoffID == 0:
                ReproductionCutoffID = 1
            ParentGenomeIDs = SpeciesGenomeIDs[:ReproductionCutoffID]

            for _ in range(AllotedOffspring[SpeciesID]):
                ParentGenome1 = Population.GetGenome(choice(ParentGenomeIDs))

                # Create random value and choose either one of the crossovers/mutations
                EvolutionChoice = random()
                if EvolutionChoice < self.CrossoverProbability:
                    # Out of simplicity currently possible that both parent genomes are the same
                    ParentGenome2 = Population.GetGenome(choice(ParentGenomeIDs))
                    NewGenomeID, NewGenome = self.CreateCrossoverGenome(ParentGenome1, ParentGenome2)
                elif EvolutionChoice < MutationWeightsThreshold:
                    NewGenomeID, NewGenome = self.CreateMutatedWeightsGenome(ParentGenome1)
                elif EvolutionChoice < MutationAddConnectionThreshold:
                    NewGenomeID, NewGenome = self.CreateAddedConnectionGenome(ParentGenome1)
                else:
                    NewGenomeID, NewGenome = self.CreateAddedNodeGenome(ParentGenome1)
                # Add the newly created genome to the population immediately
                Population.AddGenome(NewGenomeID, NewGenome)

            # Delete all parent genomes of species as generations completely replace each other, though keep the species
            # champion as NEAT has genome elitism of 1
            for SpeciesGenomeID in SpeciesGenomeIDs[1:]:
                Population.DeleteGenome(SpeciesGenomeID)
                del self.GenomesAdjustedFitness[SpeciesGenomeID]
            del self.SpeciesAssignment[SpeciesID][1:]

    def CreateCrossoverGenome(self, ParentGenome1, ParentGenome2) -> (int, DirectEncodingGenome):
        """
        Create a crossed over genome according to NEAT crossover illustration (since written specification in
        O Stanley's PhD thesis contradictory) by joining all disjoint and excess genes from both parents and choosing
        the parent gene randomly from either parent in case both parents possess the gene. Return that genome.
        :param ParentGenome1: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :param ParentGenome2: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        Genotype1 = ParentGenome1.GetGenotype()
        Genotype2 = ParentGenome2.GetGenotype()
        ExistingGenes = set(Genotype1).union(set(Genotype2))
        NewGenotype = dict()
        for GeneID in ExistingGenes:
            # If matching genes of both genotypes
            if GeneID in Genotype1 and GeneID in Genotype2:
                # Choose randomly from which parent the gene will be carried over
                if randint(0, 1):
                    NewGenotype[GeneID] = deepcopy(Genotype1[GeneID])
                else:
                    NewGenotype[GeneID] = deepcopy(Genotype2[GeneID])
            # If gene a excess or disjoint gene from genotype 1
            elif GeneID in Genotype1:
                NewGenotype[GeneID] = deepcopy(Genotype1[GeneID])
            # If gene a excess or disjoint gene from genotype 2
            else:
                NewGenotype[GeneID] = deepcopy(Genotype2[GeneID])
        return self.Encoding.CreateGenome(NewGenotype)

    def CreateMutatedWeightsGenome(self, ParentGenome) -> (int, DirectEncodingGenome):
        """
        Create a mutated weights genome according to NEAT by adding to each chosen gene's ConnectionWeight or bias a random
        value from a normal distribution with cfg specified mean and stddev. Only x percent (as specified via cfg
        parameter 'MutationWeightsFraction') of all gene's weights are actually mutated, to allow for a more fine-
        grained evolution. Return that genome.
        :param ParentGenome: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        NewGenotype = deepcopy(ParentGenome.GetGenotype())
        GeneIDs = tuple(NewGenotype)
        for _ in range(int(len(NewGenotype) * self.MutationWeightsFraction)):
            # Choose random gene to mutate
            MutatedGeneID = choice(GeneIDs)
            # Create weight to mutate with (as identical for both ConnectionWeight and bias)
            MutationWeight = np.random.normal(loc=self.MutationWeightsMean, scale=self.MutationWeightsStandardDeviation)
            # Identify type of gene and mutate its weight by going with a pythonic 'try and fail safely' approach
            try:
                NewGenotype[MutatedGeneID].ConnectionWeight += MutationWeight
            except AttributeError:
                NewGenotype[MutatedGeneID].Bias += MutationWeight
        return self.Encoding.CreateGenome(NewGenotype)

    def CreateAddedConnectionGenome(self, ParentGenome) -> (int, DirectEncodingGenome):
        """
        Create a added conn genome according to NEAT by randomly connecting two previously unconnected nodes. Return
        that genome. If parent genome is fully connected, return the parent genome.
        :param ParentGenome: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        NewGenotype = deepcopy(ParentGenome.GetGenotype())
        TopologyLevels = ParentGenome.GetTopologyLevels()
        # Record existing connections in genotype through pythonic 'try and fail safely' approach
        ExistingGenotypeConnections = set()
        for Gene in NewGenotype.values():
            try:
                ExistingGenotypeConnections.add((Gene.ConnectionIn, Gene.ConnectionOut))
            except AttributeError:
                pass
        # Convert PossibleConnectionsIn to a list in order to shuffle it (which is not possible would I stick with an
        # Iterator over the set), which in turn is necessary to add the connection at a genuinely random place.
        MaxIndexConnectionIn = len(TopologyLevels) - 1
        PossibleConnectionsIn = list(set.union(*TopologyLevels[:MaxIndexConnectionIn]))
        shuffle(PossibleConnectionsIn)
        bConnectionAdded = False
        for ConnectionIn in PossibleConnectionIn:
            # Determine a list of (shuffled) conn_outs for the randomized ConnectionIn
            for LayerIndex in range(MaxIndexConnectionIn):
                if ConnectionIn in TopologyLevels[LayerIndex]:
                    MinIndexConnectionOut = LayerIndex + 1
            PossibleConnectionsOut = list(set.union(*TopologyLevels[MinIndexConnectionOut:]))
            shuffle(PossibleConnectionsOut)
            for ConnectionOut in PossibleConnectionsOut:
                if (ConnectionIn, ConnectionOut) not in ExistingGenotypeConnections:
                    ConnectionWeight = np.random.normal(loc=self.MutationWeightsMean, scale=self.MutationWeightsStandardDeviation)
                    NewGeneID, NewGeneConnection = self.Encoding.CreateGeneConnection(ConnectionIn, ConnectionOut, ConnectionWeight)
                    NewGenotype[NewGeneID] = NewGeneConnection
                    bConnectionAdded = True
                    break
            if bConnectionAdded:
                break
        return self.Encoding.CreateGenome(NewGenotype)

    def CreateAddedNodeGenome(self, ParentGenome) -> (int, DirectEncodingGenome):
        """
        Create a added node genome accordng to NEAT by randomly splitting a connection. The connection is split by
        introducing a new node that has a connection to the old ConnectionIn with a ConnectionWeight of 1 and a connection to the
        old ConnectionOut with the old ConnectionWeight. Return that genome.
        :param parent_genome: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        NewGenotype = deepcopy(ParentGenome.GetGenotype())
        # Choose a random gene connection(!) from the genotype (and check that no gene node was accidentally chosen)
        GenotypeGeneIDs = tuple(NewGenotype)
        while True:
            Gene = NewGenotype[choice(GenotypeGeneIDs)]
            if hasattr(Gene, 'ConnectionWeight'):
                break
        # Extract all required information from chosen gene to build a new node in between and then disable it (Not
        # remove it as per NEAT specification). If between the ConnectionIn and ConnectionOut has already been another node added
        # in another mutation, use the same node
        ConnectionIn = Gene.ConnectionIn
        ConnectionOut = Gene.ConnectionOut
        ConnectionWeight = Gene.ConnectionWeight
        Gene.SetEnabled(False)
        if (ConnectionIn, ConnectionOut) in self.AddNodeHistory:
            Node = self.AddNodeHistory[(ConnectionIn, ConnectionOut)]
        else:
            self.NodeCounter += 1
            self.AddNodeHistory[(ConnectionIn, ConnectionOut)] = self.NodeCounter
            Node = self.NodeCounter
        GeneNodeID, GeneNode = self.Encoding.CreateGeneNode(Node, 0, self.activation_hidden)
        ConnectionInNodeID, ConnectionInNode = self.Encoding.CreateGeneConnection(ConnectionIn, Node, 1)
        ConnectionOutNodeID, ConnectionOutNode = self.Encoding.CreateGeneConnection(Node, ConnectionOut, ConnectionWeight)
        NewGenotype[GeneNodeID] = GeneNode
        NewGenotype[ConnectionInNodeID] = ConnectionInNode
        new_genotype[ConnectionOutNodeID] = ConnectionOutNode
        return self.encoding.create_genome(NewGenotype)

    def EvaluatePopulation(self, Population, GenomeEvaluationFunction):
        """
        Evaluate population by first evaluating each previously unevaluated genome on the genome_eval_function and
        saving its fitness. The population is the clustered and fitness sharing is applied according to the NEAT
        specification.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param genome_eval_function: callable method that takes a genome as input and returns the fitness score
                                     corresponding to the genomes performance in the environment
        """
        GenomeIDs = Population.GetGenomeIDs()
        for GenomeID in GenomeIDs:
            Genome = Population.GetGenome(GenomeID)
            # Only evaluate genome fitness if it has not been evaluated before (as genome doesn't change)
            if Genome.GetFitness() == 0:
                Genome.SetFitness(GenomeEvaluationFunction(Genome))
        # Speciate population by first clustering it and then applying fitness sharing
        self.ClusterPopulation(Population, GenomeIDs)
        self.ApplyFitnessSharing(Population)

    def ClusterPopulation(self, Population, GenomeIDs):
        """
        Cluster population by assigning each genome either to an existing species or to a new species, for which that
        genome will then become the representative. If a genome's distance to a species representative is below the
        distance threshold it is assigned to the species, whose species representative is closest.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param genome_ids: list of all keys/genome_ids of the population's genomes
        """
        # Create a seperate dict that already contains each species representative, as they will be accessed often
        SpeciesRepresentatives = dict()
        SpeciesRepresentativeIDs = set()
        for SpeciesID in self.SpeciesAssignment:
            SpeciesRepresentativeID = self.SpeciesAssignment[SpeciesID][0]
            SpeciesRepresentatives[SpeciesID] = Population.GetGenome(SpeciesRepresentativeID)
            SpeciesRepresentativeIDs.add(SpeciesRepresentativeID)
        # Assert that the chosen species clustering algorithm is 'threshold-fixed' as other not yet implemented
        assert self.SpeciesClustering[0] == 'threshold-fixed'
        # Determine the distance of each genome to the representative genome of each species (first genome in the list
        # of species assigned genomes) and save it in the distance dict (key: SpeciesID, value: genome's distance to
        # that the representative of that species)
        Distance = dict()
        DistanceThreshold = self.SpeciesClustering[1]
        for GenomeID in GenomeIDs:
            # Skip evaluation of genome if it is already a representative
            if GenomeID in SpeciesRepresentativeIDs:
                continue
            Genome = Population.GetGenome(GenomeID)
            for SpeciesID, SpeciesRepresentative in SpeciesRepresentatives.items():
                Distance[SpeciesID] = self.CalculateGenomeDistance(Genome, SpeciesRepresentative)
            # Determine closest species ID based on distance
            ClosestSpeciesID = min(Distance, key=Distance.get)
            if Distance[ClosestSpeciesID] <= DistanceThreshold:
                # Assign genome to the closest existing species, as distance to other species not great enough to
                # warrant the creation of a new species
                self.SpeciesAssignment[ClosestSpeciesID].append(GenomeID)
            else:
                # Genome is distant enough from any other species representative that the creation of a new species with
                # itself as the species representative is appropriate
                self.NewestSpeciesID += 1
                self.SpeciesAssignment[self.NewestSpeciesID] = [GenomeID]
                self.SpeciesAverageFitnessHistory[self.NewestSpeciesID] = []
                SpeciesRepresentatives[self.NewestSpeciesID] = Population.GetGenome(GenomeID)
        # Since all clusters are created, sort the genomes of each species by their fitness. This will set the fittest
        # genome as the new genome representative as well as allow for easy determination of the fittest fraction that
        # is allowed to reproduce later.
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            SortedSpeciesGenomeIDs = sorted(SpeciesGenomeIDs, key=lambda x: Population.GetGenome(x).GetFitness(), reverse=True)
            self.SpeciesAssignment[SpeciesID] = SortedSpeciesGenomeIDs

    def CalculateGenomeDistance(self, Genome1, Genome2) -> float:
        """
        Calculate the distance between 2 DirectEncodingGenomes according to NEAT's genome distance formula:
        distance = (c1 * E)/N + (c2 * D)/N + c3 * W
        E: amount of excess genes between both genotypes
        D: amount of disjoint genes between both genotypes
        W: average weight difference of matching genes between both genotypes. For this, TFNE does not only consider the
            weight differences between connection weights, but also weight differences between node biases.
        N: length of the longer genotype
        c1: cfg specified coefficient adjusting the importance of excess gene distance
        c2: cfg specified coefficient adjusting the importance of disjoint gene distance
        c3: cfg specified coefficient adjusting the importance weight distance
        :param genome_1: DirectEncoding genome
        :param genome_2: DirectEncoding genome
        :return: Distance between the two supplied DirectEncoding genomes in terms of number of excess genes, number of
                    disjoint genes and avg weight difference for the matching genes
        """
        Genotype1 = Genome1.GetGenotype()
        Genotype2 = Genome2.GetGenotype()
        GeneIDs1 = set(Genotype1)
        GeneIDs2 = set(Genotype2)
        MaxGenotypeLength = max(len(GeneIDs1), len(GeneIDs2))

        # Determine gene_id from which on out other genes count as excess or up to which other genes count as disjoint
        MaxGeneID1 = max(GeneIDs1)
        MaxGeneID2 = max(GeneIDs2)
        ExcessThresholdID = min(MaxGeneID1, MaxGeneID2)

        # Calculation of the first summand of the total distance, the excess gene distance. First determine excess genes
        if ExcessThresholdID == MaxGeneID1:
            ExcessGenes = set()
            for GeneID in GeneIDs2:
                if GeneID > ExcessThresholdID:
                    ExcessGenes.add(GeneID)
        else:
            ExcessGenes = set()
            for GeneID in GeneIDs1:
                if GeneID > ExcessThresholdID:
                    ExcessGenes.add(GeneID)
        ExcessGeneDistance = self.GeneExcessDistanceScalar * len(ExcessGenes) / MaxGenotypeLength

        # Calculation of the second summand of the total distance, the disjoint gene distance
        DisjointGenesLength = len((GeneIDs1.symmetric_difference(GeneIDs2)).difference(ExcessGenes))
        DisjointGenesDistance = self.GeneDisjointDistanceScalar * DisjointGenesLength / MaxGenotypeLength

        # Calculation of the third summand of the total distance, the average weight differences of matching genes
        MatchingGeneIDs = GeneIDs1.intersection(GeneIDs2)
        TotalWeightDifference = 0
        for GeneID in MatchingGeneIDs:
            try:
                TotalWeightDifference += np.abs(np.subtract(Genotype1[GeneID].ConnectionWeight, Genotype2[GeneID].ConnectionWeight))
            except AttributeError:
                TotalWeightDifference += np.abs(np.subtract(Genotype1[GeneID].Bias, Genotype2[GeneID].Bias))
        MatchingGeneDistance = self.GeneMatchingDistanceScalar * TotalWeightDifference / len(MatchingGeneIDs)

        return ExcessGeneDistance + DisjointGenesDistance + MatchingGeneDistance

    def ApplyFitnessSharing(self, Population):
        """
        Calculate the adjusted fitness score of each genome and save it in an internal dict as it is required for the
        calculation of the allotted offspring for each species. Also log the species avg fitness as all calculations are
        performed here anyway
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        """
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            SpeciesSize = len(SpeciesGenomeIDs)
            FitnessSum = 0
            for GenomeID in SpeciesGenomeIDs:
                GenomeFitness = Population.GetGenome(GenomeID).GetFitness()
                FitnessSum += GenomeFitness
                # Calculate the genome's adjusted fitness to 3 decimal places and save it internally
                AdjustedFitness = round(GenomeFitness / SpeciesSize, 3)
                self.GenomesAdjustedFitness[GenomeID] = AdjustedFitness
            SpeciesAverageFitness = round(FitnessSum / SpeciesSize, 3)
            self.SpeciesAverageFitnessHistory[SpeciesID].append(SpeciesAverageFitness)

    def SummarizePopulation(self, Population):
        """
        Output a summary of the population to logging.info, summarizing the status of the whole population as well as
        the status of each species in particular. Also output the string representation fo each genome to logging.debug.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        """
        # Summarize whole population:
        Generation = Population.get_generation_counter()
        BestFitness = Population.GetBestGenome().GetFitness()
        AverageFitness = Population.GetAverageFitness()
        PopulationSize = Population.GetPopulationSize()
        SpeciesCount = len(self.SpeciesAssignment)
        logging.info("#### Generation: {:>4} ## BestFitness: {:>8} ## AverageFitness: {:>8} ## PopulationSize: {:>4} ## SpeciesCount: {:>4} ####"
                     .format(Generation, BestFitness, AverageFitness, PopulationSize, SpeciesCount))

        # Summarize each species and its genomes seperately
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            SpeciesBestFitness = Population.GetGenome(SpeciesGenomeIDs[0]).GetFitness()
            SpeciesAverageFitness = self.SpeciesAverageFitnessHistory[SpeciesID][-1]
            SpeciesSize = len(SpeciesGenomeIDs)
            logging.info("---- SpeciesID: {:>4} -- SpeciesBestFitness: {:>4} -- SpeciesAverageFitness: {:>4} -- SpeciesSize: {:>4} ----"
                         .format(SpeciesID, SpeciesBestFitness, SpeciesAverageFitness, SpeciesSize))
            for GenomeID in SpeciesGenomeIDs:
                logging.debug(Population.GetGenome(GenomeID))

    def GetSpeciesReport(self) -> dict:
        """
        Create a species report dict listing all currently present species ids as keys and assigning them the size of
        their species as value.
        :return: dict, containing said species report assigning SpeciesID to size of species
        """
        SpeciesReport = dict()
        for SpeciesID, SpeciesGenomeIDs in self.SpeciesAssignment.items():
            SpeciesReport[SpeciesID] = len(SpeciesGenomeIDs)
        return SpeciesReport