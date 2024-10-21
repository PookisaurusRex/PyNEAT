"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function

import numpy as np
from itertools import count
from random import choice, random, shuffle

import sys
import time

from NEAT.Activations import ActivationFunctionSet
from NEAT.Aggregations import AggregationFunctionSet
from NEAT.Config import ConfigParameter, WritePrettyParameters
from NEAT.Genes import DefaultConnectionGene, DefaultNodeGene
from NEAT.Graphs import CreatesCycle

from Benchmarking import PerfTimer

class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    AllowedConnectionTypes = ['Unconnected', 'FeatureSelection', 'FeatureSelectionHidden', 'FeatureSelectionNoHidden', 
                              'Full', 'FullDirect', 'FullNoDirect', 'Partial', 'PartialDirect', 'PartialNoDirect',]

    def __init__(self, Parameters):
        # Create full set of available activation functions.
        self.ActivationFunctionsSet = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.AggregationFunctionsSet = AggregationFunctionSet()

        self.Parameters = [ConfigParameter('NumInputPins', int),
                        ConfigParameter('NumOutputPins', int),
                        ConfigParameter('NumHiddenNodes', int),
                        ConfigParameter('bFeedForward', bool),
                        ConfigParameter('CompatibilityDisjointCoefficient', float),
                        ConfigParameter('CompatibilityWeightCoefficient', float),
                        ConfigParameter('ConnectionAddProbability', float),
                        ConfigParameter('ConnectionDeleteProbability', float),
                        ConfigParameter('NodeAddProbability', float),
                        ConfigParameter('NodeDeleteProbability', float),
                        ConfigParameter('bSingleStructuralMutation', bool, 'false'),
                        ConfigParameter('StructuralMutationSurer', str, 'default'),
                        ConfigParameter('InitialConnectionType', str, 'Unconnected')]

        self.NodeIndexer = None

        # Gather configuration data from the gene classes.
        self.NodeGeneType = Parameters['NodeGeneType']
        self.Parameters += self.NodeGeneType.GetConfigParameters()
        self.ConnectionGeneType = Parameters['ConnectionGeneType']
        self.Parameters += self.ConnectionGeneType.GetConfigParameters()

        # Use the configuration data to interpret the supplied parameters.
        for Parameter in self.Parameters:
            if isinstance(Parameter, list):
                for SubParam in Parameter:
                    setattr(self, SubParam.Name, SubParam.Interpret(Parameters))
            else:
                setattr(self, Parameter.Name, Parameter.Interpret(Parameters))

        # By convention, input pins have negative keys, and the output pins have keys 0,1,...
        self.InputKeys = [-InputPinKey - 1 for InputPinKey in range(self.NumInputPins)]
        self.OutputKeys = [OutputPinKey for OutputPinKey in range(self.NumOutputPins)]

        self.ConnectionFraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'Partial' in self.InitialConnectionType:
            InitialConnectionType, ConnectionFraction = self.InitialConnectionType.split()
            self.InitialConnectionType = InitialConnectionType
            self.ConnectionFraction = float(ConnectionFraction)
            if not (0 <= self.ConnectionFraction <= 1):
                raise RuntimeError("'partial' connection value must be between 0.0 and 1.0, inclusive.")
        assert self.InitialConnectionType in DefaultGenomeConfig.AllowedConnectionTypes
        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.StructuralMutationSurer.lower() in ['1', 'yes', 'true', 'on']:
            self.StructuralMutationSurer = 'true'
        elif self.StructuralMutationSurer.lower() in ['0', 'no', 'false', 'off']:
            self.StructuralMutationSurer = 'false'
        elif self.StructuralMutationSurer.lower() == 'default':
            self.StructuralMutationSurer = 'default'
        else:
            ErrorString = "Invalid StructuralMutationSurer {!r}".format(self.StructuralMutationSurer)
            raise RuntimeError(ErrorString)

    def AddActivationFunction(self, Name, Function):
        self.ActivationFunctionsSet.Add(Name, Function)

    def AddAggregationFunction(self, Name, Function):
        self.AggregationFunctionsSet.Add(Name, Function)

    def SetNumInputs(self, NumInputs):
        self.NumInputPins = NumInputs
        self.InputKeys = [-InputPinKey - 1 for InputPinKey in range(self.NumInputPins)]

    def SetNumOutputs(self, NumOutputs):
        self.NumOutputPins = NumOutputs
        self.OutputKeys = [OutputPinKey for OutputPinKey in range(self.NumOutputPins)]

    def Save(self, File):
        if 'Partial' in self.InitialConnectionType:
            if not (0 <= self.ConnectionFraction <= 1):
                raise RuntimeError("'Partial' connection value must be between 0.0 and 1.0, inclusive.")
            File.write('InitialConnectionType      = {0} {1}\n'.format(self.InitialConnectionType, self.ConnectionFraction))
        else:
            File.write('InitialConnectionType      = {0}\n'.format(self.InitialConnectionType))
        assert self.InitialConnectionType in DefaultGenomeConfig.AllowedConnectionTypes
        WritePrettyParameters(File, self, [Parameter for Parameter in self.Parameters if 'InitialConnectionType' not in Parameter.Name])

    def GetNewNodeKey(self, NodeDictionary):
        if self.NodeIndexer is None:
            self.NodeIndexer = count(max(list(NodeDictionary)) + 1)
        NewID = next(self.NodeIndexer)
        assert NewID not in NodeDictionary
        return NewID

    def CheckStructuralMutationSurer(self):
        if self.StructuralMutationSurer == 'true':
            return True
        elif self.StructuralMutationSurer == 'false':
            return False
        elif self.StructuralMutationSurer == 'default':
            return self.StructuralMutationSurer
        else:
            ErrorString = "Invalid structural_mutation_surer {!r}".format(self.StructuralMutationSurer)
            raise RuntimeError(ErrorString)

class DefaultGenome(object):
    """
    A genome for generalized neural networks.
    Terminology
        pin: Point at which the network is conceptually connected to the external world; pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.
    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique neuron by an implicit connection with weight one. This connection is permanently enabled.
        2. The output pin's key is always the same as the key for its associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """
    @classmethod
    def ParseConfig(cls, ParameterDictionary):
        ParameterDictionary['NodeGeneType'] = DefaultNodeGene
        ParameterDictionary['ConnectionGeneType'] = DefaultConnectionGene
        return DefaultGenomeConfig(ParameterDictionary)

    @classmethod
    def WriteConfig(cls, File, Config):
        Config.Save(File)

    def __init__(self, Key):
        # Unique identifier for a genome instance.
        self.Key = Key
        # (gene_key, gene) pairs for gene sets.
        self.Connections = {}
        self.Nodes = {}
        # Fitness results.
        self.Fitness = None

    def ConfigureNew(self, Config):
        """Configure a new genome based on the given configuration."""
        # Create node genes for the output pins.
        for NodeKey in Config.OutputKeys:
            self.Nodes[NodeKey] = self.CreateNode(Config, NodeKey)
        # Add hidden nodes if requested.
        if Config.NumHiddenNodes > 0:
            for i in range(Config.NumHiddenNodes):
                NodeKey = Config.GetNewNodeKey(self.Nodes)
                assert NodeKey not in self.Nodes
                Node = self.CreateNode(Config, NodeKey)
                self.Nodes[NodeKey] = Node
        # Add connections based on initial connectivity type.
        if 'FeatureSelection' in Config.InitialConnectionType:
            if Config.InitialConnectionType == 'FeatureSelectionNoHidden':
                self.ConnectFeatureSelectionNoHidden(Config)
            elif Config.InitialConnectionType == 'FeatureSelectionHidden':
                self.ConnectFeatureSelectionHidden(Config)
            else:
                if Config.NumHiddenNodes > 0:
                    print("Warning: InitialConnectionType = fs_neat will not connect to hidden nodes;",
                          "\tif this is desired, set InitialConnectionType = fs_neat_nohidden;",
                          "\tif not, set InitialConnectionType = FeatureSelectionHidden",
                          sep='\n', file=sys.stderr)
                self.ConnectFeatureSelectionNoHidden(Config)
        elif 'Full' in Config.InitialConnectionType:
            if Config.InitialConnectionType == 'FullNoDirect':
                self.ConnectFullNoDirect(Config)
            elif Config.InitialConnectionType == 'FullDirect':
                self.ConnectFullDirect(Config)
            else:
                if Config.NumHiddenNodes > 0:
                    print("Warning: InitialConnectionType = full with hidden nodes will not do direct input-output connections;",
                          "\tif this is desired, set InitialConnectionType = FullNoDirect;", "\tif not, set InitialConnectionType = FullDirect",
                          sep='\n', file=sys.stderr)
                self.ConnectFullNoDirect(Config)
        elif 'Partial' in Config.InitialConnectionType:
            if Config.InitialConnectionType == 'PartialNoDirect':
                self.ConnectPartialNoDirect(Config)
            elif Config.InitialConnectionType == 'PartialDirect':
                self.ConnectPartialDirect(Config)
            else:
                if Config.NumHiddenNodes > 0:
                    print("Warning: InitialConnectionType = partial with hidden nodes will not do direct input-output connections;",
                          "\tif this is desired, set InitialConnectionType = PartialNoDirect {0};".format(Config.ConnectionFraction),
                          "\tif not, set InitialConnectionType = PartialDirect {0}".format(Config.ConnectionFraction),
                          sep='\n', file=sys.stderr)
                self.ConnectPartialNoDirect(Config)

    def ConfigureCrossover(self, Genome1, Genome2, Config):
        """ Configure a new genome by crossover from two parent genomes. """
        if (Genome1.Fitness > Genome2.Fitness):
            Parent1, Parent2 = Genome1, Genome2  
        else:
            Parent1, Parent2 = Genome2, Genome1 

        # Inherit connection genes
        for Key, ConnectionGene1 in Parent1.Connections.items():
            ConnectionGene2 = Parent2.Connections.get(Key)
            # Excess or disjoint gene: copy from the fittest parent. # Homologous gene: combine genes from both parents.
            self.Connections[Key] = ConnectionGene1.Copy() if (ConnectionGene2 is None) else ConnectionGene1.Crossover(ConnectionGene2)
        # Inherit node genes
        ParentNodes1 = Parent1.Nodes
        ParentNodes2 = Parent2.Nodes
        for Key, NodeGene1 in ParentNodes1.items():
            NodeGene2 = ParentNodes2.get(Key)
            assert Key not in self.Nodes
            # Extra gene: copy from the fittest parent. # Homologous gene: combine genes from both parents.
            self.Nodes[Key] = NodeGene1.Copy() if (NodeGene2 is None) else NodeGene1.Crossover(NodeGene2)

    def ConfigureClone(self, Parent, Config):
        for Key, ConnectionGene in Parent.Connections.items():
            self.Connections[Key] = ConnectionGene.Copy()
        for Key, NodeGene in Parent.Nodes.items():
            self.Nodes[Key] = NodeGene.Copy()

    def Mutate(self, Config):
        """ Mutates this genome. """
        if Config.bSingleStructuralMutation:
            ProbabilitySum = max(1, (Config.NodeAddProbability + Config.NodeDeleteProbability + Config.ConnectionAddProbability + Config.ConnectionDeleteProbability))
            NodeAddProbability = Config.NodeAddProbability / ProbabilitySum
            NodeDeleteProbability = Config.NodeDeleteProbability / ProbabilitySum
            ConnectionAddProbability = Config.ConnectionAddProbability / ProbabilitySum
            ConnectionDeleteProbability = Config.ConnectionDeleteProbability / ProbabilitySum
            r = random()
            if r < NodeAddProbability:
                self.MutateAddNode(Config)
            elif r < (NodeAddProbability + NodeDeleteProbability):
                self.MutateDeleteNode(Config)
            elif r < (NodeAddProbability + NodeDeleteProbability + ConnectionAddProbability):
                self.MutateAddConnection(Config)
            elif r < (NodeAddProbability + NodeDeleteProbability + ConnectionAddProbability + ConnectionDeleteProbability):
                self.MutateDeleteConnection()
        else:
            # Node structural mutations
            NodeMutationProbability = max(1, (Config.NodeAddProbability + Config.NodeDeleteProbability))
            NodeAddProbability = Config.NodeAddProbability / NodeMutationProbability
            NodeDeleteProbability = Config.NodeDeleteProbability / NodeMutationProbability
            NodeMutationRoll = random()
            if NodeMutationRoll < NodeAddProbability:
                self.MutateAddNode(Config)
            elif NodeMutationRoll < (NodeAddProbability + NodeDeleteProbability):
                self.MutateDeleteNode(Config)
            # Connection structural mutations
            ConnectionMutationProbability = max(1, (Config.ConnectionAddProbability + Config.ConnectionDeleteProbability))
            ConnectionAddProbability = Config.ConnectionAddProbability / ConnectionMutationProbability
            ConnectionDeleteProbability = Config.ConnectionDeleteProbability / ConnectionMutationProbability
            ConnectionMutationRoll = random()
            if ConnectionMutationRoll < ConnectionAddProbability:
                self.MutateAddConnection(Config)
            elif ConnectionMutationRoll < (ConnectionAddProbability + ConnectionDeleteProbability):
                self.MutateDeleteConnection()
        # Mutate connection genes.
        for ConnectionGene in self.Connections.values():
            ConnectionGene.Mutate(Config)
        # Mutate node genes (bias, response, etc.).
        for NodeGene in self.Nodes.values():
            NodeGene.Mutate(Config)

    def MutateAddNode(self, Config):
        if not self.Connections:
            if Config.CheckStructuralMutationSurer():
                self.MutateAddConnection(Config)
            return
        # Choose a random connection to split
        ConnectionToSplit = choice(list(self.Connections.values()))
        NewNodeID = Config.GetNewNodeKey(self.Nodes)
        NodeGene = self.CreateNode(Config, NewNodeID)
        self.Nodes[NewNodeID] = NodeGene
        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        ConnectionToSplit.bEnabled = False
        InNodeID, OutNodeID = ConnectionToSplit.Key
        self.AddConnection(Config, InNodeID, NewNodeID, 1.0, True)
        self.AddConnection(Config, NewNodeID, OutNodeID, ConnectionToSplit.Weight, True)

    def AddConnection(self, Config, InNodeID, OutNodeID, Weight, bEnabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(InNodeID, int)
        assert isinstance(OutNodeID, int)
        assert OutNodeID >= 0
        assert isinstance(bEnabled, bool)
        Key = (InNodeID, OutNodeID)
        Connection = Config.ConnectionGeneType(Key)
        Connection.InitAttributes(Config)
        Connection.Weight = Weight
        Connection.bEnabled = bEnabled
        self.Connections[Key] = Connection

    def MutateAddConnection(self, Config):
        """
        Attempt to add a new connection, the only restriction being that the output node cannot be one of the network input pins.
        """
        PossibleOutNodes = list(self.Nodes)
        OutNodeKey = choice(PossibleOutNodes)
        PossibleInNodes = PossibleOutNodes + Config.InputKeys
        InNodeKey = choice(PossibleInNodes)
        # Don't duplicate connections.
        Key = (InNodeKey, OutNodeKey)
        if Key in self.Connections:
            if Config.CheckStructuralMutationSurer():
                self.Connections[Key].bEnabled = True
            return
        # Don't allow connections between two output nodes
        if InNodeKey in Config.OutputKeys and OutNodeKey in Config.OutputKeys:
            return
        # No need to check for connections between input nodes: they cannot be the output end of a connection (see above).
        # For feed-forward networks, avoid creating cycles.
        if Config.bFeedForward and CreatesCycle(list(self.Connections), Key):
            return
        ConnectionGene = self.CreateConnection(Config, InNodeKey, OutNodeKey)
        self.Connections[ConnectionGene.Key] = ConnectionGene

    def MutateDeleteNode(self, Config):
        # Do nothing if there are no non-output nodes.
        AvailableNodes = [PossibleNode for PossibleNode in self.Nodes if PossibleNode not in Config.OutputKeys]
        if not AvailableNodes:
            return -1
        DeleteKey = choice(AvailableNodes)
        ConnectionsToDelete = set([Value.Key for Key, Value in self.Connections.items() if DeleteKey in Value.Key])
        for Key in ConnectionsToDelete:
            del self.Connections[Key]
        del self.Nodes[DeleteKey]
        return DeleteKey

    def MutateDeleteConnection(self):
        if self.Connections:
            Key = choice(list(self.Connections.keys()))
            del self.Connections[Key]

    def Distance(self, Other, Config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """
        # Compute node gene distance component.
        NodeDistance = 0.0
        if self.Nodes or Other.Nodes:
            DisjointNodes = np.sum([1 for Key2 in Other.Nodes if Key2 not in self.Nodes])
            #DisjointNodes = 0
            #for Key2 in Other.Nodes:
            #    if Key2 not in self.Nodes:
            #        DisjointNodes += 1
            for Key1, Node1 in self.Nodes.items():
                Node2 = Other.Nodes.get(Key1)
                if Node2 is None:
                    DisjointNodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    NodeDistance += Node1.Distance(Node2, Config)
            MaxNodes = max(len(self.Nodes), len(Other.Nodes))
            NodeDistance = (NodeDistance + (Config.CompatibilityDisjointCoefficient * DisjointNodes)) / MaxNodes
        # Compute connection gene differences.
        ConnectionDistance = 0.0
        if self.Connections or Other.Connections:
            DisjointConnections = 0
            for Key2 in Other.Connections:
                if Key2 not in self.Connections:
                    DisjointConnections += 1
            for Key1, Connection1 in self.Connections.items():
                Connection2 = Other.Connections.get(Key1)
                if Connection2 is None:
                    DisjointConnections += 1
                else:
                    # Homologous genes compute their own distance value.
                    ConnectionDistance += Connection1.Distance(Connection2, Config)
            MaxConnections = max(len(self.Connections), len(Other.Connections))
            ConnectionDistance = (ConnectionDistance + (Config.CompatibilityDisjointCoefficient * DisjointConnections)) / MaxConnections
        Distance = NodeDistance + ConnectionDistance
        return Distance

    def Size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        NumNodes = len(self.Nodes)
        NumEnabledConnections = 0
        for ConnectionGene in self.Connections.values():
            if ConnectionGene.bEnabled:
                NumEnabledConnections += 1
        return NumNodes, NumEnabledConnections

    def __str__(self):
        String = "Key: {0}\nFitness: {1}\nNodes:".format(self.Key, self.Fitness)
        for Key, NodeGene in self.Nodes.items():
            String += "\n\t{0} {1!s}".format(Key, NodeGene)
        String += "\nConnections:"
        Connections = list(self.Connections.values())
        Connections.sort()
        for Connection in Connections:
            String += "\n\t" + str(Connection)
        return String

    @staticmethod
    def CreateNode(Config, NodeID):
        Node = Config.NodeGeneType(NodeID)
        Node.InitAttributes(Config)
        return Node

    @staticmethod
    def CreateConnection(Config, InNodeID, OutNodeID):
        Connection = Config.ConnectionGeneType((InNodeID, OutNodeID))
        Connection.InitAttributes(Config)
        return Connection

    def ConnectFeatureSelectionNoHidden(self, Config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        InNodeID = choice(Config.InputKeys)
        for OutNodeID in Config.OutputKeys:
            Connection = self.CreateConnection(Config, InNodeID, OutNodeID)
            self.Connections[Connection.Key] = Connection

    def ConnectFeatureSelectionHidden(self, Config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        InNodeID = choice(Config.InputKeys)
        Others = [OtherNode for OtherNode in self.Nodes if OtherNode not in Config.InputKeys]
        for OutNodeID in Others:
            Connection = self.CreateConnection(Config, InNodeID, OutNodeID)
            self.Connections[Connection.Key] = Connection

    def ComputeFullConnections(self, Config, bDirect):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        HiddenNodes = [Node for Node in self.Nodes if Node not in Config.OutputKeys]
        OutNodes = [Node for Node in self.Nodes if Node in Config.OutputKeys]
        Connections = []
        if HiddenNodes:
            for InNodeID in Config.InputKeys:
                for HiddenNode in HiddenNodes:
                    Connections.append((InNodeID, HiddenNode))
            for HiddenNode in HiddenNodes:
                for OutNodeID in OutNodes:
                    Connections.append((HiddenNode, OutNodeID))
        if bDirect or (not HiddenNodes):
            for InNodeID in Config.InputKeys:
                for OutNodeID in OutNodes:
                    Connections.append((InNodeID, OutNodeID))
        # For recurrent genomes, include node self-connections.
        if not Config.bFeedForward:
            for Node in self.Nodes:
                Connections.append((Node, Node))
        return Connections

    def ConnectFullNoDirect(self, Config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for InNodeID, OutNodeID in self.ComputeFullConnections(Config, False):
            Connection = self.CreateConnection(Config, InNodeID, OutNodeID)
            self.Connections[Connection.Key] = Connection

    def ConnectFullDirect(self, Config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for InNodeID, OutNodeID in self.ComputeFullConnections(Config, True):
            Connection = self.CreateConnection(Config, InNodeID, OutNodeID)
            self.Connections[Connection.Key] = Connection

    def ConnectPartialNoDirect(self, Config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= Config.ConnectionFraction <= 1
        AllConnections = self.ComputeFullConnections(Config, False)
        shuffle(AllConnections)
        NumberToAdd = int(round(len(AllConnections) * Config.ConnectionFraction))
        for InNodeID, OutNodeID in AllConnections[:NumberToAdd]:
            Connection = self.CreateConnection(Config, InNodeID, OutNodeID)
            self.Connections[Connection.Key] = Connection

    def ConnectPartialDirect(self, Config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= Config.ConnectionFraction <= 1
        AllConnections = self.ComputeFullConnections(Config, True)
        shuffle(AllConnections)
        NumberToAdd = int(round(len(AllConnections) * Config.ConnectionFraction))
        for InNodeID, OutNodeID in AllConnections[:NumberToAdd]:
            Connection = self.CreateConnection(Config, InNodeID, OutNodeID)
            self.Connections[Connection.key] = Connection