import random
from Benchmarking import PerfTimer
from NEAT.Graphs import FeedForwardLayers


class FeedForwardNetwork(object):
    def __init__(self, Inputs, Outputs, NodeEvals):
        self.InNodes = Inputs
        self.OutNodes = Outputs
        self.NodeEvals = NodeEvals
        self.Values = dict((Key, 0.0) for Key in Inputs + Outputs)

    def Activate(self, Inputs):
        if len(self.InNodes) != len(Inputs):
            raise RuntimeError("Expected {0:n} Inputs, got {1:n}".format(len(self.InNodes), len(Inputs)))
        for Key, Node in zip(self.InNodes, Inputs):
            self.Values[Key] = Node
        for Node, ActivationFunction, AggregationFunction, Bias, Response, Links in self.NodeEvals:
            NodeInputs = [(self.Values[Key] * Weight) for Key, Weight in Links]
            Aggregated = AggregationFunction(NodeInputs)
            self.Values[Node] = ActivationFunction(Bias + (Response * Aggregated))
        return [self.Values[Key] for Key in self.OutNodes]

    @staticmethod
    def Create(Genome, Config):
        """ Receives a Genome and returns its phenotype (a FeedForwardNetwork). """
        # Gather expressed Connections.
        Connections = [GenomeConnection.Key for GenomeConnection in Genome.Connections.values() if GenomeConnection.bEnabled]
        Layers = FeedForwardLayers(Config.GenomeConfig.InputKeys, Config.GenomeConfig.OutputKeys, Connections)
        NodeEvals = []
        for Layer in Layers:
            for Node in Layer:
                Inputs = []
                NodeExpression = [] # currently unused
                for ConnectionKey in Connections:
                    InNode, OutNode = ConnectionKey
                    if OutNode == Node:
                        GenomeConnection = Genome.Connections[ConnectionKey]
                        Inputs.append((InNode, GenomeConnection.Weight))
                        NodeExpression.append("v[{}] * {:.7e}".format(InNode, GenomeConnection.Weight))
                GenomeNode = Genome.Nodes[Node]
                AggregationFunction = Config.GenomeConfig.AggregationFunctionsSet.Get(GenomeNode.Aggregation)
                ActivationFunction = Config.GenomeConfig.ActivationFunctionsSet.Get(GenomeNode.Activation)
                NodeEvals.append((Node, ActivationFunction, AggregationFunction, GenomeNode.Bias, GenomeNode.Response, Inputs))
        return FeedForwardNetwork(Config.GenomeConfig.InputKeys, Config.GenomeConfig.OutputKeys, NodeEvals)