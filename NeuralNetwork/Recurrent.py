from NEAT.Graphs import RequiredForOutput


class RecurrentNetwork(object):
    def __init__(self, Inputs, Outputs, NodeEvals):
        self.InNodes = Inputs
        self.OutNodes = Outputs
        self.NodeEvals = NodeEvals
        self.Values = [{}, {}]
        for Value in self.Values:
            for Key in list(Inputs) + list(Outputs):
                Value[Key] = 0.0
            for Node, IgnoredActivation, IgnoredAggregation, IgnoredBias, IgnoredResponse, Links in self.NodeEvals:
                Value[Node] = 0.0
                for Key, Weight in Links:
                    Value[Key] = 0.0
        self.Active = 0

    def Reset(self):
        self.Values = [dict((Key, 0.0) for Key in Value) for Value in self.Values]
        self.Active = 0

    def Activate(self, Inputs):
        if len(self.InNodes) != len(Inputs):
            raise RuntimeError("Expected {0:n} Inputs, got {1:n}".format(len(self.InNodes), len(Inputs)))
        InputValues = self.Values[self.Active]
        OutputValues = self.Values[1 - self.Active]
        self.Active = 1 - self.Active
        for Key, Value in zip(self.InNodes, Inputs):
            InputValues[Key] = Value
            OutputValues[Key] = Value
        for Node, Activation, Aggregation, Bias, Response, Links in self.NodeEvals:
            NodeInputs = [InputValues[Key] * Weight for Key, Weight in Links]
            Aggregated = Aggregation(NodeInputs)
            OutputValues[Node] = Activation(Bias + Response * Aggregated)
        return [OutputValues[Key] for Key in self.OutNodes]

    @staticmethod
    def Create(Genome, Config):
        """ Receives a Genome and returns its phenotype (a RecurrentNetwork). """
        GenomeConfig = Config.GenomeConfig
        Required = RequiredForOutput(GenomeConfig.InputKeys, GenomeConfig.OutputKeys, Genome.Connections)
        # Gather Inputs and expressed Connections.
        NodeInputs = {}
        for GenomeConnection in Genome.Connections.values():
            if not GenomeConnection.bEnabled:
                continue
            Key, Output = GenomeConnection.Key
            if Output not in Required and Key not in Required:
                continue
            if Output not in NodeInputs:
                NodeInputs[Output] = [(Key, GenomeConnection.Weight)]
            else:
                NodeInputs[Output].append((Key, GenomeConnection.Weight))
        NodeEvals = []
        for NodeKey, Inputs in NodeInputs.items():
            Node = Genome.Nodes[NodeKey]
            ActivationFunction = GenomeConfig.ActivationFunctionsSet.Get(Node.Activation)
            AggregationFunction = GenomeConfig.AggregationFunctionsSet.Get(Node.Aggregation)
            NodeEvals.append((NodeKey, ActivationFunction, AggregationFunction, Node.Bias, Node.Response, Inputs))
        return RecurrentNetwork(GenomeConfig.InputKeys, GenomeConfig.OutputKeys, NodeEvals)