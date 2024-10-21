"""Directed graph algorithm implementations."""


def CreatesCycle(Connections, TestKey):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    InNodeID, OutNodeID = TestKey
    if InNodeID == OutNodeID:
        return True
    VisitedNodes = {OutNodeID}
    while True:
        NumAdded = 0
        for LoopInNode, LoopOutNode in Connections:
            if LoopInNode in VisitedNodes and LoopOutNode not in VisitedNodes:
                if LoopOutNode == InNodeID:
                    return True
                VisitedNodes.add(LoopOutNode)
                NumAdded += 1
        if NumAdded == 0:
            return False


def RequiredForOutput(Inputs, Outputs, Connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.
    Returns a set of identifiers of required nodes.
    """
    Required = set(Outputs)
    Seen = set(Outputs)
    while 1:
        # Find nodes not in Seen whose output is consumed by a node in Seen.
        Chained = set(InNode for (InNode, OutNode) in Connections if OutNode in Seen and InNode not in Seen)
        if not Chained:
            break
        LayerNodes = set(Node for Node in Chained if Node not in Inputs)
        if not LayerNodes:
            break
        Required = Required.union(LayerNodes)
        Seen = Seen.union(Chained)
    return Required


def FeedForwardLayers(Inputs, Outputs, Connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """
    Required = RequiredForOutput(Inputs, Outputs, Connections)
    Layers = []
    Seen = set(Inputs)
    while True:
        # Find candidate nodes Chained for the next layer. These nodes should connect a node in Seen to a node not in Seen.
        Candidates = set(OutNode for (InNode, OutNode) in Connections if InNode in Seen and OutNode not in Seen)
        # Keep only the used nodes whose entire input set is contained in Seen.
        Chained = set()
        for Node in Candidates:
            if Node in Required and all(InNode in Seen for (InNode, OutNode) in Connections if OutNode == Node):
                Chained.add(Node)
        if not Chained:
            break
        Layers.append(Chained)
        Seen = Seen.union(Chained)
    return Layers