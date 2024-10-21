import numpy as np
import tensorflow as tf
from toposort import toposort

def ProcessGenotype(Genotype) -> (dict, dict, dict):
    """
    Process genotype by creating different dicts that contain all required information to create a Tensorflow model
    out of the genotype. Those three created dicts are explained in detail below.
    :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
    :return: tuple of nodes dict, connections dict and node_dependencies dict
             Nodes: dict with node (usually int) as dict key and a tuple of the node bias and activation function as
                    the dict value
             Connections: dict representing each connection by associating each node (dict key) with the nodes it
                          receives input from and the weight of their connection (dict value). The conn_out of the
                          connection gene is the dict key and a seperate dict, associating the conn_in with the
                          conn_weight of the connection gene, is the dict value.
             NodeDependencies: dict associating each node (dict key) with a set of the nodes (dict value) it
                                receives input from
    """
    Nodes = dict()
    Connections = dict()
    NodeDependencies = dict()
    for Gene in Genotype.values():
        try:  # If Gene isinstance of DirectEncodingConnection:
            # skip Gene if it is disabled
            if not Gene.bEnabled:
                continue
            ConnectionIn = Gene.ConnectionIn
            ConnectionOut = Gene.ConnectionOut
            if ConnectionOut in Connections:
                Connections[ConnectionOut][ConnectionIn] = Gene.ConnectionWeight
                NodeDependencies[ConnectionOut].add(ConnectionIn)
            else:
                Connections[ConnectionOut] = {ConnectionIn: Gene.ConnectionWeight}
                NodeDependencies[ConnectionOut] = {ConnectionIn}
        except AttributeError:  # else (Gene isinstance of DirectEncodingNode):
            Nodes[Gene.Node] = (Gene.Bias, Gene.Activation)
    return Nodes, Connections, NodeDependencies

def CreateNodeCoordinates(TopologyLevels) -> dict:
    """
    Create and return a dict associating each node (dict key) with their coordinate value (dict value) in the form
    of (layer_index, node_index) as they are organized in the supplied topology_level parameter.
    :param topology_levels: tuple with each element specifying the set of nodes that have to be precomputed before
                            the next element's set of nodes can be computed, as they serve as input nodes to this
                            next element's set of nodes
    :return: dict associating the nodes in topology_levels (dict key) with the coordinates in the topology_levels
             (dict value)
    """
    NodeCoordinates = dict()
    for LayerIndex in range(len(TopologyLevels)):
        LayerIterable = iter(TopologyLevels[LayerIndex])
        for NodeIndex in range(len(TopologyLevels[LayerIndex])):
            Node = next(LayerIterable)
            NodeCoordinates[Node] = (LayerIndex, NodeIndex)
    return NodeCoordinates

def JoinKeys(NodeDependencies) -> dict:
    """
    Recreate node_dependencies parameter dict by using frozensets as keys consisting of all keys that have the same
    dict value. Also convert the dict value of the input parameter to tuple. Return this converted dict.
    :param node_dependencies: dict associating each node (dict key) with a set of the nodes (dict value) it receives
                              input from
    :return: dict associating the frozenset of all keys with the same dict value with this dict value.
    """
    ValuesToKeys = dict()
    for Key, Value in NodeDependencies.items():
        FrozenValue = frozenset(Value)
        if FrozenValue in ValuesToKeys:
            ValuesToKeys[FrozenValue].add(Key)
        else:
            ValuesToKeys[FrozenValue] = {Key}
    return {frozenset(Value): tuple(Key) for Key, Value in ValuesToKeys.items()}

class CustomWeightAndInputLayerTrainable(tf.keras.layers.Layer):
    """
    Custom Tensorflow layer that allows for arbitrary input nodes from any layer as well as custom kernel and bias
    weight setting. The arbitrariness of the input nodes is made possible through usage of coordinates specifying the
    layer and exact node in that layer for every input node for the CustomWeightAndInputLayer. The layer is fully
    compatible with the rest of the Tensorflow infrastructure and supports static-graph building, auto-gradient, etc.
    """
    Initializer = tf.keras.initializers.zeros()

    def __init__(self, Activation, KernalWeights, BiasWeights, InputNodeCoordinates, DataType, bDynamic):
        super(CustomWeightAndInputLayerTrainable, self).__init__(trainable=True, dtype=DataType, dynamic=bDynamic)
        self.Activation = Activation
        self.Kernel = self.add_weight(shape=KernalWeights.shape, dtype=self.dtype, initializer=CustomWeightAndInputLayerTrainable.Initializer, trainable=True)
        self.bias = self.add_weight(shape=BiasWeights.shape, dtype=self.dtype, initializer=CustomWeightAndInputLayerTrainable.Initializer, trainable=True)
        self.set_weights((KernalWeights, BiasWeights))
        self.built = True

        if len(InputNodeCoordinates) >= 2:
            self.InputNodeCoordinates = InputNodeCoordinates
            self.call = self.CallMultipleInputs
        else:
            (self.LayerIndex, self.NodeIndex) = InputNodeCoordinates[0]
            self.call = self.CallSingleInput

    def CallSingleInput(self, Inputs, **kwargs) -> tf.Tensor:
        """
        Layer call, whereby the layer has only a single input node
        :param inputs: array of Tensorflow tensors representing the output of each preceding layer
        :return: Tensorflow tensor of the computed layer results
        """
        SelectedInputs = inputs[self.LayerIndex][:, self.NodeIndex:self.NodeIndex + 1]
        return self.Activation(tf.matmul(SelectedInputs, self.Kernel) + self.Bias)

    def CallMultipleInputs(self, Inputs, **kwargs) -> tf.Tensor:
        """
        Layer call, whereby the layer has more than one input nodes
        :param inputs: array of Tensorflow tensors representing the output of each preceding layer
        :return: Tensorflow tensor of the computed layer results
        """
        SelectedInputs = tf.concat(values=[Inputs[LayerIndex][:, NodeIndex:NodeIndex + 1] for (LayerIndex, NodeIndex) in self.InputNodeCoordinates], axis=1)
        return self.Activation(tf.matmul(SelectedInputs, self.Kernel) + self.Bias)

class DirectEncodingModelTrainable(tf.keras.Model):
    """
    Tensorflow model that builds a (exclusively) feed-forward topology with custom set connection weights and node
    biases/activations from the supplied genotype in the constructor. The built Tensorflow model is fully compatible
    with the rest of the Tensorflow infrastructure and supports static-graph building, auto-gradient, etc
    """

    def __init__(self, Genotype, DataType, bRunEager):
        """
        Creates the trainable feed-forward Tensorflow model out of the supplied genotype with custom parameters
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :param dtype: Tensorflow datatype of the model
        :param run_eagerly: bool flag if model should be run eagerly (by CPU) or if static GPU graph should be build
        """
        super(DirectEncodingModelTrainable, self).__init__(trainable=True, dtype=DataType)
        self.run_eagerly = bRunEager

        Nodes, Connections, NodeDependencies = ProcessGenotype(Genotype)

        self.TopologyLevels = tuple(toposort(NodeDependencies))
        NodeCoordinates = CreateNodeCoordinates(self.TopologyLevels)

        self.CustomLayers = [[] for _ in range(len(self.TopologyLevels) - 1)]
        for LayerIndex in range(len(self.CustomLayers)):
            # Create NodeDependencies specific for the current layer and with joined keys (conn_outs) if the have the
            # same input values (conn_ins)
            LayerNodeDependencies = {Node: NodeDependencies[Node] for Node in self.TopologyLevels[LayerIndex + 1]}
            JoinedLayerNodeDependencies = JoinKeys(LayerNodeDependencies)

            for JoinedNodes, JoinedNodesInput in JoinedLayerNodeDependencies.items():
                JoinedNodes = tuple(JoinedNodes)

                Activation = Nodes[JoinedNodes[0]][1]
                # Assert that all nodes for which the same CustomWeightAndInputLayer is created have the same activation
                assert all(Nodes[Node][1] == Activation for Node in JoinedNodes)

                InputNodeCoordinates = [NodeCoordinates[Node] for Node in JoinedNodesInput]

                # Create custom kernel weight matrix from connection weights supplied in genotype
                KernelWeights = np.empty(shape=(len(InputNodeCoordinates), len(JoinedNodes)), dtype=DataType.as_numpy_dtype)
                for ColumnIndex in range(KernelWeights.shape[1]):
                    for RowIndex in range(KernelWeights.shape[0]):
                        Weight = Connections[joined_nodes[ColumnIndex]][JoinedNodesInput[RowIndex]]
                        KernelWeights[RowIndex, ColumnIndex] = Weight

                # Create custom bias weight matrix from bias weights supplied in genotype
                BiasWeights = np.empty(shape=(len(JoinedNodes),), dtype=DataType.as_numpy_dtype)
                for NodeIndex in range(len(JoinedNodes)):
                    Weight = Nodes[JoinedNodes[NodeIndex]][0]
                    BiasWeights[NodeIndex] = Weight

                # Create nodes function for those joined nodes that have the same input as their value can be computed
                # in unison
                NodesFunction = CustomWeightAndInputLayerTrainable(Activation=Activation, KernelWeights=KernelWeights, BiasWeights=BiasWeights, InputNodeCoordinates=InputNodeCoordinates, DataType=DataType, bDynamic=bRunEager)
                self.CustomLayers[LayerIndex].append(NodesFunction)

    def call(self, Inputs) -> np.ndarray:
        """
        Model call of the DirectEncoding feed-forward model with arbitrarily connected nodes. The output of each layer
        is continually preserved, concatenated and then supplied together with the ouputs of all preceding layers to the
        next layer.
        :param inputs: Tensorflow or numpy array of one or multiple inputs to predict the output for
        :return: numpy array representing the predicted output to the input
        """
        Inputs = [tf.cast(x=Inputs, dtype=self.dtype)]
        for Layers in self.CustomLayers:
            LayerOut = None
            for NodesFunction in Layers:
                Out = NodesFunction(Inputs)
                LayerOut = Out if LayerOut is None else tf.concat(values=[LayerOut, Out], axis=1)
            Inputs.append(LayerOut)
        return Inputs[-1]

class CustomWeightAndInputLayerNontrainable:
    """
    Custom sparsely connected layer that allows for arbitrary input nodes from any layer, multiplying the inputs with
    the custom set kernel and bias. The arbitrariness of the input nodes is made possible through usage of coordinates
    specifying the layer and exact node in that layer for every input node. The layer is not trainable and even though
    it uses Tensorflow functionality is not compatible with the rest of the Tensorflow infrastructure.
    """
    def __init__(self, Activation, Kernel, Bias, InputNodeCoordinates, DataType):
        self.Activation = Activation
        self.Kernel = Kernel
        self.Bias = Bias
        self.InputNodeCoordinates = InputNodeCoordinates
        self.DataType = DataType

    def __call__(self, Inputs) -> tf.Tensor:
        """
        Layer call, whereby the size of the input nodes is determined and then accordingly multiplied with the kernel,
        added with the bias and the activation function is applied.
        :param inputs: array of Tensorflow or numpy tensors representing the output of each preceding layer
        :return: Tensorflow tensor of the computed layer results
        """
        if len(self.InputNodeCoordinates) >= 2:
            SelectedInputs = np.concatenate([Inputs[LayerIndex][:, NodeIndex:NodeIndex + 1] for (LayerIndex, NodeIndex) in self.InputNodeCoordinates], axis=1)
            return self.Activation(np.matmul(SelectedInputs, self.Kernel) + self.Bias)
        else:
            LayerIndex, NodeIndex = self.InputNodeCoordinates[0]
            SelectedInputs = Inputs[LayerIndex][:, NodeIndex:NodeIndex + 1]
            return self.Activation(np.matmul(SelectedInputs, self.Kernel) + self.Bias)

class DirectEncodingModelNontrainable:
    """
    Neural Network model that builds a (exclusively) feed-forward topology with custom set connection weights and node
    biases/activations from the supplied genotype in the constructor. The built model is non trainable and not
    compatible with the rest of the Tensorflow infrastructure.
    """
    def __init__(self, Genotype, DataType):
        """
        Creates the non-trainable feed-forward model out of the supplied genotype with custom parameters
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :param dtype: Tensorflow datatype of the model
        """
        self.DataType = DataType.as_numpy_dtype

        Nodes, Connections, NodeDependencies = ProcessGenotype(Genotype)

        self.TopologyLevels = tuple(toposort(NodeDependencies))
        NodeCoordinates = CreateNodeCoordinates(self.TopologyLevels)

        self.CustomLayers = [[] for _ in range(len(self.TopologyLevels) - 1)]
        for LayerIndex in range(len(self.CustomLayers)):
            # Create node_dependencies specific for the current layer and with joined keys (conn_outs) if the have the
            # same input values (conn_ins)
            LayerNodeDependencies = {Node: NodeDependencies[Node] for Node in self.TopologyLevels[LayerIndex + 1]}
            JoinedLayerNodeDependencies = JoinKeys(LayerNodeDependencies)

            for JoinedNodes, JoinedNodesInput in JoinedLayerNodeDependencies.items():
                JoinedNodes = tuple(JoinedNodes)

                activation = Nodes[JoinedNodes[0]][1]
                # Assert that all nodes for which the same CustomWeightAndInputLayer is created have the same activation
                assert all(Nodes[Node][1] == activation for Node in JoinedNodes)

                InputNodeCoordinates = [NodeCoordinates[Node] for Node in JoinedNodesInput]

                # Create custom kernel weight matrix from connection weights supplied in genotype
                Kernel = np.empty(shape=(len(InputNodeCoordinates), len(JoinedNodes)), dtype=self.dtype)
                for ColumnIndex in range(Kernel.shape[1]):
                    for RowIndex in range(Kernel.shape[0]):
                        Weight = Connections[JoinedNodes[ColumnIndex]][JoinedNodesInput[RowIndex]]
                        Kernel[RowIndex, ColumnIndex] = weight

                # Create custom bias weight matrix from bias weights supplied in genotype
                Bias = np.empty(shape=(len(JoinedNodes),), dtype=self.dtype)
                for NodeIndex in range(len(JoinedNodes)):
                    Weight = Nodes[JoinedNodes[NodeIndex]][0]
                    Bias[NodeIndex] = Weight

                # Create nodes function for those joined nodes that have the same input as their value can be computed
                # in unison
                NodesFunction = CustomWeightAndInputLayerNontrainable(Activation=Activation, Kernel=Kernel, Bias=Bias, InputNodeCoordinates=InputNodeCoordinates, DataType=DataType)
                self.CustomLayers[LayerIndex].append(NodesFunction)

    def predict(self, Inputs) -> tf.Tensor:
        """
        Model call of the DirectEncoding feed-forward model with arbitrarily connected nodes. The output of each layer
        is continually preserved, concatenated and then supplied together with the ouputs of all preceding layers to the
        next layer.
        :param inputs: Tensorflow or numpy array of one or multiple inputs to predict the output for
        :return: Tensorflow tensor representing the predicted output to the input
        """
        Inputs = [Inputs.astype(self.DataType)]
        for Layers in self.CustomLayers:
            LayerOut = None
            for NodesFunction in Layers:
                Out = NodesFunction(Inputs)
                LayerOut = Out if LayerOut is None else np.concatenate((LayerOut, Out), axis=1)
            Inputs.append(LayerOut)
        return Inputs[-1]