from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np

import NEAT.NeuralNetwork
from NEAT.Config import Config
from NEAT.Population import Population, CompleteExtinctionException
from NEAT.Genome import DefaultGenome
from NEAT.Reproduction import DefaultReproduction
from NEAT.Stagnation import DefaultStagnation
from NEAT.Reports import StdOutReporter
from NEAT.Species import DefaultSpeciesSet
from NEAT.Statistics import StatisticsReporter
from NEAT.Parallel import ParallelEvaluator, ParallelEvaluatorTrainingSets
from NEAT.Distributed import DistributedEvaluator, HostIsLocal
#from NEAT.Threads import ThreadedEvaluator
from NEAT.Checkpoints import Checkpointer

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

InputKeyNames = {}

def DrawSpecies(Statistics, bView=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    SpeciesSizes = Statistics.GetSpeciesSizes()
    NumGenerations = len(SpeciesSizes)
    Curves = np.array(SpeciesSizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(NumGenerations), *Curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if bView:
        plt.show()

    plt.close()

def DrawNetwork(Config, Genome, bView=False, Filename=None, NodeNames=None, bShowDisabled=True, bPruneUnused=False, NodeColors=None, fmt='svg'):
    """ Receives a Genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return
    if NodeNames is None:
        NodeNames = {}
    assert type(NodeNames) is dict
    if NodeColors is None:
        NodeColors = {}
    assert type(NodeColors) is dict
    graph_attrs= {'overlap':'false','rankdir':'TB','engine':'dot','splines':'ortho','concentrate':'True'}
    #graph_attrs= {'overlap':'false','rankdir':'LR','engine':'dot','splines':'ortho'}
    node_attrs = { 'shape': 'record', 'fontsize': '9', 'height': '0.2', 'width': '0.2'}
    NetworkVizualization = graphviz.Digraph(format=fmt, graph_attr=graph_attrs, node_attr=node_attrs)
    Inputs = set()
    graph_attrs= {'overlap':'False','rankdir':'LR'}
    graph_attrs= {'overlap':'false','rankdir':'LR','engine':'dot','pack':'True','packmode':'node','splines':'ortho','concentrate':'True','fixedsize':'True'}
    InputSubgraph = graphviz.Digraph(format=fmt, graph_attr=graph_attrs, node_attr=node_attrs)
    InputSubgraph.attr(rank='same')
    for Key in Config.GenomeConfig.InputKeys:
        Inputs.add(Key)
        Name = NodeNames.get(Key, str(Key))
        input_attrs = {'style': 'filled', 'shape': 'record', 'fillcolor': NodeColors.get(Key, 'lightgray')}
        NetworkVizualization.node(Name, _attributes=input_attrs)
        InputSubgraph.node(Name, _attributes=input_attrs)
    NetworkVizualization.subgraph(InputSubgraph)
    Outputs = set()
    OutputSubgraph = graphviz.Digraph(format=fmt, graph_attr=graph_attrs, node_attr=node_attrs)
    OutputSubgraph.attr(rank='same')
    for Key in Config.GenomeConfig.OutputKeys:
        Outputs.add(Key)
        Name = NodeNames.get(Key, str(Key))
        Activation = Genome.Nodes[Key].Activation
        Aggregation = Genome.Nodes[Key].Aggregation
        Response = str(round(Genome.Nodes[Key].Response, 5))
        Bias = str(round(Genome.Nodes[Key].Bias, 5))
        Label = '{'+Name+'|Aggregation: '+Aggregation+'\l|Activation: '+Activation+'\l|Response: '+Response+'\l|Bias: '+Bias+'\l}'
        node_attrs = {'style': 'filled', 'fillcolor': NodeColors.get(Key, 'lightblue'),'label':Label}
        NetworkVizualization.node(Name, _attributes=node_attrs)
        OutputSubgraph.node(Name, _attributes=input_attrs)
    NetworkVizualization.subgraph(OutputSubgraph)
    if bPruneUnused:
        Connections = set(GeneConnection.Key for GeneConnection in Genome.Connections.values() if GeneConnection.bEnabled or bShowDisabled)
        UsedNodes = copy.copy(Outputs)
        bPending = copy.copy(Outputs)
        while bPending:
            #print(bPending, UsedNodes)
            NewPending = set()
            for InNode, OutNode in Connections:
                if OutNode in bPending and InNode not in UsedNodes:
                    NewPending.add(InNode)
                    UsedNodes.add(InNode)
            bPending = NewPending
    else:
        UsedNodes = set(Genome.Nodes.keys())
    HiddenNodeCount = 0
    for Node in UsedNodes:
        if Node in Inputs or Node in Outputs:
            continue
        HiddenNodeCount+=1
        Name = 'Hidden '+str(HiddenNodeCount)
        Activation = Genome.Nodes[Node].Activation
        Aggregation = Genome.Nodes[Node].Aggregation
        Response = str(round(Genome.Nodes[Node].Response, 5))
        Bias = str(round(Genome.Nodes[Node].Bias, 5))
        Label = '{'+Name+'|Aggregation: '+Aggregation+'\l|Activation: '+Activation+'\l|Response: '+Response+'\l|Bias: '+Bias+'\l}'
        attrs = {'style': 'filled', 'fillcolor': NodeColors.get(Node, 'white'), 'label':Label}
        NetworkVizualization.node(str(Node), _attributes=attrs)
    for GeneConnection in Genome.Connections.values():
        if GeneConnection.bEnabled or bShowDisabled:
            Input, Output = GeneConnection.Key
            if Input not in UsedNodes or Output not in UsedNodes:
                continue
            Node1 = NodeNames.get(Input, str(Input))
            Node2 = NodeNames.get(Output, str(Output))
            Style = 'solid' if GeneConnection.bEnabled else 'dotted'
            Color = 'green' if GeneConnection.Weight > 0 else 'red'
            Width = str(0.1 + abs(GeneConnection.Weight / 10.0))
            NetworkVizualization.edge(Node1, Node2, _attributes={'style': Style, 'color': Color, 'penwidth': Width})
    NetworkVizualization.render(Filename, view=bView)

    return NetworkVizualization

def plot_stats(Statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statiStatisticsstics.most_fit_genomes))
    best_fitness = [c.fitness for c in Statistics.most_fit_genomes]
    avg_fitness = np.array(Statistics.get_fitness_mean())
    stdev_fitness = np.array(Statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()