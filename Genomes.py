"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function


from itertools import count
from random import choice, random, shuffle

import sys

from NEAT.Activations import ActivationFunctionSet
from NEAT.Aggregations import AggregationFunctionSet
from NEAT.Config import ConfigParameter, WritePrettyParameters
from NEAT.Genes import DefaultConnectionGene, DefaultNodeGene