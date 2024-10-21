import types
import warnings
import numpy as np
from operator import mul
from functools import reduce

def ProductAggregation(x):  # note: `x` is a list or other iterable
    return np.nanprod(x)

def SumAggregation(x):
    return np.nansum(x)

def MaxAggregation(x):
    return np.nanmax(x)

def MinAggregation(x):
    return np.nanmin(x)

def MaxAbsAggregation(x):
    return np.nanmax(np.abs(x))

def MinAbsAggregation(x):
    return np.nanmin(np.abs(x))

def MedianAggregation(x):
    return np.nanmedian(x)

def MeanAggregation(x):
    return np.nanmean(x)

class InvalidAggregationFunction(TypeError):
    pass

def ValidateAggregation(Function):  # TODO: Recognize when need `reduce`
    if not isinstance(Function, (types.BuiltinFunctionType, types.FunctionType, types.LambdaType)):
        raise InvalidAggregationFunction("A function object is required.")
    if not (Function.__code__.co_argcount >= 1):
        raise InvalidAggregationFunction("A function taking at least one argument is required")

AggregationFunctionsDictionary = \
{ 
    "product": ProductAggregation,
    "sum": SumAggregation,
    "max": MaxAggregation,
    "min": MinAggregation,
    "maxabs": MaxAbsAggregation,
    "minabs": MinAbsAggregation,
    "median": MedianAggregation,
    "mean": MeanAggregation,
}

class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""
    def __init__(self):
        self.AggregationFunctions = AggregationFunctionsDictionary

    def Add(self, Name, Function):
        ValidateAggregation(Function)
        self.AggregationFunctions[Name] = Function

    def Get(self, Name):
        Function = self.AggregationFunctions.get(Name)
        if Function is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(Name))
        return Function

    def __getitem__(self, index):
        warnings.warn("Use Get, not indexing ([{!r}]), for aggregation functions".format(index), DeprecationWarning)
        return self.Get(index)

    def IsValid(self, Name):
        return (Name in self.AggregationFunctions)