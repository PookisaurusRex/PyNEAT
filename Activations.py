import math
import types
import numpy as np
import tensorflow as tf

def Sigmoid(x):
    return float(1/(1 + np.exp(-x)))

def SigmoidActivation(x):
    return tf.keras.activations.sigmoid(x)

def HardSigmoid(x):
    return float(max(0, min(1, (x * 0.2) + 0.5)))

def HardSigmoidActivation(x):
    return tf.keras.activations.hard_sigmoid(x)

def Tanh(x):
    return np.tanh(x)

def TanhActivation(x):
    return tf.keras.activations.tanh(x)

def AbsVal(x):
    return abs(x)

def AbsoluteValueActivation(x):
    return tf.abs(x)

def Gauss(x):
    return np.exp(-5.0 * x**2)

def GaussActivation(x):
    return tf.exp(-5.0 * x**2)

def Linear(x):
    return x

def Clamped(x):
    return max(-1.0, min(1.0, x))

def sin(x):
    return np.sin(x)

def SineActivation(x):
    return tf.sin(x)

def sinh(x):
    return np.sinh(x)

def HyperbolicSineActivation(x):
    return tf.sinh(x)

def asinh(x):
    return np.arcsinh(x)

def HyperbolicArcSineActivation(x):
    return tf.math.asinh(x)

def cos(x):
    return np.cos(x)

def CosineActivation(x):
    return tf.cos(x)

def cosh(x):
    return np.cosh(x)

def HyperbolicCosineActivation(x):
    return tf.cosh(x)

def sech(x):
    return 1.0 / np.cosh(x)

def HyperbolicSecantActivation(x):
    return 1.0 / tf.cosh(x)

def ReLU(x):
    return 0.0 if x < 0.0 else x

def ReLUActivation(x):
    return tf.keras.activations.relu(x)

def ELU(x):
    return (np.exp(x) - 1) if x < 0.0 else x

def ELUActivation(x):
    return tf.keras.activations.elu(x)

def SELU(x):
    alpha=1.67326324 
    scale=1.05070098
    return (scale * alpha * (np.exp(x) - 1)) if x < 0.0 else (scale * x)

def SELUActivation(x):
    return tf.keras.activations.selu(x)

def SoftPlus(x):
    return np.log(np.exp(x) + 1)

def SoftPlusActivation(x):
    return tf.keras.activations.softplus(x)

def Swish(x):
    return x / (1 + np.exp(-x))

def SoftSign(x):
    return x / (abs(x) + 1)

def SoftSignActivation(x):
    return tf.keras.activations.softsign(x)

def Log(x):
    x = max(1e-7, x)
    return np.log(x)

def LogActivation(x):
    x = max(1e-7, x)
    return math.log(x)

def Inverse(x):
    try:
        x = 1.0 / x
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return x

def SquareActivation(x):
    return x ** 2

def CubeActivation(x):
    return x ** 3

def LeLUActivation(x):
    Leakiness = 0.005
    return (x if x > 0.0 else Leakiness * x)

def HatActivation(x):
    return max(0.0, 1 - abs(x))

def Bent(x):
    return ((np.sqrt(x**2 + 1) - 1) / 2.0) + x

def Sinc(x):
    return np.sin(x)/x

def NormSinc(x):
    return np.sin(np.pi * x) / (np.pi * x)

class InvalidActivationFunction(TypeError):
    pass

def ValidateActivation(Function):
    if not isinstance(Function, (types.BuiltinFunctionType, types.FunctionType, types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")
    if Function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")

ActivationFunctionsDictionary = \
{ 
    "sigmoid": Sigmoid,
    "hardsig": HardSigmoid,
    "tanh": Tanh,
    "abs": AbsVal,
    "gauss": Gauss,
    "linear": Linear,
    "clamped": Clamped,
    "inverse": Inverse,
    "sin": sin,
    "sinh": sinh,
    "asinh": asinh,
    "cos": cos,
    "cosh": cosh,
    "sech": sech,
    "relu": ReLU,
    "elu": ELU,
    "selu": SELU,
    "softplus": SoftPlus,
    "softsign": SoftSign,
    "log": Log,
    "square": SquareActivation,
    "cube": CubeActivation,
    "hat": HatActivation,
    "swish":Swish,
    "bent":Bent,
    "sinc":Sinc,
    "normsinc":NormSinc
}

class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """
    def __init__(self):
        self.ActivationFunctions = ActivationFunctionsDictionary

    def Add(self, Name, Function):
        ValidateActivation(Function)
        self.ActivationFunctions[Name] = Function

    def Get(self, Name):
        Function = self.ActivationFunctions.get(Name)
        if Function is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(Name))
        return Function

    def IsValid(self, Name):
        return Name in self.ActivationFunctions


if __name__ == "__main__":
    import timeit
    import random
    import tensorflow as tf
    #x = random.uniform(-100.0, 100.0)
    x = random.random()
    print(x)
    NumpyTime = timeit.timeit('ActivationFunc(y)', setup="from __main__ import Log as ActivationFunc; import random; y={}".format(x), number=20000)
    TensorFlowTime = timeit.timeit('ActivationFunc(y)', setup="from __main__ import LogActivation as ActivationFunc; import random; import tensorflow as tf; y = tf.constant({}, dtype=tf.float32)".format(x), number=20000)
    print("Numpy: {} | Value: {}".format(round(NumpyTime, 5), Log(x)))
    print("Tensorflow: {} | Value: {}".format(round(TensorFlowTime, 5), LogActivation(tf.constant(x, dtype=tf.float32))))