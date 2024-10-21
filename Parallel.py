"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool

class ParallelEvaluator(object):
    def __init__(self, NumWorkers, EvaluationFunction, **kwargs):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.NumWorkers = NumWorkers
        self.EvaluationFunction = EvaluationFunction
        self.Timeout = kwargs.pop('Timeout') if 'Timeout' in kwargs.keys() else None
        self.Pool = Pool(NumWorkers)

    def __del__(self):
        self.Pool.close() # should this be terminate?
        self.Pool.join()

    def Evaluate(self, Genomes, Config):
        Jobs = []
        for IgnoredGenomeID, Genome in Genomes:
            Jobs.append(self.Pool.apply_async(self.EvaluationFunction, (Genome, Config)))
        # assign the fitness back to each genome
        for Job, (IgnoredGenomeID, Genome) in zip(Jobs, Genomes):
            Genome.Fitness = Job.get(timeout=self.Timeout)


class ParallelEvaluatorTrainingSets(object):
    def __init__(self, NumWorkers, EvaluationFunction, **kwargs):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.NumWorkers = NumWorkers
        self.EvaluationFunction = EvaluationFunction
        self.Timeout = kwargs.pop('Timeout') if 'Timeout' in kwargs.keys() else None
        self.kwargs = kwargs
        self.Pool = Pool(NumWorkers)

    def __del__(self):
        self.Pool.close() # should this be terminate?
        self.Pool.join()

    def Evaluate(self, Genomes, Config):
        Jobs = []
        for IgnoredGenomeID, Genome in Genomes:
            kwargs = self.kwargs
            #print('--------------------------------------------------------')
            #print(Genome)

            Jobs.append(self.Pool.apply_async(self.EvaluationFunction, (Genome, Config), kwargs))
        # assign the fitness back to each genome
        for Job, (IgnoredGenomeID, Genome) in zip(Jobs, Genomes):
            Genome.Fitness = Job.get(timeout=self.Timeout)