"""Threaded evaluation of genomes"""
from __future__ import print_function

import warnings

try:
    import threading
except ImportError: # pragma: no cover
    import dummy_threading as threading
    HAVE_THREADS = False
else:
    HAVE_THREADS = True

import queue


class ThreadedEvaluator(object):
    """
    A threaded genome evaluator.
    Useful on python implementations without GIL (Global Interpreter Lock).
    """
    def __init__(self, NumWorkers, EvaluationFunction):
        """
        eval_function should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        """
        self.NumWorkers = NumWorkers
        self.EvaluationFunction = EvaluationFunction
        self.Workers = []
        self.bWorking = False
        self.InQueue = queue.Queue()
        self.OutQueue = queue.Queue()

        if not HAVE_THREADS: # pragma: no cover
            warnings.warn("No threads available; use ParallelEvaluator, not ThreadedEvaluator")

    def __del__(self):
        """
        Called on deletion of the object. We stop our workers here.
        WARNING: __del__ may not always work!
        Please stop the threads explicitly by calling self.stop()!
        TODO: ensure that there are no reference-cycles.
        """
        if self.bWorking:
            self.Stop()

    def Start(self):
        """Starts the worker threads"""
        if self.bWorking:
            return
        self.bWorking = True
        for Idx in range(self.NumWorkers):
            Worker = threading.Thread(name="Worker Thread #{Idx}".format(Idx=Idx), target=self._Worker,)
            Worker.daemon = True
            Worker.start()
            self.Workers.append(Worker)

    def Stop(self):
        """Stops the worker threads and waits for them to finish"""
        self.bWorking = False
        for Worker in self.Workers:
            Worker.join()
        self.Workers = []

    def _Worker(self):
        """The worker function"""
        while self.bWorking:
            try:
                GenomeID, Genome, Config = self.InQueue.get(block=True, timeout=0.2,)
            except queue.Empty:
                continue
            Fitness = self.EvaluationFunction(Genome, GenomeID, Config)
            self.OutQueue.put((GenomeID, Genome, Fitness))

    def Evaluate(self, Genomes, Config):
        """Evaluate the genomes"""
        if not self.bWorking:
            self.Start()
        WorkRemainingCounter = 0
        for GenomeID, Genome in Genomes:
            WorkRemainingCounter += 1
            self.InQueue.put((GenomeID, Genome, Config))
        # assign the fitness back to each genome
        while WorkRemainingCounter > 0:
            WorkRemainingCounter -= 1
            IgnoredGenomeID, Genome, Fitness = self.OutQueue.get()
            Genome.Fitness = Fitness
