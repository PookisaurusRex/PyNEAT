"""
Distributed evaluation of Genomes.
About compute nodes:
The primary node (=the node which creates and mutates Genomes) and the secondary
nodes (=the nodes which evaluate Genomes) can execute the same script. The
role of a compute node is determined using the ``Mode`` argument of the
DistributedEvaluator. If the Mode is MODE_AUTO, the `HostIsLocal()` function
is used to check if the ``Address`` argument points to the LocalHost. If it does,
the compute node starts as a primary node, otherwise as a secondary node. If
``Mode`` is MODE_PRIMARY, the compute node always starts as a primary node. If
``Mode`` is MODE_SECONDARY, the compute node will always start as a secondary node.
There can only be one primary node per NEAT, but any number of secondary nodes.
The primary node will not evaluate any Genomes, which means you will always need
at least two compute nodes.
You can run any number of compute nodes on the same physical machine (or VM).
However, if a machine has both a primary node and one or more secondary nodes,
MODE_AUTO cannot be used for those secondary nodes - MODE_SECONDARY will need to be
specified.
NOTE: This module is in a **beta** state, and still *unstable* even in single-machine testing. Reliability is likely to
vary, including depending on the Python version and implementation (e.g., cpython vs pypy) in use and the likelihoods of
timeouts (due to machine and/or network slowness). In particular, while the code can try to reconnect between between
primary and secondary nodes, as noted in the `multiprocessing` documentation this may not work due to Data
loss/corruption. Note also that this module is not responsible for starting the script copies on the different compute
nodes, since this is very site/configuration-dependent.
Usage:
1. Import modules and define the evaluation logic (the eval_genome function).
  (After this, check for ``if __name__ == '__main__'``, and put the rest of
  the code inside the body of the statement.)
2. Load config and create a population - here, the variable ``p``.
3. If required, create and add reporters.
4. Create a ``DistributedEvaluator(addr_of_primary_node, b'some_password',
  eval_function, Mode=MODE_AUTO)`` - here, the variable ``de``.
5. Call ``de.start(exit_on_stop=True)``. The `start()` call will block on the
  secondary nodes and call `sys.exit(0)` when the NEAT evolution finishes. This
  means that the following code will only be executed on the primary node.
6. Start the evaluation using ``p.run(de.evaluate, number_of_generations)``.
7. Stop the secondary nodes using ``de.stop()``.
8. You are done. You may want to save the winning Genome or show some statistics.
See ``examples/xor/evolve-feedforward-distributed.py`` for a complete example.
Utility functions:
``HostIsLocal(HostName, Port=22)`` returns True if ``HostName`` points to
the local node/Host. This can be used to check if a compute node will run as
a primary node or as a secondary node with MODE_AUTO.
``Chunked(Data, ChunkSize)``: splits Data into a list of chunks with at most
``ChunkSize`` elements.
"""
from __future__ import print_function

import socket
import sys
import time
import warnings

import queue

import multiprocessing
from multiprocessing import managers
from argparse import Namespace

# Some of this code is based on
# http://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing
# According to the website, the code is in the public domain
# ('public domain' links to unlicense.org).
# This means that we can use the code from this website.
# Thanks to Eli Bendersky for making his code open for use.


# modes to determine the role of a compute node
# the primary handles the evolution of the Genomes
# the secondary handles the evaluation of the Genomes
MODE_AUTO = 0  # auto-determine Mode
MODE_PRIMARY = MODE_MASTER = 1  # enforce primary Mode
MODE_SECONDARY = MODE_SLAVE = 2  # enforce secondary Mode

# states to determine whether the secondaries should shut down
_STATE_RUNNING = 0
_STATE_SHUTDOWN = 1
_STATE_FORCED_SHUTDOWN = 2


class ModeError(RuntimeError):
    """
    An exception raised when a Mode-specific method is being
    called without being in the Mode - either a primary-specific method
    called by a secondary node or a secondary-specific method called by a primary node.
    """
    pass


def HostIsLocal(HostName, Port=22): # no Port specified, just use the ssh Port
    """
    Returns True if the HostName points to the LocalHost, otherwise False.
    """
    HostName = socket.getfqdn(HostName)
    if HostName in ("LocalHost", "0.0.0.0", "127.0.0.1", "1.0.0.127.in-Address.arpa", "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa"):
        return True
    LocalHost = socket.gethostname()
    if HostName == LocalHost:
        return True
    LocalAddress = socket.getaddrinfo(LocalHost, Port)
    TargetAddress = socket.getaddrinfo(HostName, Port)
    for (IgnoredFamily, IgnoredSocketType, IgnoredProto, IgnoredCanonName, SocketAddress) in LocalAddress:
        for (IgnoredRFamily, IgnoredRSocketType, IgnoredRProto, IgnoredRCanonName, TargetSocketAddress) in TargetAddress:
            if TargetSocketAddress[0] == SocketAddress[0]:
                return True
    return False


def DetermineMode(Address, Mode):
    """
    Returns the Mode which should be used.
    If Mode is MODE_AUTO, this is determined by checking if 'Address' points to the
    local Host. If it does, return MODE_PRIMARY, else return MODE_SECONDARY.
    If Mode is either MODE_PRIMARY or MODE_SECONDARY,
    return the 'Mode' argument. Otherwise, a ValueError is raised.
    """
    if isinstance(Address, tuple):
        Host = Address[0]
    elif isinstance(Address, bytes):
        Host = Address
    else:
        raise TypeError("'Address' needs to be a tuple or an bytestring!")
    if Mode == MODE_AUTO:
        if HostIsLocal(Host):
            return MODE_PRIMARY
        return MODE_SECONDARY
    elif Mode in (MODE_SECONDARY, MODE_PRIMARY):
        return Mode
    else:
        raise ValueError("Invalid Mode {!r}!".format(Mode))

def Chunked(Data, ChunkSize):
    """
    Returns a list of chunks containing at most ``ChunkSize`` elements of Data.
    """
    if ChunkSize < 1:
        raise ValueError("Chunksize must be at least 1!")
    if int(ChunkSize) != ChunkSize:
        raise ValueError("Chunksize needs to be an integer")
    res = []
    cur = []
    for e in Data:
        cur.append(e)
        if len(cur) >= ChunkSize:
            res.append(cur)
            cur = []
    if cur:
        res.append(cur)
    return res


class _ExtendedManager(object):
    """A class for managing the multiprocessing.managers.SyncManager"""
    __safe_for_unpickling__ = True  # this may not be safe for unpickling, but this is required by pickle.

    def __init__(self, Address, AuthKey, Mode, bStart=False):
        self.Address = Address
        self.AuthKey = AuthKey
        self.Mode = DetermineMode(Address, Mode)
        self.Manager = None
        self._secondary_state = multiprocessing.managers.Value(int, _STATE_RUNNING)
        if bStart:
            self.Start()

    def __reduce__(self):
        """
        This method is used by pickle to serialize instances of this class.
        """
        return (self.__class__, (self.Address, self.AuthKey, self.Mode, True),)

    def start(self):
        """Starts or connects to the manager."""
        if self.Mode == MODE_PRIMARY:
            i = self._start()
        else:
            i = self._connect()
        self.Manager = i

    def stop(self):
        """Stops the manager."""
        self.Manager.shutdown()

    def set_secondary_state(self, value):
        """Sets the value for 'secondary_state'."""
        if value not in (_STATE_RUNNING, _STATE_SHUTDOWN, _STATE_FORCED_SHUTDOWN):
            raise ValueError(
                "State {!r} is invalid - needs to be one of _STATE_RUNNING, _STATE_SHUTDOWN, or _STATE_FORCED_SHUTDOWN".format(
                    value)
                )
        if self.Manager is None:
            raise RuntimeError("Manager not started")
        self.Manager.set_state(value)

    def _get_secondary_state(self):
        """
        Returns the value for 'secondary_state'.
        This is required for the manager.
        """
        return self._secondary_state

    def _get_manager_class(self, register_callables=False):
        """
        Returns a new 'Manager' subclass with registered methods.
        If 'register_callable' is True, defines the 'callable' arguments.
        """

        class _EvaluatorSyncManager(managers.BaseManager):
            """
            A custom BaseManager.
            Please see the documentation of `multiprocessing` for more
            information.
            """
            pass

        InQueue = queue.Queue()
        OutQueue = queue.Queue()
        namespace = Namespace()

        if register_callables:
            _EvaluatorSyncManager.register("get_inqueue", callable=lambda: InQueue,)
            _EvaluatorSyncManager.register("get_outqueue", callable=lambda: OutQueue,)
            _EvaluatorSyncManager.register("get_state", callable=self._get_secondary_state,)
            _EvaluatorSyncManager.register("set_state", callable=lambda v: self._secondary_state.set(v),)
            _EvaluatorSyncManager.register("get_namespace", callable=lambda: namespace,)
        else:
            _EvaluatorSyncManager.register("get_inqueue",)
            _EvaluatorSyncManager.register("get_outqueue",)
            _EvaluatorSyncManager.register("get_state",)
            _EvaluatorSyncManager.register("set_state",)
            _EvaluatorSyncManager.register("get_namespace",)
        return _EvaluatorSyncManager

    def _connect(self):
        """Connects to the manager."""
        cls = self._get_manager_class(register_callables=False)
        ins = cls(address=self.Address, AuthKey=self.AuthKey)
        ins.connect()
        return ins

    def _start(self):
        """Starts the manager."""
        cls = self._get_manager_class(register_callables=True)
        ins = cls(address=self.Address, AuthKey=self.AuthKey)
        ins.start()
        return ins

    @property
    def secondary_state(self):
        """Whether the secondary nodes should still process elements."""
        Value = self.Manager.get_state()
        return Value.get()

    def get_inqueue(self):
        """Returns the InQueue."""
        if self.Manager is None:
            raise RuntimeError("Manager not started")
        return self.Manager.get_inqueue()

    def get_outqueue(self):
        """Returns the OutQueue."""
        if self.Manager is None:
            raise RuntimeError("Manager not started")
        return self.Manager.get_outqueue()

    def get_namespace(self):
        """Returns the namespace."""
        if self.Manager is None:
            raise RuntimeError("Manager not started")
        return self.Manager.get_namespace()


class DistributedEvaluator(object):
    """An evaluator working across multiple machines"""
    def __init__(self, Address, AuthKey, eval_function, secondary_chunksize=1, num_workers=None, WorkerTimeout=60, Mode=MODE_AUTO,):
        """
        ``Address`` should be a tuple of (HostName, Port) pointing to the machine
        running the DistributedEvaluator in primary Mode. If Mode is MODE_AUTO,
        the Mode is determined by checking whether the HostName points to this
        Host or not.
        ``AuthKey`` is the password used to restrict access to the manager; see
        ``Authentication Keys`` in the `multiprocessing` manual for more information.
        All DistributedEvaluators need to use the same AuthKey. Note that this needs
        to be a `bytes` object for Python 3.X, and should be in 2.7 for compatibility
        (identical in 2.7 to a `str` object).
        ``eval_function`` should take two arguments (a Genome object and the
        configuration) and return a single float (the Genome's fitness).
        'secondary_chunksize' specifies the number of Genomes that will be sent to
        a secondary at any one time.
        ``num_workers`` is the number of child processes to use if in secondary
        Mode. It defaults to None, which means `multiprocessing.cpu_count()`
        is used to determine this value. If 1 in a secondary node, the process creating
        the DistributedEvaluator instance will also do the evaulations.
        ``WorkerTimeout`` specifies the timeout (in seconds) for a secondary node
        getting the results from a worker subprocess; if None, there is no timeout.
        ``Mode`` specifies the Mode to run in; it defaults to MODE_AUTO.
        """
        self.Address = Address
        self.AuthKey = AuthKey
        self.eval_function = eval_function
        self.secondary_chunksize = secondary_chunksize
        self.slave_chunksize = secondary_chunksize # backward compatibility
        if num_workers:
            self.num_workers = num_workers
        else:
            try:
                self.num_workers = max(1, multiprocessing.cpu_count())
            except (RuntimeError, AttributeError): # pragma: no cover
                print("multiprocessing.cpu_count() gave an error; assuming 1", file=sys.stderr)
                self.num_workers = 1
        self.WorkerTimeout = WorkerTimeout
        self.Mode = DetermineMode(self.Address, Mode)
        self.ExtendedManager = _ExtendedManager(self.Address, self.AuthKey, Mode=self.Mode, start=False)
        self.InQueue = None
        self.OutQueue = None
        self.Namespace = None
        self.started = False

    def __getstate__(self):
        """Required by the pickle protocol."""
        # we do not actually save any state, but we need __getstate__ to be
        # called.
        return True  # return some nonzero value

    def __setstate__(self, state):
        """Called when instances of this class are unpickled."""
        self._set_shared_instances()

    def is_primary(self):
        """Returns True if the caller is the primary node"""
        return self.Mode == MODE_PRIMARY

    def is_master(self): # pragma: no cover
        """Returns True if the caller is the primary (master) node"""
        warnings.warn("Use is_primary, not is_master", DeprecationWarning)
        return self.is_primary()

    def start(self, exit_on_stop=True, secondary_wait=0, reconnect=False):
        """
        If the DistributedEvaluator is in primary Mode, starts the manager
        process and returns. In this case, the ``exit_on_stop`` argument will
        be ignored.
        If the DistributedEvaluator is in secondary Mode, it connects to the manager
        and waits for tasks.
        If in secondary Mode and ``exit_on_stop`` is True, sys.exit() will be called
        when the connection is lost.
        ``secondary_wait`` specifies the time (in seconds) to sleep before actually
        starting when in secondary Mode.
        If 'reconnect' is True, the secondary nodes will try to reconnect when
        the connection is lost. In this case, sys.exit() will only be called
        when 'exit_on_stop' is True and the primary node send a forced shutdown
        command.
        """
        if self.started:
            raise RuntimeError("DistributedEvaluator already started!")
        self.started = True
        if self.Mode == MODE_PRIMARY:
            self._start_primary()
        elif self.Mode == MODE_SECONDARY:
            time.sleep(secondary_wait)
            self._start_secondary()
            self._secondary_loop(reconnect=reconnect)
            if exit_on_stop:
                sys.exit(0)
        else:
            raise ValueError("Invalid Mode {!r}!".format(self.Mode))

    def stop(self, wait=1, shutdown=True, force_secondary_shutdown=False):
        """
        Stops all secondaries.
        'wait' specifies the time (in seconds) to wait before shutting down the
        manager or returning.
        If 'shutdown', shutdown the manager.
        If 'force_secondary_shutdown', shutdown the secondary nodes even if
        they are started with 'reconnect=True'.
        """
        if self.Mode != MODE_PRIMARY:
            raise ModeError("Not in primary Mode!")
        if not self.started:
            raise RuntimeError("Not yet started!")
        if force_secondary_shutdown:
            state = _STATE_FORCED_SHUTDOWN
        else:
            state = _STATE_SHUTDOWN
        self.ExtendedManager.set_secondary_state(state)
        time.sleep(wait)
        if shutdown:
            self.ExtendedManager.stop()
        self.started = False
        self.InQueue = self.OutQueue = self.Namespace = None

    def _start_primary(self):
        """Start as the primary"""
        self.ExtendedManager.start()
        self.ExtendedManager.set_secondary_state(_STATE_RUNNING)
        self._set_shared_instances()

    def _start_secondary(self):
        """Start as a secondary."""
        self.ExtendedManager.start()
        self._set_shared_instances()

    def _set_shared_instances(self):
        """Sets attributes from the shared instances."""
        self.InQueue = self.ExtendedManager.get_inqueue()
        self.OutQueue = self.ExtendedManager.get_outqueue()
        self.Namespace = self.ExtendedManager.get_namespace()

    def _reset_em(self):
        """Resets self.ExtendedManager and the shared instances."""
        self.ExtendedManager = _ExtendedManager(self.Address, self.AuthKey, Mode=self.Mode, start=False)
        self.ExtendedManager.start()
        self._set_shared_instances()

    def _secondary_loop(self, reconnect=False):
        """The worker loop for the secondary nodes."""
        if self.num_workers > 1:
            pool = multiprocessing.Pool(self.num_workers)
        else:
            pool = None
        should_reconnect = True
        while should_reconnect:
            i = 0
            running = True
            try:
                self._reset_em()
            except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                continue
            while running:
                i += 1
                if i % 5 == 0:
                    # for better performance, only check every 5 cycles
                    try:
                        state = self.ExtendedManager.secondary_state
                    except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                        if not reconnect:
                            raise
                        else:
                            break
                    if state == _STATE_FORCED_SHUTDOWN:
                        running = False
                        should_reconnect = False
                    elif state == _STATE_SHUTDOWN:
                        running = False
                    if not running:
                        continue
                try:
                    tasks = self.InQueue.get(block=True, timeout=0.2)
                except queue.Empty:
                    continue
                except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                    break
                except (managers.RemoteError, multiprocessing.ProcessError) as e:
                    if ('Empty' in repr(e)) or ('TimeoutError' in repr(e)):
                        continue
                    if (('EOFError' in repr(e)) or ('PipeError' in repr(e)) or ('AuthenticationError' in repr(e))): # Second for Python 3.X, Third for 3.6+
                        break
                    raise
                if pool is None:
                    res = []
                    for GenomeID, Genome, config in tasks:
                        fitness = self.eval_function(Genome, config)
                        res.append((GenomeID, fitness))
                else:
                    genome_ids = []
                    jobs = []
                    for GenomeID, Genome, config in tasks:
                        genome_ids.append(GenomeID)
                        jobs.append(pool.apply_async(self.eval_function, (Genome, config)))
                    results = [job.get(timeout=self.WorkerTimeout) for job in jobs]
                    res = zip(genome_ids, results)
                try:
                    self.OutQueue.put(res)
                except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                    break
                except (managers.RemoteError, multiprocessing.ProcessError) as e:
                    if ('Empty' in repr(e)) or ('TimeoutError' in repr(e)):
                        continue
                    if (('EOFError' in repr(e)) or ('PipeError' in repr(e)) or ('AuthenticationError' in repr(e))): # Second for Python 3.X, Third for 3.6+
                        break
                    raise
            if not reconnect:
                should_reconnect = False
                break
        if pool is not None:
            pool.terminate()

    def evaluate(self, Genomes, config):
        """
        Evaluates the Genomes.
        This method raises a ModeError if the
        DistributedEvaluator is not in primary Mode.
        """
        if self.Mode != MODE_PRIMARY:
            raise ModeError("Not in primary Mode!")
        tasks = [(GenomeID, Genome, config) for GenomeID, Genome in Genomes]
        id2genome = {GenomeID: Genome for GenomeID, Genome in Genomes}
        tasks = Chunked(tasks, self.secondary_chunksize)
        n_tasks = len(tasks)
        for task in tasks:
            self.InQueue.put(task)
        tresults = []
        while len(tresults) < n_tasks:
            try:
                sr = self.OutQueue.get(block=True, timeout=0.2)
            except (queue.Empty, managers.RemoteError):
                continue
            tresults.append(sr)
        results = []
        for sr in tresults:
            results += sr
        for GenomeID, fitness in results:
            Genome = id2genome[GenomeID]
            Genome.fitness = fitness
