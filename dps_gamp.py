import os
import sys
from collections import namedtuple
import multiprocessing
from dps_ga import find_best_cast_history, standard_spells


class WorkerPipePool(object):
    NeighborIndexes = namedtuple('NeighborIndexes', 'n s e w')

    def __init__(self, sides):
        """Create enough pipes to connect sides^2 workers in a square grid."""
        self._sides = sides
        self._side_sq = sides * sides
        self._make_pipe_dict()

    def __getitem__(self, workerno):
        """Return a list of the pipes connecting the specified grid
        position to its north, south, east, and west neighbors."""
        neighbor_pipes = []
        for neighbor_idx in self._calc_neighbors(workerno):
            key = self._neighbor_tuple(workerno, neighbor_idx)
            little, big = self._pipes[key]
            if workerno < neighbor_idx:
                neighbor_pipes.append(little)
            else:
                neighbor_pipes.append(big)
        return neighbor_pipes

    def _calc_neighbors(self, pos):
        """Return a named tuple with the indexes of the neighbors of index
        `pos`."""
        # north is one row above, or wrap around to bottom row
        north = pos - self._sides
        if north < 0:
            north += self._side_sq
        # south is a row below or wrap around to top row
        south = (pos + self._sides) % self._side_sq
        # east is one column to the right, or wrap around to first column
        row = pos / self._sides
        east = pos + 1
        if east / self._sides > row:
            east = row * self._sides
        # west is one column to the left, or wrap around to last column
        west = pos - 1
        if west < 0:
            west = self._sides - 1
        elif west / self._sides < row:
            west += self._sides
        return WorkerPipePool.NeighborIndexes(north, south, east, west)

    def _make_pipe_dict(self):
        self._pipes = {}
        for i in xrange(self._side_sq):
            for neighbor_idx in self._calc_neighbors(i):
                t = self._neighbor_tuple(i, neighbor_idx)
                if not self._pipes.has_key(t):
                    self._pipes[t] = multiprocessing.Pipe()

    def _neighbor_tuple(self, i, j):
        if i < j:
            return (i, j)
        return (j, i)


class ImportExport(object):
    """Manage the flow of cast histories between worker processes."""

    def __init__(self, pipes):
        self._pipes = pipes

    def get_incoming(self):
        """Return any cast histories waiting to be read in pipes."""
        incoming = []
        for pipe in self._pipes:
            if pipe.poll():
                incoming += pipe.recv()
        print os.getpid(), 'received', len(incoming)
        sys.stdout.flush()


    def set_outgoing(self, ch_list):
        #print os.getpid(), 'sending', len(ch_list)
        sys.stdout.flush()
        for pipe in self._pipes:
            pipe.send(ch_list)


def worker(spells, result_q, pipes, num_runs, num_targets, time_limit, throttle):
    ie = ImportExport(pipes)
    best = []
    for run in xrange(num_runs):
        best = find_best_cast_history(
            spells,
            num_targets,
            time_limit,
            ie.get_incoming(),
            throttle)
        ie.set_outgoing(best)
    result_q.put(best)


def main():
    # there will be sides^2 worker processes
    sides = 3
    # seconds to sleep per iteration.
    THROTTLE = 0.001
    NUM_RUNS = 12
    NUM_TARGETS = 1
    ENC_TIME = 30.0
    procs = []
    pipes = WorkerPipePool(sides)

    # We must use multiprocessing.Manager.Queue rather than
    # multiprocessing.Queue because using the latter causes some of the worker
    # processes to hang on exit if they put cast histories in it.
    mgr = multiprocessing.Manager()
    q = mgr.Queue()
    for workerno in xrange(sides * sides):
        p = multiprocessing.Process(
            target=worker,
            args=(standard_spells, q, pipes[workerno], NUM_RUNS, NUM_TARGETS, ENC_TIME, THROTTLE))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    ch_dict = {}
    while not q.empty():
        ch_list = q.get()
        for ch in ch_list:
            ch_dict[ch.total_damage] = ch

    for dmg in sorted(ch_dict.iterkeys()):
        print '{:,.0f} {}'.format(dmg, ch_dict[dmg])


if __name__ == '__main__':
    main()

