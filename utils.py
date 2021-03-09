import math
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt
from ribs.archives import GridArchive#, SlidingBoundaryArchive

EPSILON = np.finfo(float).eps


def entropy(distribution, base):
    if base == 2:
        return -distribution @ torch.log2(distribution)
    elif base == math.e:
        return -distribution @ torch.log(distribution)
    elif base == 10:
        return -distribution @ torch.log10(distribution)
    sys.exit("Unsupported base. Please choose {}, {}, or {}."
             .format(2, "e", 10))


def plot_entropies(entropies):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for axis in axes:
        axis.set_xlabel('Steps')
        axis.set_ylabel('Value')

    axes[0].plot(entropies)
    axes[0].set_title('H')
    axes[1].plot(np.diff(entropies))
    axes[1].set_title('d/dH(H)')
    fig.tight_layout()
    plt.show()



class FlexArchive(GridArchive):
    def __init__(self, *args, **kwargs):
        self.score_hists = {}
        super().__init__(*args, **kwargs)

    def update_elite(self, behavior_values, obj):
        index = self._get_index(behavior_values)
        self.update_elite_idx(index, obj)

    def update_elite_idx(self, index, obj):
        if index not in self.score_hists:
            self.score_hists[index] = []
        score_hists = self.score_hists[index]
        score_hists.append(obj)
        obj = np.mean(score_hists)
        self._solutions[index][2] = obj
        self._objective_values[index] = obj

        while len(score_hists) > 500:
            score_hists.pop(0)

    def add(self, solution, objective_value, behavior_values):
        index = self._get_index(behavior_values)

        if index in self.score_hists:
            self.score_hists[index] = [objective_value]

        return super().add(solution, objective_value, behavior_values)
