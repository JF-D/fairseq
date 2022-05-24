import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from collections import defaultdict
from itertools import groupby


class GenPreemptionTrace:
    def __init__(self, world_size, rank, prob=0.05, iters_per_preemption=5, sec_per_iter=0.2, seed=12345):
        self.world_size = world_size
        self.rank = rank
        self.prob = prob
        self.iters_per_preemption = iters_per_preemption
        self.sec_per_iter = sec_per_iter

        self.seed = seed
        random.seed(self.seed)

        self.preemption_trace = None

    def gen_trace(self):
        self.total_ticks_per_rank = math.ceil(3600 / self.sec_per_iter / self.iters_per_preemption)
        total_preemptions = math.ceil(3600 * self.world_size * self.prob / self.sec_per_iter / self.iters_per_preemption)

        self.preempted = random.sample(list(range(self.world_size * self.total_ticks_per_rank // 2)), total_preemptions)
        for i in range(len(self.preempted)):
            self.preempted[i] = self.preempted[i] * 2

        self.preemption_trace = []
        for i in range(self.total_ticks_per_rank):
            if i * self.world_size + self.rank in self.preempted:
                self.preemption_trace.append(i)

    def check_preemption(self, iter_count):
        if self.preemption_trace is None:
            self.gen_trace()
        tick = (iter_count // self.iters_per_preemption) % self.total_ticks_per_rank
        if tick == 0 and iter_count % self.iters_per_preemption == 0:
            self.gen_trace()
        return tick in self.preemption_trace

    def plot(self):
        node_num = [0] * self.total_ticks_per_rank
        for i in range(self.world_size * self.total_ticks_per_rank):
            iter_num = i // self.world_size
            if i not in self.preempted:
                node_num[iter_num] += 1

        sec_per_tick = self.iters_per_preemption * self.sec_per_iter
        x, y = [0], [self.world_size]
        for minute, n in enumerate(node_num):
            if n != y[-1]:
                x.append(x[-1])
                y.append(n)
            x.append((minute + 1) * sec_per_tick)
            y.append(n)

        mean = np.mean(node_num)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(mean, color='red', linestyle='--')
        trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0, mean, "{:.1f}".format(mean), color="red", transform=trans, ha="right", va="center")
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)
        ax.set_yticks([0, 12, 24, 32])
        fig.savefig('log/trace/minutes.jpeg')

def mimic_preemption_trace_by_minutes(nnodes, prob):
    n_preemption_minutes = math.ceil(nnodes * 60 * prob)
    preempted = random.sample(list(range(nnodes * 60)), n_preemption_minutes)

    preempted_hist = defaultdict(list)
    node_num = [0] * 60
    for i in range(nnodes * 60):
        iter_num = i // nnodes
        if i not in preempted:
            node_num[iter_num] += 1
            preempted_hist[i % nnodes].append(1)
        else:
            preempted_hist[i % nnodes].append(0)

    for i in range(nnodes):
        print([len(list(g)) for i, g in groupby(preempted_hist[i]) if i == 0])

    x, y = [0], [nnodes]
    for minute, n in enumerate(node_num):
        if n != y[-1]:
            x.append(x[-1])
            y.append(n)
        x.append(minute + 1)
        y.append(n)

    mean = np.mean(node_num)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(mean, color='red', linestyle='--')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, mean, "{:.1f}".format(mean), color="red", transform=trans, ha="right", va="center")
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    ax.set_yticks([0, 12, 24, 32])
    fig.savefig('log/trace/minutes.jpeg')


if __name__ == '__main__':
    # mimic_preemption_trace_by_minutes(32, 0.5)

    # gen = GenPreemptionTrace(32, 0, prob=0.1)
    # gen.gen_trace()
    # gen.plot()
    pass
