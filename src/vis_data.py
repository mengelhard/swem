import numpy as np
import matplotlib.pyplot as plt
import os


class TrainingSummary:

    def __init__(self, metrics):

        self.metrics = {x: [] for x in metrics}

    def add_point(self, metric, point):

        self.metrics[metric].append(point)

    def plot_metrics(self, ax, metrics):

        for metric in metrics:

            ax.plot(*np.array(self.metrics[metric]).T)

        ax.legend(metrics)
        ax.set_xlabel('Iteration')

        return ax

    def get_metric(self, metric):

        return np.array(self.metrics[metric])
