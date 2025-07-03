"""Visualizations.distance module"""

# pylint: disable = import-error, too-few-public-methods, line-too-long, missing-final-newline
import numpy as np
import matplotlib.pyplot as plt


class DistanceViz:
    """Visualizes Levenshtein distance"""

    def plot(self, explanations):
        """Plot the Levenshtein distance matrix"""
        cmap = plt.cm.viridis  # pylint: disable=no-member

        _, axes = plt.subplots()
        cax = axes.imshow(explanations.matrix, cmap=cmap, interpolation="nearest")

        plt.colorbar(cax)

        axes.set_xticks(np.arange(len(explanations.reference)))
        axes.set_yticks(np.arange(len(explanations.hypothesis)))
        axes.set_xticklabels(explanations.reference)
        axes.set_yticklabels(explanations.hypothesis)

        plt.setp(
            axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        nrows, ncols = explanations.matrix.shape
        for i in range(nrows):
            for j in range(ncols):
                color = (
                    "white"
                    if explanations.matrix[i, j] < explanations.matrix.max() / 2
                    else "black"
                )
                axes.text(
                    j,
                    i,
                    int(explanations.matrix[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                )

        plt.show()
