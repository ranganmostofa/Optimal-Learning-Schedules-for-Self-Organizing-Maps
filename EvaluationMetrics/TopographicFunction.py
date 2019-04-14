import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


class TopographicFunction:
    """

    """
    @staticmethod
    def compute(self_organizing_map):
        prototype_matrix = self_organizing_map.get_prototype_matrix()
        prototype_vectors = \
            prototype_matrix.reshape(
                (np.prod(
                    prototype_matrix.shape[:len(prototype_matrix.shape) - 1]
                ), ) +
                prototype_matrix.shape[-1:]
            )

        voronoi = Voronoi(prototype_vectors)

        # voronoi_plot_2d(voronoi)
        # plt.show()
