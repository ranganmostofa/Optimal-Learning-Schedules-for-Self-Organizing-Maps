import numpy as np


class DensityMatrix:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        """

        """
        prototype_matrix = self_organizing_map.get_prototype_matrix()
        density_matrix = np.zeros(list(prototype_matrix.shape)[:len(prototype_matrix.shape) - 1])

        for input_signal in input_map.values():
            difference_matrix = np.subtract(input_signal, prototype_matrix)
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)
            winner_index = np.unravel_index(np.argmin(euclidean_distance_matrix), euclidean_distance_matrix.shape)
            density_matrix[winner_index] += 1

        return density_matrix
