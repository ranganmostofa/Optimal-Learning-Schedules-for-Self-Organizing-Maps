import numpy as np


class QuantizationError:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        """

        """
        quantization_error = 0
        prototype_matrix = self_organizing_map.get_prototype_matrix()

        for input_signal in input_map.values():
            difference_matrix = np.subtract(input_signal, prototype_matrix)
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)
            winner_index = np.unravel_index(np.argmin(euclidean_distance_matrix), euclidean_distance_matrix.shape)
            quantization_error += np.linalg.norm(np.subtract(prototype_matrix[winner_index], input_signal))
        quantization_error /= len(input_map.values())

        return quantization_error
