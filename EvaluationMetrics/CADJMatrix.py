import numpy as np


class CADJMatrix:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        """

        """
        prototype_matrix = self_organizing_map.get_prototype_matrix()
        N = int(np.prod(self_organizing_map.get_lattice_dimension()))  # TODO check this

        cadj_matrix = np.zeros(tuple((N, N)))

        for input_signal in input_map.values():
            difference_matrix = np.subtract(input_signal, prototype_matrix)
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)
            first_bmu_index = np.argmin(euclidean_distance_matrix)

            euclidean_distance_matrix[np.unravel_index(first_bmu_index, euclidean_distance_matrix.shape)] = np.inf
            second_bmu_index = np.argmin(euclidean_distance_matrix)

            cadj_matrix[tuple((first_bmu_index, second_bmu_index))] += 1

        return cadj_matrix
