import numpy as np


class WeightedCADJMatrix:
    """

    """

    @staticmethod
    def compute(self_organizing_map, input_map):
        prototype_matrix = self_organizing_map.get_prototype_matrix()
        N = int(np.prod(np.array(prototype_matrix.shape[:len(prototype_matrix.shape) - 1])))

        wcadj_matrix = np.zeros(tuple((N, N)))

        for input_signal in input_map.values():
            difference_matrix = np.subtract(input_signal, prototype_matrix)
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)
            first_bmu_index = np.argmin(euclidean_distance_matrix)

            euclidean_distance_matrix[np.unravel_index(first_bmu_index, euclidean_distance_matrix.shape)] = np.inf
            second_bmu_index = np.argmin(euclidean_distance_matrix)

            first_bmu = prototype_matrix[np.unravel_index(first_bmu_index, euclidean_distance_matrix.shape)]
            second_bmu = prototype_matrix[np.unravel_index(second_bmu_index, euclidean_distance_matrix.shape)]

            d1 = np.linalg.norm(np.subtract(first_bmu, input_signal))
            d2 = np.linalg.norm(np.subtract(second_bmu, input_signal))
            d3 = np.linalg.norm(np.subtract(first_bmu, second_bmu))

            cosine_theta1 = abs((d1 ** 2 + d3 ** 2 - d2 ** 2) / (2 * d1 * d3))
            cosine_theta2 = abs((d2 ** 2 + d3 ** 2 - d1 ** 2) / (2 * d2 * d3))

            wcadj_matrix[tuple((first_bmu_index, second_bmu_index))] += \
                d1 * cosine_theta1 / (d1 * cosine_theta1 + d2 * cosine_theta2)

        return wcadj_matrix
