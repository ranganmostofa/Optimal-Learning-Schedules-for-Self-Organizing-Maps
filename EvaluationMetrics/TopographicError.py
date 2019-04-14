import numpy as np


class TopographicError:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        topographic_error = 0
        prototype_matrix = self_organizing_map.get_prototype_matrix()

        for input_signal in input_map.values():
            difference_matrix = np.subtract(input_signal, prototype_matrix)
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)
            first_bmu_index = np.unravel_index(np.argmin(euclidean_distance_matrix), euclidean_distance_matrix.shape)

            euclidean_distance_matrix[first_bmu_index] = np.inf
            second_bmu_index = np.unravel_index(np.argmin(euclidean_distance_matrix), euclidean_distance_matrix.shape)

            topographic_error += \
                int(
                    np.sum(
                        np.absolute(
                            np.subtract(
                                np.array(first_bmu_index),
                                np.array(second_bmu_index)
                            )
                        )
                    ) > 1
                )

        return topographic_error / len(input_map.values())
