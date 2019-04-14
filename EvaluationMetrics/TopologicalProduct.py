import numpy as np


class TopologicalProduct:
    """

    """
    @staticmethod
    def compute(self_organizing_map):
        """

        """
        denormalized_products = list()
        prototype_matrix = self_organizing_map.get_prototype_matrix()
        index_grid_shape = tuple(list(prototype_matrix.shape)[:len(prototype_matrix.shape) - 1])
        N = int(np.prod(index_grid_shape))
        for j in range(N):
            neuron_index = np.unravel_index(j, index_grid_shape)
            neuron_neighborhood_map = \
                TopologicalProduct.__construct_neuron_neighborhood_map(
                    self_organizing_map,
                    neuron_index
                )
            prototype_neighborhood_map = \
                TopologicalProduct.__construct_prototype_neighborhood_map(
                    self_organizing_map,
                    neuron_index
                )
            for k in range(N - 1):
                denormalized_products.append(
                    TopologicalProduct.__compute_denormalized_product(
                        self_organizing_map,
                        neuron_index,
                        k + 1,
                        neuron_neighborhood_map,
                        prototype_neighborhood_map
                    )
                )
        return 1.0 / (N * (N - 1)) * np.sum(np.log(np.array(denormalized_products)))

    @staticmethod
    def __compute_denormalized_product(self_organizing_map, neuron_index, k,
                                       neuron_neighborhood_map, prototype_neighborhood_map):
        """

        """
        return \
            np.power(
                np.prod(
                    np.array(
                        [
                            TopologicalProduct.__compute_q1_ratio(
                                self_organizing_map,
                                neuron_index,
                                l + 1,
                                neuron_neighborhood_map,
                                prototype_neighborhood_map
                            ) *
                            TopologicalProduct.__compute_q2_ratio(
                                neuron_index,
                                l + 1,
                                neuron_neighborhood_map,
                                prototype_neighborhood_map
                            ) for l in range(k)
                        ]
                    )
                ),
                1.0 / (2.0 * k)
            )

    @staticmethod
    def __compute_q1_ratio(self_organizing_map, neuron_index, k, neuron_neighborhood_map, prototype_neighborhood_map):
        """

        """
        prototype_matrix = self_organizing_map.get_prototype_matrix()

        target_prototype = prototype_matrix[neuron_index]
        input_space_neighboring_prototype = prototype_matrix[prototype_neighborhood_map[k]]
        output_space_neighboring_prototype = prototype_matrix[neuron_neighborhood_map[k]]

        numerator = np.linalg.norm(np.subtract(target_prototype, output_space_neighboring_prototype))
        denominator = np.linalg.norm(np.subtract(target_prototype, input_space_neighboring_prototype))

        return numerator / denominator

    @staticmethod
    def __compute_q2_ratio(neuron_index, k, neuron_neighborhood_map, prototype_neighborhood_map):
        """

        """
        input_space_neighboring_neuron = prototype_neighborhood_map[k]
        output_space_neighboring_neuron = neuron_neighborhood_map[k]

        numerator = np.sum(np.absolute(np.subtract(output_space_neighboring_neuron, np.array(neuron_index))))
        denominator = np.sum(np.absolute(np.subtract(input_space_neighboring_neuron, np.array(neuron_index))))

        return numerator / denominator

    @staticmethod
    def __construct_neuron_neighborhood_map(self_organizing_map, neuron_index):
        """

        """
        prototype_matrix = self_organizing_map.get_prototype_matrix()

        index_grid_shape = tuple(list(prototype_matrix.shape)[:len(prototype_matrix.shape) - 1])
        index_grid = np.swapaxes(np.transpose(np.indices(index_grid_shape), axes=[0, 2, 1]), 0, 2)

        manhattan_distance_matrix = np.sum(np.absolute(np.subtract(index_grid, np.array(neuron_index))), axis=2)

        sorted_indices = \
            list(
                zip(
                    *np.unravel_index(
                        np.argsort(
                            manhattan_distance_matrix,
                            axis=None
                        ),
                        manhattan_distance_matrix.shape
                    )
                )
            )[1:]

        return dict({k_val + 1: sorted_indices[k_val] for k_val in range(len(sorted_indices))})

    @staticmethod
    def __construct_prototype_neighborhood_map(self_organizing_map, neuron_index):
        """

        """
        prototype_matrix = self_organizing_map.get_prototype_matrix()
        target_prototype = prototype_matrix[neuron_index]

        difference_matrix = np.subtract(target_prototype, prototype_matrix)
        euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)

        sorted_indices = \
            list(
                zip(
                    *np.unravel_index(
                        np.argsort(
                            euclidean_distance_matrix,
                            axis=None
                        ),
                        euclidean_distance_matrix.shape
                    )
                )
            )[1:]

        return dict({k_val + 1: sorted_indices[k_val] for k_val in range(len(sorted_indices))})
