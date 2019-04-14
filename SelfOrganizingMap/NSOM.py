import random
import numpy as np
from EvaluationMetrics import \
    NeuronUtilization, \
    QuantizationError, \
    EmbeddingError, \
    TopographicError, \
    TopologicalProduct, \
    TopographicFunction, \
    CADJMatrix, \
    CONNMatrix, \
    WeightedCADJMatrix, \
    WeightedCONNMatrix


class NSOM:
    """

    """
    def __init__(self, lattice_dimension, prototype_dimension, prototype_matrix=None):
        """

        """
        self.lattice_dimension = tuple(lattice_dimension)
        self.prototype_dimension = prototype_dimension

        if type(prototype_matrix) is np.ndarray: self.prototype_matrix = prototype_matrix
        else: self.prototype_matrix = np.random.rand(*(list(lattice_dimension) + list([prototype_dimension])))

        self.__validate_prototype_matrix()

    def train(self, input_map, learning_rate_schedule, neighborhood_radius_schedule, num_iter):
        """

        """
        self.__validate_input_data(input_map)

        for iteration_index in range(num_iter):
            input_signal = np.array(random.choice(list(input_map.values())))

            current_learning_rate = learning_rate_schedule.compute_learning_parameter_value(iteration_index)
            current_neighborhood_radius = neighborhood_radius_schedule.compute_learning_parameter_value(iteration_index)

            difference_matrix = np.subtract(input_signal, self.get_prototype_matrix())
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)

            winner_index = np.unravel_index(np.argmin(euclidean_distance_matrix), euclidean_distance_matrix.shape)

            index_grid = np.swapaxes(np.transpose(np.indices(euclidean_distance_matrix.shape), axes=[0, 2, 1]), 0, 2)
            manhattan_distance_matrix = np.sum(np.absolute(np.subtract(index_grid, np.array(winner_index))), axis=2)
            neighborhood_mask = (manhattan_distance_matrix <= current_neighborhood_radius).astype(int)
            neighborhood_matrix = \
                np.multiply(
                    current_learning_rate,
                    np.multiply(
                        neighborhood_mask,
                        np.exp(
                            -np.power(
                                np.divide(
                                    manhattan_distance_matrix,
                                    (2 * current_neighborhood_radius)
                                ), 2
                            )
                        )
                    )
                )

            self.prototype_matrix = \
                np.add(
                    self.get_prototype_matrix(),
                    np.multiply(
                        np.repeat(
                            neighborhood_matrix[:, :, np.newaxis],
                            self.get_prototype_dimension(),
                            axis=2
                        ),
                        difference_matrix
                    )
                )

            loser_index = np.unravel_index(np.argmax(euclidean_distance_matrix), euclidean_distance_matrix.shape)

            d1 = np.linalg.norm(np.subtract(self.get_prototype_matrix()[winner_index], input_signal))
            d2 = np.linalg.norm(np.subtract(self.get_prototype_matrix()[winner_index], self.get_prototype_matrix()[loser_index]))
            d3 = np.linalg.norm(np.subtract(self.get_prototype_matrix()[loser_index], input_signal))
            cosine = (d1 ** 2 + d3 ** 2 - d2 ** 2) / (2 * d1 * d3)
            denominator = d3 / (d1 * abs(cosine))

            index_grid = np.swapaxes(np.transpose(np.indices(euclidean_distance_matrix.shape), axes=[0, 2, 1]), 0, 2)
            manhattan_distance_matrix = np.sum(np.absolute(np.subtract(index_grid, np.array(loser_index))), axis=2)
            neighborhood_mask = (manhattan_distance_matrix <= current_neighborhood_radius).astype(int)
            neighborhood_matrix = \
                np.multiply(
                    current_learning_rate / denominator,
                    np.multiply(
                        neighborhood_mask,
                        np.exp(
                            -np.power(
                                np.divide(
                                    manhattan_distance_matrix,
                                    (2 * current_neighborhood_radius)
                                ), 2
                            )
                        )
                    )
                )

            self.prototype_matrix = \
                np.subtract(
                    self.get_prototype_matrix(),
                    np.multiply(
                        np.repeat(
                            neighborhood_matrix[:, :, np.newaxis],
                            self.get_prototype_dimension(),
                            axis=2
                        ),
                        difference_matrix
                    )
                )

    def get_lattice_dimension(self):
        """

        """
        return self.lattice_dimension

    def get_prototype_dimension(self):
        """

        """
        return self.prototype_dimension

    def get_prototype_matrix(self):
        """

        """
        return self.prototype_matrix

    def set_lattice_dimension(self, lattice_dimension):
        """

        """
        return \
            NSOM(
                lattice_dimension=lattice_dimension,
                prototype_dimension=self.get_prototype_dimension(),
                prototype_matrix=self.get_prototype_matrix()
            )

    def set_prototype_dimension(self, prototype_dimension):
        """

        """
        return \
            NSOM(
                lattice_dimension=self.get_lattice_dimension(),
                prototype_dimension=prototype_dimension,
                prototype_matrix=self.get_prototype_matrix()
            )

    def set_prototype_matrix(self, prototype_matrix):
        """

        """
        return \
            NSOM(
                lattice_dimension=self.get_lattice_dimension(),
                prototype_dimension=self.get_prototype_dimension(),
                prototype_matrix=prototype_matrix
            )

    def __validate_prototype_matrix(self):
        """

        """
        actual_dimension = tuple(self.get_prototype_matrix().shape)
        expected_dimension = tuple(list(self.get_lattice_dimension()) + list([self.get_prototype_dimension()]))

        if not expected_dimension.__eq__(actual_dimension):
            raise \
                Exception(
                    "Mismatch between prototype matrix and lattice dimensions! Terminating..."
                )

    def __validate_input_data(self, data):
        """

        """
        for input_signal in data.values():
            if not len(input_signal).__eq__(self.get_prototype_dimension()):
                raise \
                    Exception(
                        "Dimensional mismatch between prototype matrix and at least one input signal! Terminating..."
                    )


from time import time
from FileIO import FileIO
from LearningSchedule import LearningSchedule
from SOMViz.SOMVisualizer import SOMVisualizer


prototype_dimension = 2

x_dim = 8
y_dim = 8
lattice_dimension = tuple((x_dim, y_dim))

som = NSOM(lattice_dimension, prototype_dimension)

# data = \
#     {
#         0: np.array([random.random() for _ in range(prototype_dimension)]),
#         1: np.array([random.random() for _ in range(prototype_dimension)]),
#         2: np.array([random.random() for _ in range(prototype_dimension)]),
#         3: np.array([random.random() for _ in range(prototype_dimension)]),
#         4: np.array([random.random() for _ in range(prototype_dimension)]),
#         5: np.array([random.random() for _ in range(prototype_dimension)]),
#         6: np.array([random.random() for _ in range(prototype_dimension)]),
#         7: np.array([random.random() for _ in range(prototype_dimension)]),
#         8: np.array([random.random() for _ in range(prototype_dimension)])
#     }

data = FileIO.read_lrn("./../Data/WingNut.lrn")

class_map = FileIO.read_cls("./../Data/WingNut.cls")

num_iter = int(4e3)

alpha_sched = LearningSchedule(dict({200: 0.5, 600: 0.2, 1200: 0.1, 2000: 0.05}))

sigma_sched = LearningSchedule(dict({200: 6, 600: 4, 1200: 2, 2000: 1}))

scale = 1.0 / 10.0
data_ndarray = np.array(list(data.values()))
data_mean = np.mean(data_ndarray, axis=0)
data_range = np.ptp(data_ndarray, axis=0)
prototype_range = scale * data_range
initial_prototype_matrix = np.array([np.array([np.array([prototype_range[index] * np.random.random() + (data_mean[index] - prototype_range[index] / 2) for index in range(len(prototype_range))]) for __ in range(y_dim)]) for _ in range(x_dim)])

t0 = time()

som.train(data, alpha_sched, sigma_sched, num_iter)

t1 = time()

# print(som.get_prototype_matrix(), "\n")
#
# print(np.around(som.get_prototype_matrix()), "\n")

print("Time taken to train SOM:", str(t1 - t0), "seconds\n")

print("Neuron Utilization:", NeuronUtilization.compute(som, data), "\n")

print("Quantization Error:", QuantizationError.compute(som, data), "\n")

# print("Embedding Error:", EmbeddingError.compute(som, data, 0.05), "\n")

print("Topographic Error:", TopographicError.compute(som, data), "\n")

# t2 = time()
#
# print("Topological Product:", TopologicalProduct.compute(som), "\n")
#
# t3 = time()
#
# print("Time taken to compute TP:", t3 - t2, "seconds\n")

# print("Topographic Function:", TopographicFunction.compute(som), "\n")
#
# print("CADJ Matrix:\n", CADJMatrix.compute(som, data), "\n")
#
# print("CONN Matrix:\n", CONNMatrix.compute(som, data), "\n")
#
# print("Weighted CADJ Matrix:\n", WeightedCADJMatrix.compute(som, data), "\n")
#
# print("Weighted CONN Matrix:\n", WeightedCONNMatrix.compute(som, data), "\n")

# input("Press Enter to continue...")

if prototype_dimension == 2:
    SOMVisualizer.plot_som_input_space(som, data, class_map)

    SOMVisualizer.plot_density_matrix(som, data)

    SOMVisualizer.plot_unified_distance_matrix(som)
