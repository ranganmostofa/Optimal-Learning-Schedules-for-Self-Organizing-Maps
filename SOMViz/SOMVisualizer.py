import numpy as np
import matplotlib.pyplot as plt
from SOMVizGlobals import SOMVizGlobals


class SOMVisualizer:
    """

    """
    @staticmethod
    def plot_som_input_space(som, input_map, class_map, display=True):
        """

        """
        SOMVisualizer.plot_input_signals(input_map, class_map, display=False)

        SOMVisualizer.plot_som_grid(som, display=False)

        if display: plt.show()

    @staticmethod
    def plot_input_signals(input_map, class_map, display=True):
        """

        """
        input_x, input_y = list(zip(*input_map.values()))
        # class_labels = SOMVisualizer.__compute_class_labels(input_map, input_x, input_y, class_map)
        # plt.scatter(input_x, input_y, marker=SOMVizGlobals.DATA_MARKER, c=class_labels)
        plt.scatter(input_x, input_y, marker=SOMVizGlobals.DATA_MARKER)
        if display: plt.show()

    @staticmethod
    def __compute_class_labels(input_map, input_x, input_y, class_map):
        """

        """
        class_labels = list()
        for point_index in range(len(input_x)):
            x, y = input_x[point_index], input_y[point_index]
            for key, value in input_map.items():
                if list(value) == list([x, y]):
                    class_labels.append(class_map[key])
                    break
        return class_labels

    @staticmethod
    def plot_som_grid(som, display=True):
        """

        """
        som_x, som_y = list(), list()
        prototype_matrix = som.get_prototype_matrix()
        for row in prototype_matrix:
            x, y = list(zip(*list(row)))
            som_x += list(x)
            som_y += list(y)
        plt.plot(som_x, som_y, SOMVizGlobals.PROTOTYPE_MARKER + SOMVizGlobals.PROTOTYPE_COLOR)

        for row_index in range(len(prototype_matrix)):
            for column_index in range(len(prototype_matrix[row_index])):
                x, y = prototype_matrix[row_index][column_index]

                if row_index:
                    upper_x, upper_y = prototype_matrix[row_index - 1][column_index]
                    plt.plot(list([x, upper_x]), list([y, upper_y]), SOMVizGlobals.PROTOTYPE_COLOR)

                if column_index:
                    left_x, left_y = prototype_matrix[row_index][column_index - 1]
                    plt.plot(list([x, left_x]), list([y, left_y]), SOMVizGlobals.PROTOTYPE_COLOR)

                if row_index is not len(prototype_matrix) - 1:
                    lower_x, lower_y = prototype_matrix[row_index + 1][column_index]
                    plt.plot(list([x, lower_x]), list([y, lower_y]), SOMVizGlobals.PROTOTYPE_COLOR)

                if column_index is not len(prototype_matrix[row_index]) - 1:
                    right_x, right_y = prototype_matrix[row_index][column_index + 1]
                    plt.plot(list([x, right_x]), list([y, right_y]), SOMVizGlobals.PROTOTYPE_COLOR)

        if display: plt.show()

    @staticmethod
    def plot_density_matrix(som, input_map):
        """

        """
        density_matrix = SOMVisualizer.construct_density_matrix(som, input_map)
        plt.imshow(density_matrix, cmap='hot', interpolation='nearest')
        plt.show()

    @staticmethod
    def construct_density_matrix(som, input_map):
        """

        """
        prototype_matrix = som.get_prototype_matrix()
        density_matrix = np.zeros(list(prototype_matrix.shape)[:len(prototype_matrix.shape) - 1])

        for input_signal in input_map.values():
            difference_matrix = np.subtract(input_signal, prototype_matrix)
            euclidean_distance_matrix = np.linalg.norm(difference_matrix, axis=2)
            winner_index = np.unravel_index(np.argmin(euclidean_distance_matrix), euclidean_distance_matrix.shape)
            density_matrix[winner_index] += 1
        density_matrix = np.divide(density_matrix, len(input_map.values()))

        return density_matrix

    @staticmethod
    def construct_lattice_histogram(som):
        pass

    @staticmethod
    def plot_unified_distance_matrix(som):
        """

        """
        unified_distance_matrix = SOMVisualizer.construct_unified_distance_matrix(som)
        plt.imshow(unified_distance_matrix, cmap='gray')
        plt.show()

    @staticmethod
    def construct_unified_distance_matrix(som):
        """

        """
        prototype_matrix = som.get_prototype_matrix()
        unified_distance_matrix = np.zeros(list(prototype_matrix.shape)[:len(prototype_matrix.shape) - 1])

        for row_index in range(len(prototype_matrix)):
            for column_index in range(len(prototype_matrix[row_index])):
                prototype_vector = prototype_matrix[row_index][column_index]

                if row_index:
                    upper_vector = prototype_matrix[row_index - 1][column_index]
                    unified_distance_matrix[row_index][column_index] += \
                        np.linalg.norm(np.subtract(prototype_vector, upper_vector))

                if column_index:
                    left_vector = prototype_matrix[row_index][column_index - 1]
                    unified_distance_matrix[row_index][column_index] += \
                        np.linalg.norm(np.subtract(prototype_vector, left_vector))

                if row_index is not len(prototype_matrix) - 1:
                    lower_vector = prototype_matrix[row_index + 1][column_index]
                    unified_distance_matrix[row_index][column_index] += \
                        np.linalg.norm(np.subtract(prototype_vector, lower_vector))

                if column_index is not len(prototype_matrix[row_index]) - 1:
                    right_vector = prototype_matrix[row_index][column_index + 1]
                    unified_distance_matrix[row_index][column_index] += \
                        np.linalg.norm(np.subtract(prototype_vector, right_vector))

        return unified_distance_matrix

    @staticmethod
    def construct_mod_ultsch_matrix():
        pass
