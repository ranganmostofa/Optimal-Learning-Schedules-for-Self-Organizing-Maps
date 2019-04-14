import numpy as np
from DensityMatrix import DensityMatrix


class NeuronUtilization:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        """

        """
        density_matrix = DensityMatrix.compute(self_organizing_map, input_map)

        return \
            len(
                list(
                    np.where(
                        density_matrix > 0
                    )
                ).pop()
            ) / density_matrix.size
