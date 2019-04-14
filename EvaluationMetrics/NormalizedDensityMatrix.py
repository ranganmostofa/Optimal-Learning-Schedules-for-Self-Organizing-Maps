import numpy as np
from DensityMatrix import DensityMatrix


class NormalizedDensityMatrix:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        """

        """
        return \
            np.divide(
                DensityMatrix.compute(
                    self_organizing_map,
                    input_map
                ),
                len(input_map.values())
            )
