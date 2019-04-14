from CADJMatrix import CADJMatrix


class CONNMatrix:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        """

        """
        cadj_matrix = CADJMatrix.compute(self_organizing_map, input_map)
        conn_matrix = cadj_matrix + cadj_matrix.T
        return conn_matrix
