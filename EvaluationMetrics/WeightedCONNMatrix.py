from WeightedCADJMatrix import WeightedCADJMatrix


class WeightedCONNMatrix:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map):
        wcadj_matrix = WeightedCADJMatrix.compute(self_organizing_map, input_map)
        wconn_matrix = wcadj_matrix + wcadj_matrix.T
        return wconn_matrix
