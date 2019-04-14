import numpy as np
from scipy.stats import f


class EmbeddingError:
    """

    """
    @staticmethod
    def compute(self_organizing_map, input_map, alpha_val):
        """

        """
        n1 = len(input_map.values())
        input_vectors = np.array(list(input_map.values()))

        prototype_matrix = self_organizing_map.get_prototype_matrix()
        n2 = int(np.prod(prototype_matrix.shape[:len(prototype_matrix.shape) - 1]))
        prototype_vectors = \
            prototype_matrix.reshape(
                tuple((n2, )) +
                prototype_matrix.shape[-1:]
            )

        embedding_error = 0.0
        N = prototype_matrix.shape[-1]
        for feature_index in range(N):
            input_variance = np.var(input_vectors[:, feature_index])
            prototype_variance = np.var(prototype_vectors[:, feature_index])

            variance_p_val = f.cdf(input_variance / prototype_variance, n1 - 1, n2 - 1)

            z_score = 2.33
            input_mean = np.mean(input_vectors[:, feature_index])
            prototype_mean = np.mean(prototype_vectors[:, feature_index])

            mean_lower_bound = (input_mean - prototype_mean) - z_score * np.sqrt(input_variance / n1 + prototype_variance / n2)
            mean_upper_bound = (input_mean - prototype_mean) + z_score * np.sqrt(input_variance / n1 + prototype_variance / n2)

            print(variance_p_val < alpha_val)

            embedding_error += float(variance_p_val < alpha_val) / N

        return embedding_error
