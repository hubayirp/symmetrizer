import numpy as np


class Group:
    """
    Abstract group class
    """
    def __init__(self):
        """
        Set group parameters
        """
        raise NotImplementedError

    def _input_transformation(self, weights, transformation):
        """
        Specify input transformation
        """
        raise NotImplementedError

    def _output_transformation(self, weights, transformation):
        """
        Specify output transformation
        """
        raise NotImplementedError


class MatrixRepresentation(Group):
    """
    Representing group elements as matrices
    """
    def __init__(self, input_matrices, output_matrices):
        """
        """
        self.repr_size_in = input_matrices[0].shape[1]
        self.repr_size_out = output_matrices[0].shape[1]
        self._input_matrices = input_matrices
        self._output_matrices = output_matrices

        self.parameters = range(len(input_matrices))

    def _input_transformation(self, weights, params):
        """
        Input transformation comes from the input group
        W F_g z
        """
        weights = np.matmul(weights, self._input_matrices[params])
        return weights

    def _output_transformation(self, weights, params):
        """
        Output transformation from the output group
        P_g W z
        """
        weights = np.matmul(self._output_matrices[params], weights)
        return weights
