import numpy as np
import pytest

import np_transformer


def test_single_head_attention():
    # Define input matrices
    X = np.array([[[1, 2], [3, 4]]])
    Wq = np.array([[[1, 0], [0, 1]]])
    Wk = np.array([[[1, 0], [0, 1]]])
    Wv = np.array([[[1, 0], [0, 1]]])
    Wo = np.array([[[1, 0], [0, 1]]])

    # Call the function
    output = np_transformer.single_head_attention(X, Wq, Wk, Wv, Wo)

    # Check the output shape
    assert output.shape == X.shape

    # Check the output values (this is a simple case where output should be equal to input)
    np.testing.assert_array_equal(output, X)
