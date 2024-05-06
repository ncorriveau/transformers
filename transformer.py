import numpy as np
from scipy.special import softmax


def single_head_attention(X: np.ndarray, 
                          Wq: np.ndarray, 
                          Wk: np.ndarray, 
                          Wv: np.ndarray, 
                          Wo: np.ndarray) -> np.ndarray:
    """
    This computes a single attention head with matrix values. 
        D = Model dimension (e.g. the output dim of the embedding layer)
        N = Number of elements in the input sequence
        B = Batch size
        Dk = Key dimension
        Dv = Value dimension
    
    Parameters 
    ----------
    X: input matrix of shape (B, N, D)
    Wq: query matrix of shape (B, D, Dk)
    Wk: key matrix of shape (B, D, Dk)
    Wv: value matrix of shape (B, D, Dv)
    Wo: output matrix of shape (B, Dv, D) e.g. projects back into model dimension 

    Returns
    -------
    output: output matrix of shape (B, N, D)
    """
    Q = np.matmul(X, Wq)
    K = np.matmul(X, Wk)
    V = np.matmul(X, Wv)
    print(f" Q shape = {Q.shape}")
    print(f" K.T shape = {K.transpose(0, 2, 1).shape}")

    A = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(Q.shape[-1])
    print(f"Attention matrix shape: {A.shape}")
    A = softmax(A, axis=-1)
    print(f"Attention w softmax shape: {A.shape}")

    output = np.matmul(A, V)
    print(f"Output matrix shape: {output.shape}")

    final = np.matmul(output, Wo)
    print(f"Final matrix shape: {final.shape}")
    return final

    # return (softmax(np.dot(Q, K.T) / np.sqrt(Q.shape[-1])) @ V) @ Wo
    
    

if __name__ == "__main__":
    # Define input matrices
    X = np.array([[[1, 2], [3, 4]]])
    Wq = np.array([[[1, 0], [0, 1]]])
    Wk = np.array([[[1, 0], [0, 1]]])
    Wv = np.array([[[1, 0], [0, 1]]])
    Wo = np.array([[[1, 0], [0, 1]]])
    print(X.shape)

    # Call the function
    output = single_head_attention(X, Wq, Wk, Wv, Wo)
    print(output.shape)

    # Check the output shape
    assert (output.shape == X.shape)

    # Check the output values (this is a simple case where output should be equal to input)
    # np.testing.assert_array_equal(output, X)
