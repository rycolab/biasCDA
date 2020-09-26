import torch as tr


def logsumexp(A):
    """
    Addition of log values
    :param A: torch.tensor
    :return: the logsumexp of all elements of A
    """
    offset = tr.max(A)
    A_exp = tr.exp(A - offset)
    return offset + tr.log(tr.sum(A_exp))


def logsumexp_col(A):
    """
    Addition of log values across columns of a 2D tensor
    :param A: 2D torch.tensor
    :return: the logsumexp of each column
    """
    res = tr.zeros(A.shape[1], dtype=A.dtype)
    for i in range(len(res)):
        res[i] = logsumexp(A[:, i])
    return res


def logsumexp_mat(A):
    """
    Addition of log values across columns of a 3D tensor
    :param A: 3D torch.tensor
    :return: the logsumexp of each column
    """
    res = tr.zeros(A.shape[1:], dtype=A.dtype)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = logsumexp(A[:, i, j])
    return res


def logmatmul(A, B):
    """
    Matrix multiplication in log space

    :param A: matrix of dimensions (m, r)
    :param B: matrix of dimensions (r, n)
    :return: matrix multiplication AB in log-space
    """
    max_A = tr.max(A)
    max_B = tr.max(B)
    AB = tr.matmul(tr.exp(A - max_A), tr.exp(B - max_B))
    AB = tr.log(AB)
    return AB + max_A + max_B


def maxmul(A, B, use_log):
    """
    Matrix (vector * matrix) multiplication for max-product

    :param A: vector of length n
    :param B:matrix of dimensions (n, m)
    :param use_log: True if operations should be done in log space, False otherwise
    :return: matrix multiplication AB for max-product
    """
    n = len(A)
    res = tr.zeros(n, dtype=tr.float64)
    arg_res = tr.zeros(n, dtype=tr.float64)
    for i in range(n):
        mul = A + B[:, i] if use_log else A * B[:, i]
        res[i] = tr.max(mul)
        arg_res[i] = tr.argmax(mul)
    return res, arg_res
