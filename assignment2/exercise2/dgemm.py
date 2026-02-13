import array as arr
import numpy as np
import pytest


### 2.1
# dgemm with list
def dgemm_list(a, b, c):
    """DGEMM implementation using list data structure"""
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                c[i][j] += a[i][k] * b[k][j]
    return c


# helper function returning 2D matrix indexing [i, j] for 1D array
def array_index(a, b, n):
    return a * n + b

# dgemm with array
def dgemm_array(a, b, c, n):
    """DGEMM implementation using array data structure"""
    for i in range(n):
        i_n = i*n
        for j in range(n):
            c_index = i_n + j
            for k in range(n):
                c[c_index] += a[i_n + k] * b[k * n + j]
    return c


# dgemm with numpy(?) unsure if this is intended data structure
def dgemm_numpy(a, b, c):
    """DGEMM implementation using np.array data structure"""
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                c[i,j] += a[i,k] * b[k,j]
    return c


### tests for 2.2
@pytest.fixture
def generate_set_data_list():
    # very elementary unit test
    a = [[3, 1, 3, 2], [2, 6, 1, 7], [1, 2, 1, 5], [4, 8, 2, 1]]
    b = [[2, 0, 9, 1], [9, 4, 8, 3], [8, 6, 5, 0], [9, 7, 5, 2]]
    c = [[1, 3, 5, 4], [3, 2, 8, 1], [4, 7, 5, 6], [8, 5, 2, 0]]
    expected = [[58, 39, 65, 14], [132, 81, 114, 35], [77, 56, 60, 23], [113, 56, 117, 30]]
    return a, b, c, expected


@pytest.fixture
def generate_set_data_array():
    # very elementary unit test
    a = arr.array('i', [3, 1, 3, 2, 2, 6, 1, 7, 1, 2, 1, 5, 4, 8, 2, 1])
    b = arr.array('i', [2, 0, 9, 1, 9, 4, 8, 3, 8, 6, 5, 0, 9, 7, 5, 2])
    c = arr.array('i', [1, 3, 5, 4, 3, 2, 8, 1, 4, 7, 5, 6, 8, 5, 2, 0])
    expected = arr.array('i', [58, 39, 65, 14, 132, 81, 114, 35, 77, 56, 60, 23, 113, 56, 117, 30])
    return a, b, c, expected


@pytest.fixture
def generate_set_data_numpy():
    # very elementary unit test
    a = np.array([[3, 1, 3, 2], [2, 6, 1, 7], [1, 2, 1, 5], [4, 8, 2, 1]])
    b = np.array([[2, 0, 9, 1], [9, 4, 8, 3], [8, 6, 5, 0], [9, 7, 5, 2]])
    c = np.array([[1, 3, 5, 4], [3, 2, 8, 1], [4, 7, 5, 6], [8, 5, 2, 0]])
    expected = np.array([[58, 39, 65, 14], [132, 81, 114, 35], [77, 56, 60, 23], [113, 56, 117, 30]])
    return a, b, c, expected


def test_set_dgemm_list(generate_set_data_list):
    a, b, c, expected = generate_set_data_list
    actual = dgemm_list(a, b, c)
    assert np.allclose(actual, expected)


def test_set_dgemm_array(generate_set_data_array):
    a, b, c, expected = generate_set_data_array
    actual = dgemm_array(a, b, c, 4)
    assert np.allclose(actual, expected)


def test_set_dgemm_numpy(generate_set_data_numpy):
    a, b, c, expected = generate_set_data_numpy
    actual = dgemm_numpy(a, b, c)
    assert np.allclose(actual, expected)
