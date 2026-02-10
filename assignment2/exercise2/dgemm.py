import numpy as np
import pytest


def dgemm(array_a, array_b, array_c):
    for i in range(len(array_a)):
        for j in range(len(array_b)):
            for k in range(len(array_c)):
                array_c[i,j] += array_a[i,k] * array_b[k,j]
    return array_c


@pytest.fixture
def generate_set_data():
    # a traditional unit test
    a = np.array([[3, 1, 3, 2], [2, 6, 1, 7], [1, 2, 1, 5], [4, 8, 2, 1]])
    b = np.array([[2, 0, 9, 1], [9, 4, 8, 3], [8, 6, 5, 0], [9, 7, 5, 2]])
    c = np.array([[1, 3, 5, 4], [3, 2, 8, 1], [4, 7, 5, 6], [8, 5, 2, 0]])
    return a, b, c


@pytest.fixture
def generate_random_data():
    # generate random matrix, maybe we catch unwanted error
    rng = np.random.default_rng()
    SIZE = 4
    a = rng.random((SIZE, SIZE))
    b = rng.random((SIZE, SIZE))
    c = rng.random((SIZE, SIZE))
    return a, b, c


def test_set_dgemm(generate_set_data):
    a, b, c = generate_set_data
    expected = np.matmul(a, b) + c
    actual = dgemm(a, b, c)
    assert np.allclose(actual, expected)


def test_random_dgemm(generate_random_data):
    a, b, c = generate_random_data
    expected = np.matmul(a, b) + c
    actual = dgemm(a, b, c)
    assert np.allclose(actual, expected)