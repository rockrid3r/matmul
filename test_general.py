import matmul
import numpy as np
import pytest

def from_numpy(x: np.ndarray) -> matmul.Matrix:
    assert len(x.shape) == 2
    m = matmul.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            m.set(i, j, x[i, j])
    return m

def to_numpy(m: matmul.Matrix) -> np.ndarray:
    x = np.zeros((m.nrows, m.ncols))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = m.get(i, j)
    return x

def test_conversion():
    x = np.random.randn(5, 3)
    x_after = to_numpy(from_numpy(x))
    assert np.allclose(x, x_after)

def test_transpose_does_not_copy():
    x = from_numpy(np.random.randn(2, 2))
    x_t = x.transpose()
    x_t.set(0, 0, 1)
    x_t.set(0, 1, 2)
    x_t.set(1, 0, 3)
    x_t.set(1, 1, 4)
    expected = from_numpy(np.array([
        [1, 3],
        [2, 4],
    ]))
    assert np.allclose(to_numpy(expected), to_numpy(x))

def test_double_transpose():
    x = from_numpy(np.random.randn(2, 2))
    assert np.allclose(to_numpy(x.transpose().transpose()), to_numpy(x))


def test_matmul():
    x = from_numpy(np.random.randn(4, 5))
    y = from_numpy(np.random.randn(5, 13))
    assert np.allclose(
        to_numpy(x.matmul(y)),
        to_numpy(x).dot(to_numpy(y))
    )

def test_matmul_with_transpose():
    x = from_numpy(np.random.randn(5, 4))
    y = from_numpy(np.random.randn(5, 13))
    assert np.allclose(
        to_numpy(x.transpose().matmul(y)),
        to_numpy(x).T.dot(to_numpy(y))
    )

def test_full():
    expected = from_numpy(np.full((3, 5), 1337.144))
    actual = matmul.full((3, 5), 1337.144)
    assert np.allclose(
        to_numpy(expected),
        to_numpy(actual)
    )

def test_add():
    a = from_numpy(np.random.randn(3, 5))
    b = from_numpy(np.random.randn(3, 5))
    expected = from_numpy(to_numpy(a) + to_numpy(b))
    actual = a + b
    assert np.allclose(
        to_numpy(expected),
        to_numpy(actual)
    )

def test_add_wrong_type():
    with pytest.raises(ValueError):
        a = from_numpy(np.random.randn(3, 5))
        b = "asdf"
        a + b

def test_add_wrong_shape():
    with pytest.raises(ValueError):
        a = from_numpy(np.random.randn(3, 5))
        b = from_numpy(np.random.randn(4, 5))
        a + b

def test_negative():
    a = from_numpy(np.random.randn(3, 5))
    assert np.allclose(
        -to_numpy(a),
        to_numpy(-a)
    )

    assert np.allclose(
        to_numpy(a),
        to_numpy(-(-a))
    )

def test_simple_arithmetic_pipeline():
    _a = np.random.randn(3, 5)
    _b = np.random.randn(4, 5)
    _c = np.random.randn(4, 3)
    expected = (_a.dot(_b.T) + _c.T) / _c.T
    a, b, c = from_numpy(_a), from_numpy(_b), from_numpy(_c)
    actual = (a.matmul(b.transpose()) + c.transpose()) / c.transpose()
    assert np.allclose(expected, to_numpy(actual))

def test_fill():
    value = 9997
    filled_with_value = matmul.zeros((3, 4))
    filled_with_value.fill(value)
    assert np.allclose(
        np.ones((3, 4)) * value,
        to_numpy(filled_with_value)
    )
