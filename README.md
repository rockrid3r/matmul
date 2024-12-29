# Matmul
Yet another linear algebra library.
# What is this?
CPython-extension implementing basic matrix operations.
## Why?
For fun.
## Basic example
```python
import matmul
a = matmul.rand(3, 5)
b = matmul.rand(3, 5)
c = matmul.rand(5, 5)
d = a.transpose().matmul(b) + c
```
