# Matmul
Yet another linear algebra library.
# What is this?
CPython-extension implementing basic matrix operations.
## Why?
For fun.

# Installation
```bash
git clone https://github.com/rockrid3r/matmul
cd matmul
make && make install
```

## Usage
```python
import matmul
a = matmul.rand((3, 5))
b = matmul.rand((3, 5))
c = matmul.rand((5, 5))
d = a.transpose().matmul(b) + c
```
