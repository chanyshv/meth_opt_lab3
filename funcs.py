import numpy as np


def f_bukin(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)


def f_matias(x):
    return 0.28 * (x[0] ** 2 + x[1] ** 2) - 0.46 * x[0] * x[1]


def f_rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def f_rosenbrock_chunk():
    return [lambda x: 1 - x[0], lambda x: 10 * (x[1] - x[0] * x[0])]


def squared(funcs: list):
    return [(lambda x, i=i: i(x) ** 2) for i in funcs]
