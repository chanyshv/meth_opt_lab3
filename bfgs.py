import typing as ty

import numpy as np

from descent import wolfe_conditions


def rho_calc(y, s):
    return 1 / np.dot(y.T, s)


def bfgs(f: ty.Callable, df: ty.Callable, x_0, epochs, tol=1e-5, ls_tol=1e-5):
    i = 0
    E = np.eye(len(x_0))

    H_i = E  # reversed (^-1) hessian
    points = [x_0]
    x_i = x_0  # point
    df_x_i = df(f, x_0)  # derivative at x_i

    while i < epochs and np.linalg.norm(df_x_i) > tol:
        p_i = -H_i @ df_x_i
        t, f_calls, df_calls = wolfe_conditions(f, df, x_i, p_i, ls_tol)

        s = t * p_i
        x_i1 = x_i + s
        s = np.reshape(np.array([s]), (len(x_0), 1))
        y = np.reshape(np.array([df(f, x_i1) - df(f, x_i)]), (len(x_0), 1))

        rho = 1 / (y.T @ s)
        m1 = E - rho * (s @ y.T)
        m2 = E - rho * (y @ s.T)

        H_i = m1 @ H_i @ m2 + rho * (s @ s.T)
        x_i = x_i1[:]
        points.append(x_i)
        df_x_i = df(f, x_i)

        i += 1

    print(len(points))
    return points


# m --- is size of "batch" of precompiled data
def l_bfgs(f: ty.Callable, df: ty.Callable, x_0, epochs, m=5, tol=1e-9, ls_tol=1e-6):
    points = [x_0]
    x_i = x_0  # point
    df_x_i = df(f, x_0)  # derivative at x_i
    p = -df_x_i
    t, _, _ = wolfe_conditions(f, df, x_0, p, tol=ls_tol)

    ys = []
    ss = []

    ss.append(t * p)
    x_i1 = x_i + ss[-1]
    points.append(x_i1)
    df_x_i1 = df(f, x_i1)

    ys.append(df_x_i1 - df_x_i)
    df_x_i = df_x_i1
    x_i = x_i1

    def alpha_calc(j, q):
        return rho_calc(ys[len(ss) - 1 - j], ss[len(ss) - 1 - j]) * np.dot(
            ss[len(ss) - 1 - j].T, q
        )

    def calcualte_p():
        q = df(f, x_i)
        alphas = [None] * len(ss)

        for j in range(len(ss)):
            alphas[len(ss) - 1 - j] = alpha_calc(j, q)
            q = q - alphas[len(ss) - 1 - j] * ys[len(ss) - 1 - j]

        lambd = np.dot(ss[-1].T, ys[-1]) / np.dot(ys[-1].T, ys[-1])
        H_i = lambd * np.eye(len(ss[-1]))
        p_i = np.dot(H_i, q)

        for j in range(len(ss)):
            betta = rho_calc(ys[j], ss[j]) * np.dot(ys[j].T, p_i)
            p_i = p_i + ss[j] * (alphas[j] - betta)

        return -p_i

    i = 0
    # calculating reversed hessian recursively
    while i < epochs and np.linalg.norm(df_x_i) > tol:
        p_i = calcualte_p()

        t, f_calls, df_calls = wolfe_conditions(f, df, x_i, p_i, tol=1e-6)

        s = t * p_i
        x_i1 = x_i + s
        y = (df(f, x_i1) - df(f, x_i)).T

        ys.append(y)
        ss.append(s)

        if len(ss) == m + 1:
            ys.pop(0)
            ss.pop(0)

        x_i = x_i1
        points.append(x_i)
        df_x_i = df(f, x_i)

        i += 1

    return points
