import math
import random
import typing as ty
import numpy as np

from profiler import Num, to_num


def numeric_gradient(f, x, h=1e-6):
    """
    Функция для численного вычисления градиента функции f в точке x.
    Параметры:
        x: точка, в которой нужно вычислить градиент
        h: малое число для вычисления приближенного значения производной
    """
    n = x.shape[0]  # число переменных
    grad = np.zeros(n)  # инициализация градиента
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h  # прибавляем малое число к i-ой координате
        x_minus = x.copy()
        x_minus[i] -= h  # вычитаем малое число из i-ой координаты
        grad[i] = (f(x_plus) - f(x_minus)) / (
            2 * h
        )  # вычисляем приближенное значение производной
    return grad


def dichotomy(f, a, b, l):
    """
    Реализация лиенйного поиска с помощью метода дихотомии

    :param function f
    :param float a: начало отрезка
    :param float b: конец отрезка
    :param float l: искомая длина шага
    """
    eps = l / 5
    func_calls = 0
    while (b - a) > l:
        y = (a + b + eps) / 2
        z = (a + b - eps) / 2
        if f(y) > f(z):
            b = z
        else:
            a = y

        func_calls += 2
    return func_calls, (a + b) / 2


def wolfe_conditions(f: ty.Callable, df: ty.Callable, x, d, c1=0.01, c2=0.99, tol=1e-4):
    f_x = f(x)
    grad_f_x = df(f, x)
    t = 1.0

    f_calls = df_calls = 1

    while t > tol and (
        f(x + t * d) > f_x + c1 * t * grad_f_x.dot(d)
        or df(f, x + t * d).dot(d) < c2 * grad_f_x.dot(d)
    ):  # пока не выполняются условия Вольфе, уменьшаем шаг
        t *= c2
        f_calls += 1
        df_calls += 1

    return t, f_calls, df_calls

def constant_lr_decay(lr):
    def f(*_, **__):
        return lr

    return f


def exp_decay(lr0, k):
    def f(epoch):
        return lr0 * math.exp(-k * epoch)

    return f


def step_decay(lr0, drop, epochs_drop):
    def f(epoch):
        return lr0 * (drop ** (epoch // epochs_drop))

    return f


def adam_minibatch_descent(
    f: ty.List[ty.Callable],
    df,
    x0,
    decay,
    tol,
    n_epochs,
    batch_size=1,
    betta1=0.9,
    betta2=0.9,
    silent=False,
    global_stats=False,
):
    random.shuffle(f)
    n = len(f)
    points = [x0]
    grad_square_steps = [np.array([Num(0)] * len(x0), dtype=Num)]
    grad_steps = [np.array([Num(0)] * len(x0), dtype=Num)]
    eps = 1e-8

    last_grads = np.array([Num(0.0) for _ in range(len(x0))], dtype=Num)
    num_grads = 0
    last_epoch = 0

    for i in range(n // batch_size * n_epochs):
        if i * batch_size // n > last_epoch:
            if np.linalg.norm(last_grads / num_grads) < tol:
                if not silent:
                    print(np.linalg.norm(last_grads / num_grads))
                break
            last_grads = 0.0
            num_grads = 0
            last_epoch = i * batch_size // n

        x = points[-1]
        part_grad = np.array(
            to_num(
                sum(df(f[(i * batch_size + j) % n], x) for j in range(batch_size))
                / batch_size
            ),
            dtype=Num,
        )

        last_grads += part_grad
        num_grads += 1

        grad_square_steps.append(
            grad_square_steps[-1] * betta1 + part_grad**2 * (1 - betta1)
        )
        grad_steps.append(grad_steps[-1] * betta2 + part_grad * (1 - betta2))
        grad_step_normalized = grad_steps[-1] / (1 - betta1 ** (i + 1))
        grad_square_step_normalized = grad_square_steps[-1] / (1 - betta2 ** (i + 1))
        points.append(
            x
            - decay(i * batch_size // n)
            * grad_step_normalized
            * (grad_square_step_normalized + eps) ** (-1 / 2)
        )

    if global_stats:
        global _stats
        _stats.append(len(points))
    return points


def grad_descent_with_dichotomy(f, df, x0, lr, tol=0.01, epoch=1000):
    """
    :param f: Исследуемая функция
    :param df: Функция вычисления градиента
    :param x0: Начальная точка
    :param lr: Learning rate
    :param tol: Условие останова по достижению необходимой точности (нормы градиента)
    :param epoch: Ограничение числа итераций
    :return: Кортеж из точек, кол-ва вызовов функции f, количества итераций
    """
    points = [x0]
    for i in range(1, epoch):
        x = points[-1]
        grad = df(f, x)
        if np.linalg.norm(grad) < tol:
            break
        func_calls, t = dichotomy(lambda t: f(x - t * grad), 0, lr, lr / 1e3)
        points.append(x - t * grad)

    return points


def gauss_newton_descent(x0: np.ndarray, rsl, grad, tol=1e-8, max_iter=40):
    points = [x0.copy()]
    p = x0
    for i in range(max_iter):
        J = np.array([grad(ri, p).T for ri in rsl])
        r = np.array([ri(p) for ri in rsl], dtype=np.float64)
        dp = np.linalg.pinv(J.T @ J) @ J.T @ r
        p -= dp
        points.append(p.copy())
        if np.linalg.norm(dp) < tol:
            break
    return points


def powell_dog_leg(x0, rsl, grad, tol=1e-8, max_iter=10, delta0=1.0):
    points = [x0.copy()]
    p = x0
    delta = delta0
    for k in range(max_iter):
        J = np.array([grad(ri, p) for ri in rsl])
        r = np.array([ri(p) for ri in rsl], dtype=np.float64)
        B = J.T @ J
        dp_gn = np.linalg.pinv(B) @ J.T @ r
        if np.linalg.norm(dp_gn) <= delta:
            dp = dp_gn
        else:
            g = J.T @ r
            dp_sd = g.T @ g / (g.T @ B @ g) * g
            if np.linalg.norm(dp_sd) >= delta:
                dp = dp_sd / np.linalg.norm(dp_sd) * delta
            else:
                alpha = np.linalg.norm(dp_gn - dp_sd) ** 2 / (
                    2 * (np.linalg.norm(dp_gn) ** 2 - np.linalg.norm(dp_sd) ** 2)
                )
                dp = alpha * dp_gn + (1 - alpha) * dp_sd
        p = p - dp
        points.append(p.copy())
        if np.linalg.norm(dp) < tol:
            break
        rho = (
            np.linalg.norm(r) - np.linalg.norm([ri(p - dp) for ri in rsl])
        ) / np.linalg.norm(dp)
        if rho > 0.75:
            delta = max(delta, 2 * np.linalg.norm(dp))
        if rho < 0.25:
            delta = 0.5 * delta

    return points
