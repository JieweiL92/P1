from numba import jit
import numpy as np
import math


@jit(nopython=True, parallel=True)
def minus(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1]]


@jit(nopython=True, parallel=True)
def CrossProduct(p1, p2):
    return (p1[0] * p2[1] - p1[1] * p2[0])


@jit(nopython=True, parallel=True)
def dot(p1, p2):
    return (p1[0] * p2[0] + p1[1] * p2[1])


@jit(nopython=True, parallel=True)
def BarycentricCoordinate(p, p1, p2, p3):
    vp = minus(p, p1)
    vb = minus(p2, p1)
    vc = minus(p3, p1)
    u_det = CrossProduct(vp, vc)
    v_det = CrossProduct(vb, vp)
    A_det = CrossProduct(vb, vc)
    u = u_det / A_det
    v = v_det / A_det
    w = 1 - u - v
    return w, u, v


def InterpBary(xytf):
    x, y, tf = xytf[0], xytf[1], xytf[2]
    if not tf:
        return np.nan
    else:
        x_fraction, x0 = math.modf(x)
        y_fraction, y0 = math.modf(y)
        pxy0 = np.array([int(y0), int(x0)])
        tri = tri2
        if x_fraction + y_fraction <= 1:
            tri = tri1
        pxy = pxy0 + tri
        slist = [(t[0], t[1]) for t in pxy]
        coef = BarycentricCoordinate([y_fraction, x_fraction], tri[0], tri[1], tri[2])
        result = 1 * coef[0] + 2 * coef[1] + 3 * coef[2]
    return result


tri1 = np.array([[0, 0],
                 [0, 1],
                 [1, 0]])
tri2 = np.array([[1, 1],
                 [1, 0],
                 [0, 1]])
x = 1.8
y = 4.6
tf = True
t = InterpBary([x, y, tf])
print(t)