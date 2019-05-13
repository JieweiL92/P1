import Read_SentinelData.SentinelClass as rd
import math
import numpy as np
from numba import jit
from scipy.spatial import Delaunay

dir = 'D:\\Academic\\MPS\\Internship_RSMAS\\Data\\Sentinel\\TEST'


def GCP_Matrix(data):
    dat = np.array(data)
    x = dat[:, 0].astype(np.int32)
    y = dat[:, 1].astype(np.int32)
    y = np.max(y) - y
    lon = dat[:, 2].astype(np.float_)
    lat = dat[:, 3].astype(np.float_)
    row_num = x.tolist().count(0)
    col_num = y.tolist().count(0)
    x = x.reshape([row_num, col_num])
    y = y.reshape([row_num, col_num])
    lon = lon.reshape([row_num, col_num])
    lat = lat.reshape([row_num, col_num])
    lon = np.flip(lon, 0)
    lat = np.flip(lat, 0)
    y = np.flip(y, 0)
    x_vec = x[0, :]
    y_vec = y[:, 0]
    return x_vec, y_vec, lon, lat


@jit(nopython=True, parallel=True)
def minus(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1]]


@jit(nopython=True, parallel=True)
def CrossProduct(p1, p2):
    return (p1[0] * p2[1] - p1[1] * p2[0])


def IsInRect(p, pA, pB, pC, pD):
    flag = False
    l_AB = CrossProduct(minus(pB, pA), minus(p, pA))
    l_BC = CrossProduct(minus(pC, pB), minus(p, pB))
    l_CD = CrossProduct(minus(pD, pC), minus(p, pC))
    l_DA = CrossProduct(minus(pA, pD), minus(p, pD))
    if (l_AB > 0 and l_BC > 0 and l_CD > 0 and l_DA > 0) or \
            (l_AB < 0 and l_BC < 0 and l_CD < 0 and l_DA < 0) or (l_AB * l_BC * l_CD * l_DA == 0):
        flag = True
    return flag


def TranferLL(arr1, arr2, x1, x2, y1, y2):
    p1 = [arr1[y1, x1], arr2[y1, x1]]
    p2 = [arr1[y1, x2], arr2[y1, x2]]
    p3 = [arr1[y2, x2], arr2[y2, x2]]
    p4 = [arr1[y2, x1], arr2[y2, x1]]
    return p1, p2, p3, p4


def FindRect(arr1, arr2, p0):
    def Divide3(t1, t2):
        if t2 - t1 == 1:
            return t1, t2, 0, 0
        else:
            unit_t = math.ceil((t2 - t1) / 3)
            t_m1 = t1 + unit_t
            t_m2 = t2 - unit_t
            return t1, t_m1, t_m2, t2

    def Tranfer(x1, x2, y1, y2):
        p1 = [arr1[y1, x1], arr2[y1, x1]]
        p2 = [arr1[y1, x2], arr2[y1, x2]]
        p3 = [arr1[y2, x2], arr2[y2, x2]]
        p4 = [arr1[y2, x1], arr2[y2, x1]]
        return p1, p2, p3, p4

    def SearchRect(p, x1, x2, y1, y2):
        flag = True
        x0, y0 = x1, y1
        while flag:
            if x2 - x1 == 1 and y2 - y1 == 1:
                x0, y0 = x1, y1
                flag = False
            else:
                print(x1, x2, y1, y2)
                x_mid = math.ceil((x1 + x2) / 2)
                y_mid = math.ceil((y1 + y2) / 2)
                sub_rec = [[x1, x_mid, y1, y_mid],
                           [x_mid, x2, y1, y_mid],
                           [x1, x_mid, y_mid, y2],
                           [x_mid, x2, y_mid, y2]]
                del_set = set()
                if y2 == y_mid:
                    del_set.add(2)
                    del_set.add(3)
                if x2 == x_mid:
                    del_set.add(1)
                    del_set.add(3)
                del_list = list(del_set)
                del_list.sort(reverse=True)
                for a in del_list:
                    sub_rec.pop(a)
                for r in sub_rec:
                    p1, p2, p3, p4 = Tranfer(r[0], r[1], r[2], r[3])
                    if IsInRect(p, p1, p2, p3, p4):
                        x1, x2, y1, y2 = r[0], r[1], r[2], r[3]
                        break
        return x0, y0

    def SearchRect3(p, x1, x2, y1, y2):
        flag = True
        xx1, xx2, yy1, yy2 = x1, x2, y1, y2
        x0, y0 = x1, y1
        while flag:
            if xx2 - xx1 == 1 and yy2 - yy1 == 1:
                x0, y0 = xx1, yy1
                flag = False
            else:
                print(xx1, xx2, yy1, yy2)
                a1, a_m1, a_m2, a2 = Divide3(xx1, xx2)
                if a2 == 0:
                    xx1, xx2 = a1, a_m1
                else:
                    p1, p2, p3, p4 = Tranfer(a1, a_m1, yy1, yy2)
                    p5, p6, p7, p8 = Tranfer(a_m2, a2, yy1, yy2)
                    if IsInRect(p, p1, p2, p3, p4):
                        xx1, xx2 = a1, a_m1
                    elif IsInRect(p, p5, p6, p7, p8):
                        xx1, xx2 = a_m2, a2
                    else:
                        xx1, xx2 = a_m1, a_m2

                b1, b_m1, b_m2, b2 = Divide3(yy1, yy2)
                if b2 == 0:
                    yy1, yy2 = b1, b_m1
                else:
                    p1, p2, p3, p4 = Tranfer(xx1, xx2, b1, b_m1)
                    p5, p6, p7, p8 = Tranfer(xx1, xx2, b_m2, b2)
                    if IsInRect(p, p1, p2, p3, p4):
                        yy1, yy2 = b1, b_m1
                    elif IsInRect(p, p5, p6, p7, p8):
                        yy1, yy2 = b_m2, b2
                    else:
                        yy1, yy2 = b_m1, b_m2
        return x0, y0

    x1, y1 = 0, 0
    y2, x2 = arr1.shape
    x2 -= 1
    y2 -= 1
    p1, p2, p3, p4 = Tranfer(x1, x2, y1, y2)
    if IsInRect(p0, p1, p2, p3, p4):
        left, up = SearchRect3(p0, x1, x2, y1, y2)
        return left, up
    else:
        return -10, 10


if __name__ == '__main__':
    d = rd.SentinelData()
    d.Get_List(dir)
    indice = 1
    temp = rd.Data_Level1(d.series[indice], d.FList[indice])
    temp.Get_Measure_Data()
    GCP = temp.GCPs
    xv, yv, lon_CP, lat_CP = GCP_Matrix(GCP)

    i = 10
    P = [-121.13127135479978, 36.34328420567563]
    x, y = FindRect(lon_CP, lat_CP, P)
    print(x, y)
    print("************************")
    x1 = 13
    x2 = 20
    y1 = 0
    y2 = 3
    print(x1, x2, y1, y2)
    p1, p2, p3, p4 = TranferLL(lon_CP, lat_CP, x1, x2, y1, y2)
    print(IsInRect(P, p1, p2, p3, p4))
    tri = Delaunay
