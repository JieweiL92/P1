from functools import partial

import Griding.LayerCalculator as Lc
import math
import numpy as np
import time
from Griding import Coordinates as cod
from Griding import IOcontrol as jo
from numba import jit
from scipy import sparse


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


def AllLonLat(name, x0, y0, GCP, n):
    @jit(nopython=True, parallel=True)
    def LinerInterp(x, x1, x2, f1, f2):
        return ((x2 - x) * f1 + (x - x1) * f2) / (x2 - x1)

    @jit(nopython=True, parallel=True)
    def BilinearInterp(x, y, x1, x2, y1, y2, f11, f12, f21, f22):  # f(x1,y1), f(x1,y2)
        ty1 = LinerInterp(x, x1, x2, f11, f21)
        ty2 = LinerInterp(x, x1, x2, f12, f22)
        t = LinerInterp(y, y1, y2, ty1, ty2)
        return t

    @jit
    def Travel(x_range, y_range, lon, lat):
        Lon_Arr, Lat_Arr = np.zeros([y0, x0], dtype=np.float_), np.zeros([y0, x0], dtype=np.float_)
        for i in range(len(x_range) - 1):
            x1, x2 = x_range[i], x_range[i + 1]
            for j in range(len(y_range) - 1):
                y1, y2 = y_range[j], y_range[j + 1]
                p11, p12, p21, p22 = lon[j, i], lon[j + 1, i], lon[j, i + 1], lon[j + 1, i + 1]
                q11, q12, q21, q22 = lat[j, i], lat[j + 1, i], lat[j, i + 1], lat[j + 1, i + 1]
                for x in range(x1, x2 + 1):
                    for y in range(y1, y2 + 1):
                        Lon_Arr[y, x] = BilinearInterp(x, y, x1, x2, y1, y2, p11, p12, p21, p22)
                        Lat_Arr[y, x] = BilinearInterp(x, y, x1, x2, y1, y2, q11, q12, q21, q22)
        return Lon_Arr, Lat_Arr

    [x_range, y_range, lon, lat] = GCP_Matrix(GCP)
    st = time.time()
    [Lon_Arr, Lat_Arr] = Travel(x_range, y_range, lon, lat)

    print('It takes %f seconds to calculate the Longitude and Latitude' % (time.time() - st))
    if name == 'Base':
        root = 'D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/BaseLayer_LL/'
        Lon_Arr = Lc.Resize_LL(Lon_Arr, n)
        Lat_Arr = Lc.Resize_LL(Lat_Arr, n)
        [subLon, pt_list] = cod.N_sub(6, Lon_Arr)
        [subLat, pt_list] = cod.N_sub(6, Lat_Arr)
        del Lon_Arr, Lat_Arr
        st = time.time()
        jo.Matrix_save(pt_list, 'Upper Left Points of SubImages', root)
        jo.NMatrix_save(subLon, 'Base-Longitude', root)
        jo.NMatrix_save(subLat, 'Base-Latitude', root)
        print('It takes %f seconds to save LL data' % (time.time() - st))
    else:
        root = 'D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/Temp/'
        up_left_list = np.empty([(len(x_range) - 1) * (len(y_range) - 1), 2], dtype=np.int32)
        n = 0
        for y in range(len(y_range) - 1):
            for x in range(len(x_range) - 1):
                x1, x2 = x_range[x], x_range[x + 1]
                y1, y2 = y_range[y], y_range[y + 1]
                subLon, subLat = Lon_Arr[y1:y2 + 1, x1:x2 + 1], Lat_Arr[y1:y2 + 1, x1:x2 + 1]
                n += 1
                up_left_list[n - 1, 0] = y1  # up
                up_left_list[n - 1, 1] = x1  # left
                jo.Matrix_save(subLon, name + '-Longitude' + str(n), root)
                jo.Matrix_save(subLat, name + 'Latitude' + str(n), root)
        jo.Matrix_save(up_left_list, 'Upper Left Points of SubImages', root)
    return None


@jit(nopython=True, parallel=True)
def minus(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1]]


@jit(nopython=True, parallel=True)
def CrossProduct(p1, p2):
    return (p1[0] * p2[1] - p1[1] * p2[0])


@jit(nopython=True, parallel=True)
def dot(p1, p2):
    return (p1[0] * p2[0] + p1[1] * p2[1])


@jit
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


@jit(nopython=True, parallel=True)
def minus_without_jit(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1]]


# delete the xp outside the GCP grid
def TFlist(dat):
    if dat[0] < 0:
        return False
    else:
        return True


def BarycentricCoordinate(p, p1, p2, p3):
    vp = minus_without_jit(p, p1)
    vb = minus_without_jit(p2, p1)
    vc = minus_without_jit(p3, p1)
    u_det = np.linalg.det(np.array([[vp[0], vc[0]],
                                    [vp[1], vc[1]]]))
    v_det = np.linalg.det(np.array([[vb[0], vp[0]],
                                    [vb[1], vp[1]]]))
    A_det = np.linalg.det(np.array([[vb[0], vc[0]],
                                    [vb[1], vc[1]]]))
    u = u_det / A_det
    v = v_det / A_det
    w = 1 - u - v
    return w, u, v


def NewSigmaNaught4(GCP, Sigma_layer, rows_grid, cols_grid):
    def FindRect(arr1, arr2, p0):

        def Divide3(t1, t2):
            if t2 - t1 == 1:
                return t1, t2, 0, 0
            else:
                unit_t = math.ceil((t2 - t1) / 3)
                t_m1 = t1 + unit_t
                t_m2 = t_m1 + unit_t
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
                    a1, a_m1, a_m2, a2 = Divide3(xx1, xx2)
                    if a2 == 0:
                        xx1, xx2 = a1, a_m1
                    else:
                        p1, p2, p3, p4 = Tranfer(a1, a_m1, yy1, yy2)
                        p5, p6, p7, p8 = Tranfer(a_m1, a_m2, yy1, yy2)
                        if IsInRect(p, p1, p2, p3, p4):
                            xx1, xx2 = a1, a_m1
                        elif IsInRect(p, p5, p6, p7, p8):
                            xx1, xx2 = a_m1, a_m2
                        else:
                            xx1, xx2 = a_m2, a2

                    b1, b_m1, b_m2, b2 = Divide3(yy1, yy2)
                    if b2 == 0:
                        yy1, yy2 = b1, b_m1
                    else:
                        p1, p2, p3, p4 = Tranfer(xx1, xx2, b1, b_m1)
                        p5, p6, p7, p8 = Tranfer(xx1, xx2, b_m1, b_m2)
                        if IsInRect(p, p1, p2, p3, p4):
                            yy1, yy2 = b1, b_m1
                        elif IsInRect(p, p5, p6, p7, p8):
                            yy1, yy2 = b_m1, b_m2
                        else:
                            yy1, yy2 = b_m2, b2

            return x0, y0

        x1, y1 = 0, 0
        y2, x2 = arr1.shape
        x2 -= 1
        y2 -= 1
        p1, p2, p3, p4 = Tranfer(x1, x2, y1, y2)
        if IsInRect(p0, p1, p2, p3, p4):
            left, up = SearchRect(p0, x1, x2, y1, y2)
            return left, up
        else:
            return -10, 10

    def EachRec(lon_grid, lat_grid):
        nonlocal Sigma_layer  # upper_left_xy, lon_CP, lat_CP, dist_xy, dist_LL
        rows, cols = lon_grid.shape
        rc_list = [(r, c) for r in range(rows) for c in range(cols)]
        P = [[lon_grid[r, c], lat_grid[r, c]] for r, c in rc_list]
        Func = partial(FindRect, lon_CP, lat_CP)

        st1 = time.time()
        xy_GCP = list(map(Func, P))
        flg = list(map(TFlist, xy_GCP))
        print(time.time() - st1, 'seconds')
        # refresh the points, delete the useless points
        rc_list = np.array([rc_list[rci] for rci in range(len(flg)) if flg[rci]])
        xy_GCP = [xy_GCP[rci] for rci in range(len(xy_GCP)) if flg[rci]]

        # get x,y value of the node in the new image's frame---x_new, y_new
        xy_unit = np.array([dist_xy[str(yp) + ',' + str(xp)] for xp, yp in xy_GCP])
        P_1, P_2, P_3 = [dist_LL[str(yp) + ',' + str(xp)][0] for xp, yp in xy_GCP], [dist_LL[str(yp) + ',' + str(xp)][1]
                                                                                     for xp, yp in xy_GCP], [
                            dist_LL[str(yp) + ',' + str(xp)][2] for xp, yp in xy_GCP]
        coef = np.array(list(map(BarycentricCoordinate, P, P_1, P_2, P_3)))
        del P_1, P_2, P_3, P

        x_new = coef[:, 1] * xy_unit[:, 0] + np.array([xv[xp] for xp, yp in xy_GCP])
        y_new = coef[:, 2] * xy_unit[:, 1] + np.array([yv[yp] for xp, yp in xy_GCP])
        x0, y0 = np.array(list(map(math.floor, x_new))), np.array(list(map(math.floor, y_new)))
        x_t, y_t = x_new - x0, y_new - y0
        P = [[y_t[t], x_t[t]] for t in range(len(x0))]
        tri_num = list(map(math.floor, x_t + y_t))
        P_1, P_2, P_3 = [tri_list[int(t)][0] for t in tri_num], [tri_list[int(t)][1] for t in tri_num], [
            tri_list[int(t)][2] for t in tri_num]
        s_list = np.array(
            [[Sigma_layer[y0[t], x0[t]], Sigma_layer[y0[t], x0[t] + 1], Sigma_layer[y0[t] + 1, x0[t]]] for t in
             range(len(xy_GCP))])
        del x0, y0, x_t, y_t, x_new, y_new, tri_num, xy_unit, xy_GCP

        coef = np.array(list(map(BarycentricCoordinate, P, P_1, P_2, P_3)))
        SigmaValue = (coef * s_list).sum(1)
        del P, P_3, P_2, P_1, s_list, coef
        grid_s = sparse.coo_matrix((SigmaValue, (rc_list[:, 0], rc_list[:, 1])), shape=(rows, cols))
        grid_s = grid_s.toarray()
        return grid_s

    b_root = 'D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/BaseLayer_LL/'
    upper_left_xy = jo.Matrix_load('Upper Left Points of SubImages', b_root)
    xv, yv, lon_CP, lat_CP = GCP_Matrix(GCP)

    dist_LL = {}  # item: 3 points list (y,x) (y,x+1) (y+1,x)
    dist_xy = {}  # item: length in x direction, length in y direction
    for r in range(len(yv) - 1):
        for c in range(len(xv) - 1):
            dist_LL[str(r) + ',' + str(c)] = [[lon_CP[r, c], lat_CP[r, c]],
                                              [lon_CP[r, c + 1], lat_CP[r, c + 1]],
                                              [lon_CP[r + 1, c], lat_CP[r + 1, c]]]
            dist_xy[str(r) + ',' + str(c)] = [xv[c + 1] - xv[c], yv[r + 1] - yv[r]]

    # tri_list = [[0, 1, 3], [1, 2, 3]]
    tri_list = [[[0, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 1], [1, 0]]]
    Sigma_New = np.empty([rows_grid, cols_grid], dtype=np.float32)
    for i in range(36):
        lon_arr = jo.Matrix_load('Base-Longitude_Sub' + str(i + 1), b_root)
        lat_arr = jo.Matrix_load('Base-Latitude_Sub' + str(i + 1), b_root)
        up = upper_left_xy[i, 0]
        left = upper_left_xy[i, 1]
        st = time.time()
        print('grid: ', i + 1)
        sub_grid = EachRec(lon_arr, lat_arr)
        print('time: ', time.time() - st, ' seconds\n')
        rows, cols = sub_grid.shape
        Sigma_New[up:up + rows, left:left + cols] = sub_grid

        # for r in range(rows):
        #     for c in range(cols):
        #         real_x, real_y = left+c, up+r             # the xy position in the grid
        #         p = [lon_arr[r,c], lat_arr[r,c]]          # lon lat of the node
        #         xp, yp = FindRect(p, lon_CP, lat_CP)      # xy position in the GCP matrix
        #         if xp>=0:
        #             p1, p2, p3 = dist_LL[str(yp)+','+str(xp)]
        #             x_unit, y_unit = dist_xy[str(yp)+','+str(xp)]
        #             [w, u, v] = BarycentricCoordinate(p, p1, p2, p3)
        #             x_new = u*x_unit + xv[xp]                 # x value of the node in the new image's frame
        #             y_new = v*y_unit + yv[yp]                 # y value of the node in the new image's frame
        #             x0, y0 = math.floor(x_new), math.floor(y_new)
        #             x_t = x_new - x0
        #             y_t = y_new - y0
        #             tri_num = 0
        #             if x_t + y_t > 1:
        #                 tri_num = 1
        #             s_list = [Sigma_layer[y0, x0], Sigma_layer[y0, x0+1], Sigma_layer[y0+1, x0+1], Sigma_layer[y0+1, x0]]
        #             a, b, c = tri_list[tri_num]
        #             [w, u, v] = BarycentricCoordinate([y_t, x_t], pt_list[a], pt_list[b], pt_list[c])
        #             Sigma_New[real_y, real_x] = w*s_list[a] + u*s_list[b] + v*s_list[c]
        #         else:
        #             Sigma_New[real_y, real_x] = np.nan

    return Sigma_New
