from functools import partial
from multiprocessing import Pool
import Internship_RSMAS.Griding.Coordinates as cod
import Internship_RSMAS.Griding.IOcontrol as jo
import Internship_RSMAS.Griding.LayerCalculator as Lc
import math, time
import numpy as np
from scipy.interpolate import griddata as grd
from numba import jit

dir = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST'
grid_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Grid/'
layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'
coast_root='D:/Academic/MPS/Internship/Data/coastline/'
temp_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Temp/'

def GCP_Matrix(data):
    dat = np.array(data)
    x = dat[:, 0].astype(np.int32)
    y = dat[:, 1].astype(np.int32)
    lon = dat[:, 2].astype(np.float_)
    lat = dat[:, 3].astype(np.float_)
    row_num = x.tolist().count(0)
    col_num = y.tolist().count(0)
    x = x.reshape([row_num, col_num])
    y = y.reshape([row_num, col_num])
    lon = lon.reshape([row_num, col_num])
    lat = lat.reshape([row_num, col_num])
    if x[0,0]>x[0,-1]:
        x = np.fliplr(x)
        lon = np.fliplr(lon)
        lat = np.fliplr(lat)
    if y[0,0]>y[-1,0]:
        y = np.flipud(y)
        lon = np.flipud(lon)
        lat = np.flipud(lat)
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
        # Lon_Arr = Lc.Resize_LL(Lon_Arr, n)
        # Lat_Arr = Lc.Resize_LL(Lat_Arr, n)
        # [subLon, pt_list] = cod.N_sub(6, Lon_Arr)
        # [subLat, pt_list] = cod.N_sub(6, Lat_Arr)
        # del Lon_Arr, Lat_Arr
        # st = time.time()
        # jo.Matrix_save(pt_list, 'Upper Left Points of SubImages', grid_root)
        # jo.NMatrix_save(subLon, 'Base-Longitude', grid_root)
        # jo.NMatrix_save(subLat, 'Base-Latitude', grid_root)
        # print('It takes %f seconds to save LL data' % (time.time() - st))
        Lon_a = GridClass()
        Lon_a.Resize(Lon_Arr, n)
        Lat_a = GridClass()
        Lat_a.Resize(Lat_Arr, n)
        del Lon_Arr, Lat_Arr
        jo.Matrix_save(Lon_a.surrounding, 'GridCell-Longitude', grid_root)
        jo.Matrix_save(Lat_a.surrounding, 'GridCell-Latitude', grid_root)
        jo.Matrix_save(Lon_a.position, 'Position-Longitude', grid_root)
        jo.Matrix_save(Lat_a.position, 'Position-Latitude', grid_root)
        return None
    else:
        Lon_a = GridClass()
        Lon_a.Resize(Lon_Arr, 5)
        del Lon_Arr
        Lat_a = GridClass()
        Lat_a.Resize(Lat_Arr, 5)
        del Lat_Arr
        sub_lon, upleft = cod.N_sub(n, Lon_a.position)
        jo.NMatrix_save(sub_lon,'Longitude', temp_root)
        jo.Matrix_save(upleft, 'Upper Left Points of LL', temp_root)
        del sub_lon, Lon_a
        sub_lat, upleft = cod.N_sub(n, Lat_a.position)
        jo.NMatrix_save(sub_lat,'Latitude', temp_root)
        del sub_lat, Lat_a
        #
        # sub_lon, upleft = cod.N_sub(n, Lon_Arr)
        # jo.NMatrix_save(sub_lon,'Longitude', temp_root)
        # jo.Matrix_save(upleft, 'Upper Left Points of LL', temp_root)
        # del sub_lon, Lon_Arr
        # sub_lat, upleft = cod.N_sub(n, Lat_Arr)
        # jo.NMatrix_save(sub_lat,'Latitude', temp_root)
        # del sub_lat, Lat_Arr
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


def EachPoint(P_1, P_2, P_3, p):
    result = -1
    w1, u1, v1 = -1, -1, -1
    for t in range(len(P_1)):
        w, u, v = BarycentricCoordinate(p, P_1[t], P_2[t], P_3[t])
        if (0 <= u <= 1) and (0 <= v <= 1) and (0 <= w <= 1):
            result = t
            w1, u1, v1 = w, u, v
            break
    return result, w1, u1, v1

@jit(nopython=True, parallel=True)
def IsInPolygon(p_list, p0):
    P1 = np.array(p_list) - np.array(p0)
    Pt = np.empty_like(P1)
    P2 = np.empty_like(P1)
    Pt[:-1] = P1[1:]
    Pt[-1] = P1[0]
    P2[:, 0] = Pt[:, 1]
    P2[:, 1] = -Pt[:, 0]
    Pt = (P1*P2).sum(axis=1)
    if (Pt<=0).all():
        return True
    else:
        return False


def DistanceP2P(p1, p2):
    v1 = minus(p1, p2)
    result = np.sqrt(v1[0]**2 + v1[1]**2)
    return result




def NewSigmaNaught5(GCP, Sigma_layer, rows_grid, cols_grid):
    # Sigma_layer: NRCS of This SAR image
    # GCP: GCP of this SAR image
    # rows_grid, cols_grid: number of rc of total grid
    def FilterTri(lonMin, lonMax, latMin, latMax):
        ind = [1 if (lonMin <= P_1[t][0] <= lonMax and latMin <= P_1[t][1] <= latMax) or (
                    lonMin <= P_2[t][0] <= lonMax and latMin <= P_2[t][1] <= latMax) or (
                    lonMin <= P_3[t][0] <= lonMax and latMin <= P_3[t][1] <= latMax) else 0 for t in range(len(tri_list))]
        TriN = [tri_list[ii] for ii in range(len(ind)) if ind[ii] ==1]
        P1N = [P_1[ii] for ii in range(len(ind)) if ind[ii] == 1]
        P2N = [P_2[ii] for ii in range(len(ind)) if ind[ii] == 1]
        P3N = [P_3[ii] for ii in range(len(ind)) if ind[ii] == 1]
        x_unitN = [x_unit[ii] for ii in range(len(ind)) if ind[ii] == 1]
        y_unitN = [y_unit[ii] for ii in range(len(ind)) if ind[ii] == 1]
        return TriN, P1N, P2N, P3N, x_unitN, y_unitN

    def FindNewXY(lon_grid, lat_grid):
        rows, cols = lon_grid.shape
        rc_list = [(r, c) for r in range(rows) for c in range(cols)]
        P = [[lon_grid[r, c], lat_grid[r, c]] for r, c in rc_list]
        func = partial(EachPoint, P1N, P2N, P3N)
        # coef = np.array(list(map(func, P)))                           # n,4  [[tri_num, w, u, v][][][]]
        pol = Pool()
        coef = np.array(pol.map(func, P))
        pol.close()
        pol.join()
        Tri_num, Coef_W, Coef_U, Coef_V = coef[:, 0].astype(np.int_), coef[:, 1], coef[:, 2], coef[:, 3]
        del coef
        TF = [True if i >= 0 else False for i in Tri_num]  # point in triangle: True
        R = [TriN[t][0][0] for t in Tri_num]
        C = [TriN[t][0][1] for t in Tri_num]
        R_unit = [y_unitN[t] for t in Tri_num]
        C_unit = [x_unitN[t] for t in Tri_num]
        X_new = [Coef_U[t]*C_unit[t] + xv[C[t]] if TF[t] else np.nan for t in range(len(TF))]
        Y_new = [Coef_V[t]*R_unit[t] + yv[R[t]] if TF[t] else np.nan for t in range(len(TF))]
        return X_new, Y_new, TF

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
            slist = [Sigma_layer[t[0], t[1]] for t in pxy]
            coef = BarycentricCoordinate([y_fraction, x_fraction], tri[0], tri[1], tri[2])
            result = slist[0] * coef[0] + slist[1] * coef[1] + slist[2] * coef[2]
            return result

    def EachRec(lon_arr, lat_arr):
        X, Y, Flag = FindNewXY(lon_arr, lat_arr)
        IterXYFlag = [[X[i], Y[i], Flag[i]] for i in range(len(X))]
        sigmaT = list(map(InterpBary, IterXYFlag))
        rows, cols = lon_arr.shape
        sigmaT = np.array(sigmaT).reshape([rows, cols])

        return sigmaT

    def Pre_Process():
        upper_left_xy = jo.Matrix_load('Upper Left Points of SubImages', grid_root)
        xv, yv, lon_CP, lat_CP = GCP_Matrix(GCP)
        rows, cols = lon_CP.shape
        upleft = [(r, c) for r in range(rows - 1) for c in range(cols - 1)]
        tri_list = [[[r, c], [r, c + 1], [r + 1, c]] for r, c in upleft]
        tri_list2 = [[[r + 1, c + 1], [r + 1, c], [r, c + 1]] for r, c in upleft]
        tri_list.extend(tri_list2)
        del tri_list2, upleft
        P_1 = [[lon_CP[t[0][0], t[0][1]], lat_CP[t[0][0], t[0][1]]] for t in tri_list]
        P_2 = [[lon_CP[t[1][0], t[1][1]], lat_CP[t[1][0], t[1][1]]] for t in tri_list]
        P_3 = [[lon_CP[t[2][0], t[2][1]], lat_CP[t[2][0], t[2][1]]] for t in tri_list]
        x_unit = [xv[t[1][1]] - xv[t[0][1]] for t in tri_list]
        y_unit = [yv[t[2][0]] - yv[t[0][0]] for t in tri_list]
        tri1 = np.array([[0, 0],
                         [0, 1],
                         [1, 0]])
        tri2 = np.array([[1, 1],
                         [1, 0],
                         [0, 1]])
        return tri_list, P_1, P_2, P_3, x_unit, y_unit, tri1, tri2, upper_left_xy, xv, yv, lon_CP, lat_CP


    tri_list, P_1, P_2, P_3, x_unit, y_unit, tri1, tri2, upper_left_xy, xv, yv, lon_CP, lat_CP = Pre_Process()
    Sigma_New = np.zeros([rows_grid, cols_grid], dtype=np.float32)
    for i in range(36):
        print('Grid Number:', i+1)
        st = time.time()
        lon_arr = jo.Matrix_load('Base-Longitude_Sub' + str(i + 1), grid_root)
        lat_arr = jo.Matrix_load('Base-Latitude_Sub' + str(i + 1), grid_root)
        up = upper_left_xy[i, 0]
        left = upper_left_xy[i, 1]
        lonMax, lonMin = lon_arr.max()+0.14, lon_arr.min()-0.14
        latMax, latMin = lat_arr.max()+0.2, lat_arr.min()-0.2
        TriN, P1N, P2N, P3N, x_unitN, y_unitN = FilterTri(lonMin, lonMax, latMin, latMax)
        if len(TriN) != 0:
            s = EachRec(lon_arr, lat_arr)
            rows, cols = s.shape
            Sigma_New[up:up + rows, left:left + cols] = s
        print('Time:', time.time()-st, 'sec')
    return Sigma_New



def MergeGridLL(root = grid_root):
    cp = np.load(root+'Upper Left Points of Subimages.npy')
    lat_name = 'Base-Latitude_Sub'
    lon_name = 'Base-Longitude_Sub'
    up, left = cp[-1,0], cp[-1,1]
    lon_list, lat_list = [],[]
    for i in range(36):
        lon_list.append(np.load(root+lon_name+str(i+1)+'.npy'))
        lat_list.append(np.load(root+lat_name+str(i+1)+'.npy'))
    arr = lat_list[-1]
    rows, cols = arr.shape
    rows, cols = rows+up, cols+left
    lon_grid = np.empty([rows, cols], dtype=np.float_)
    lat_grid = np.empty_like(lon_grid)
    for i in range(36):
        lon_arr, lat_arr = lon_list[i], lat_list[i]
        up, left = cp[i, 0], cp[i, 1]
        r,c =lon_arr.shape
        lon_grid[up:up + r, left:left + c] = lon_arr
        lat_grid[up:up + r, left:left + c] = lat_arr
    return lon_grid, lat_grid


def Line2Nodes(coastline, lon_arr, lat_arr, root=coast_root):
# coastline is a n:2 ndarray
# lon_arr, lat_arr are the longitude and latitude data of the grid
    def EdgePoints(x1, x2, y1, y2):
        Plist = [(lon_arr[y1, x], lat_arr[y1, x]) for x in range(x1, x2)]
        Ptemp = [(lon_arr[y, x2], lat_arr[y, x2]) for y in range(y1, y2)]
        Plist.extend(Ptemp)
        Ptemp = [(lon_arr[y2, x], lat_arr[y2, x]) for x in range(x2, x1,-1)]
        Plist.extend(Ptemp)
        Ptemp = [(lon_arr[y, x1], lat_arr[y, x1]) for y in range(y2, y1,-1)]
        Plist.extend(Ptemp)
        return Plist

    def PointInRect(p0):
        x1, x2, y1, y2 =  0, cols-1, 0, rows-1
        flag = False
        while not flag:
            if x2-x1<=2 and y2-y1<=2:
                flag = True
            else:
                if x2-x1>1:
                    x_un = math.ceil((x2-x1)/3)
                    x_m1 = x1 + x_un
                    x_m2 = x_m1 + x_un
                    px1 = EdgePoints(x1, x_m1, y1, y2)
                    px2 = EdgePoints(x_m1, x_m2, y1, y2)
                    if IsInPolygon(px1, p0):
                        x2 = x_m1
                    elif IsInPolygon(px2, p0):
                        x1, x2 = x_m1, x_m2
                    else:
                        x1 = x_m2
                    if x2 == x1:
                        x1 -=1
                if y2-y1>1:
                    y_un = math.ceil((y2-y1)/3)
                    y_m1 = y1 + y_un
                    y_m2 = y_m1 + y_un
                    py1 = EdgePoints(x1, x2, y1, y_m1)
                    py2 = EdgePoints(x2, x2, y_m1, y_m2)
                    if IsInPolygon(py1, p0):
                        y2 = y_m1
                    elif IsInPolygon(py2, p0):
                        y1, y2 = y_m1, y_m2
                    else:
                        y1 = y_m2
                    if y2 == y1:
                        y1 -= 1
        return x1, x2, y1, y2

    def Nearest(p0, rect):
        x1, x2, y1, y2 = rect[0], rect[1], rect[2], rect[3]
        rc = [(r, c) for r in range(y1,y2+1) for c in range(x1, x2+1)]
        P = [(lon_arr[r,c], lat_arr[r,c]) for r, c in rc]
        func2 = partial(DistanceP2P, p0)
        d = list(map(func2, P))
        ind = d.index(min(d))
        return rc[ind]

    rows, cols = lon_arr.shape
    Surrounding = EdgePoints(0, cols-1, 0, rows-1)
    func1 = partial(IsInPolygon, Surrounding)
    TF = list(map(func1, coastline))
    coast_n = [coastline[i] for i in range(len(TF)) if TF[i]]    # new coastline, delete points outside the grid
    arr = list(map(PointInRect, coast_n))
    coastline_RC = list(map(Nearest, coast_n, arr))
    coast_n = np.array(coastline_RC).astype(np.int32)
    jo.Matrix_save(coast_n, 'Coastline in Grid', root)
    return None


def OneStepCoastline():
    lon_grid, lat_grid = MergeGridLL()
    coastXYZ = Lc.LoadCoastlineXYZ()
    Line2Nodes(coastXYZ, lon_grid, lat_grid)
    line = np.load(coast_root+'/Coastline in Grid.npy')
    return line


def Merge(name, root = layer_root):
    cp = np.load(root + 'Upper Left Points of Subimages.npy')
    up, left = cp[-1, 0], cp[-1, 1]
    arr_list = []
    for i in range(9):
        arr_list.append(np.load(root+name+'_Sub'+str(i+1)+'.npy'))
    rows, cols = arr_list[-1].shape
    rows, cols = rows+up, cols+left
    arr = np.empty([rows, cols], dtype=np.float32)
    for i in range(9):
        up, left = cp[i, 0], cp[i, 1]
        r, c = arr_list[i].shape
        arr[up:up+r, left:left+c] = arr_list[i]
    return arr


class GridClass(object):
    def __init__(self):
        self.__surrounding = 0
        self.__position = 0

    @property
    def surrounding(self):
        return self.__surrounding
    @property
    def position(self):
        return self.__position


    def Resize(self, data, n):
        rows, cols = data.shape
        nr, nc = math.floor(rows / n), math.floor(cols / n)
        dat = data[(rows - nr * n):rows, (cols - nc * n):cols]
        temp = np.empty([nr+1, nc+1], dtype = np.float_)
        temp[:nr, :nc] = dat[::n, ::n]
        temp[nr, :nc] = dat[-1, ::n]
        temp[:nr, nc] = dat[::n, -1]
        temp[-1, -1] = dat[-1, -1]
        self.__surrounding = temp
        temp = [[(dat[r * n, c * n] + dat[r * n, (c + 1) * n - 1] + dat[(r + 1) * n - 1, c * n] + dat[
            (r + 1) * n - 1, (c + 1) * n - 1]) / 4 for c in range(nc)] for r in range(nr)]
        self.__position = np.array(temp, dtype=np.float_)
        return None



def GridSigmaNaught(Sigma):
    def Preprocess_1():
        lon_grid = np.load(grid_root+'GridCell-Longitude.npy')       # longitude of the grid cell(4 points)
        lat_grid = np.load(grid_root+'GridCell-Latitude.npy')        # latitude of the grid cell(4 points)
        upper_left_xy = jo.Matrix_load('Upper Left Points of LL', temp_root)
        rows, cols = lon_grid.shape
        rows, cols = rows-1, cols-1
        Pixel_num = np.zeros([rows, cols], dtype= np.uint16)
        Pixel_value = np.zeros([rows, cols], dtype = np.float32)
        del rows, cols
        return lon_grid, lat_grid, Pixel_num, Pixel_value, upper_left_xy

    def Preprocess_2(n):
        up, left = upper_left_xy[n-1, 0], upper_left_xy[n-1, 1]
        Lon_arr = np.load(temp_root + 'Longitude_Sub' + str(n) + '.npy')
        Lat_arr = np.load(temp_root + 'Latitude_Sub' + str(n) + '.npy')
        ro, co = Lon_arr.shape
        rc = [(r, c) for r in range(ro) for c in range(co)]
        LL = [[Lon_arr[r, c], Lat_arr[r, c]] for r,c in rc]                  # longitude and latitude of new image (in a list)
        del Lon_arr, Lat_arr
        values = [Sigma[r+up, c+left] for r,c in rc]                         # sigma naught value of new image (in a list)
        return LL, values
    #
    #
    # def EdgePoints(x1, x2, y1, y2):
    #     Plist = [(lon_grid[y1, x], lat_grid[y1, x]) for x in range(x1, x2)]
    #     Ptemp = [(lon_grid[y, x2], lat_grid[y, x2]) for y in range(y1, y2)]
    #     Plist.extend(Ptemp)
    #     Ptemp = [(lon_grid[y2, x], lat_grid[y2, x]) for x in range(x2, x1,-1)]
    #     Plist.extend(Ptemp)
    #     Ptemp = [(lon_grid[y, x1], lat_grid[y, x1]) for y in range(y2, y1,-1)]
    #     Plist.extend(Ptemp)
    #     return Plist # clockwise point list in the grid cell array
    #
    # def PointInRect2(p0):
    #     x1, x2, y1, y2 = 0, cols - 1, 0, rows - 1
    #     flag = False
    #     while not flag:
    #         if x2 - x1 <= 1 and y2 - y1 <= 1:
    #             flag = True
    #         else:
    #             if x2 - x1 > 1:
    #                 x_un = math.ceil((x2 - x1) / 3)
    #                 x_m1 = x1 + x_un
    #                 x_m2 = x_m1 + x_un
    #                 px1 = EdgePoints(x1, x_m1, y1, y2)
    #                 px2 = EdgePoints(x_m1, x_m2, y1, y2)
    #                 if IsInPolygon(px1, p0):
    #                     x2 = x_m1
    #                 elif IsInPolygon(px2, p0):
    #                     x1, x2 = x_m1, x_m2
    #                 else:
    #                     x1 = x_m2
    #                 if x2 == x1:
    #                     x1 -= 1
    #             if y2 - y1 > 1:
    #                 y_un = math.ceil((y2 - y1) / 3)
    #                 y_m1 = y1 + y_un
    #                 y_m2 = y_m1 + y_un
    #                 py1 = EdgePoints(x1, x2, y1, y_m1)
    #                 py2 = EdgePoints(x2, x2, y_m1, y_m2)
    #                 if IsInPolygon(py1, p0):
    #                     y2 = y_m1
    #                 elif IsInPolygon(py2, p0):
    #                     y1, y2 = y_m1, y_m2
    #                 else:
    #                     y1 = y_m2
    #                 if y2 == y1:
    #                     y1 -= 1
    #     return x1, x2, y1, y2

    def LL2XY(lon_grid, lat_grid, LL):
        rows, cols = lon_grid.shape
        rc = [(r, c) for r in range(rows) for c in range(cols)]
        points = np.array([[lon_grid[r, c], lat_grid[r, c]] for r, c in rc])
        value_x = np.array([c for r in range(rows) for c in range(cols)])
        value_y = np.array([r for r in range(rows) for c in range(cols)])
        x_position = grd(points, value_x, LL)
        y_position = grd(points, value_y, LL)
        return x_position, y_position

    lon_grid, lat_grid, Pixel_num, Pixel_value, upper_left_xy = Preprocess_1()
    rows, cols = lon_grid.shape
    # WholeArea = EdgePoints(0, cols - 1, 0, rows - 1)
    for t in range(9):
        print('The %d time is '% t)
        st1 = time.time()
        LL, values = Preprocess_2(t+1)
        print('It takes %f seconds to do Preprocess2'%(time.time()-st1))
        # update the LL and values arrays
        # st2 = time.time()
        # func1 = partial(IsInPolygon, WholeArea)
        # po = Pool()
        # TF = po.map(func1, LL)
        # po.close()
        # po.join()
        # # TF = list(map(func1, LL))
        # New_LL = [LL[i] for i in range(len(TF)) if TF[i]]
        # New_values = [values[i] for i in range(len(TF)) if TF[i]]
        # LL, values = New_LL, New_values
        # del New_LL, New_values
        # print('It takes %f seconds to update'%(time.time()-st2))
        # st2 = time.time()
        # po = Pool()
        # Cell_Ind = np.array(po.map(PointInRect2, LL))      # return a n*4 array
        # po.close()
        # po.join()
        # print('It takes %f seconds to find the grid cell' % (time.time() - st2))
        # for t1 in range(Cell_Ind.shape[0]):
        #     x, y = Cell_Ind[t1, 0], Cell_Ind[t1, 2]
        #     Pixel_value[y, x] = (Pixel_value[y, x] * Pixel_num[y, x] + values[t1])/(Pixel_num[y, x]+1)
        #     Pixel_num[y, x] += 1
        # print('It takes %f seconds to deal with 1 part'%(time.time()-st1))
        x_p, y_p = LL2XY(lon_grid, lat_grid, LL)
        x_p[np.isnan(x_p)] = -20
        y_p[np.isnan(y_p)] = -20
        x_p = np.floor(x_p).astype(np.int16)
        y_p = np.floor(y_p).astype(np.int16)
        print(len(x_p))
        for ii in range(len(values)):
            x, y = x_p[ii], y_p[ii]
            if x>=0 and y>=0:
                Pixel_value[y, x] = (Pixel_value[y, x] * Pixel_num[y, x] + values[ii]) / (Pixel_num[y, x] + 1)
                Pixel_num[y, x] += 1

    return Pixel_value, Pixel_num


if __name__ == '__main__':
    pass
