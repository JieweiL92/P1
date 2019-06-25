from functools import partial
from multiprocessing import Pool
import Internship_RSMAS.Griding.Coordinates as cod
import Internship_RSMAS.Griding.IOcontrol as jo
import math, time
# from memory_profiler import profile
import numpy as np
from numba import jit

level1_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/'
grid_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Grid/'
layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'
coast_root='D:/Academic/MPS/Internship/Data/coastline/'
temp_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Temp/'
tempN = 4


class grid_cell(object):
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
        dat = data[0:n*nr, 0:n*nc]
        temp = np.zeros([nr + 1, nc + 1], dtype=np.float_)
        upleft = dat[::n, ::n]
        upright = dat[::n, n-1::n]
        lowleft = dat[n-1::n, ::n]
        lowright = dat[n-1::n, n-1::n]
        temp[0, 0] = upleft[0, 0]
        temp[0, -1] = upright[0, -1]
        temp[-1, 0] = lowleft[-1, 0]
        temp[-1, -1] = lowright[-1, -1]
        for r in range(1, nr):
            temp[r, 0] = (upleft[r, 0]+lowleft[r-1, 0]) / 2
            temp[r, -1] = (upright[r, -1] + lowright[r - 1, -1]) / 2
        for c in range(1, nc):
            temp[0, c] = (upleft[0, c]+lowleft[0, c-1]) / 2
            temp[-1, c] = (lowleft[-1, c]+lowright[-1, c-1]) / 2
        for r in range(1, nr):
            for c in range(1, nc):
                temp[r, c] = (upleft[r, c] + upright[r, c-1] + lowleft[r-1, c] + lowright[r-1, c-1])/4
        self.__surrounding = temp
        self.__position = (upleft+upright+lowleft+lowright)/4
        return None




class uniform_grid(object):
    def __init__(self):
        self.__v = 8.996e-4         # length in degrees
        self.__h = 1.211e-3         # length in degrees
        self.__size = (2101, 2820)  # pixel numbers
        self.__lon_min = -125.8370
        self.__lat_min = 41.3856

    @property
    def size(self):
        return self.__size
    @property
    def lon_min(self):
        return self.__lon_min
    @property
    def lat_min(self):
        return self.__lat_min
    @property
    def v(self):
        return self.__v
    @property
    def h(self):
        return self.__h

    def MakeGrid(self):
        rows, cols = self.__size
        self.num = np.zeros([rows, cols], dtype=np.uint16)
        self.img = np.zeros([rows, cols], dtype=np.float32)
        return None

    def FindPosition(self, lon, lat):
        lon = (lon - self.__lon_min) / self.__h
        lat = (lat - self.__lat_min) / self.__v
        rows, cols = self.__size
        lon_n = np.floor(lon).astype(np.int16)
        lat_n = np.floor(lat).astype(np.int16)
        del lon, lat
        lat_n = rows - 1 - lat_n
        c_ind = np.where((0<=lon_n) & (lon_n<cols))[0]
        r_ind = np.where((0<=lat_n) & (lat_n<rows))[0]
        Ind = r_ind[np.isin(r_ind, c_ind)]
        del r_ind, c_ind
        return lat_n, lon_n, Ind



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
    if n!=1:
        st = time.time()
        Lon_a = grid_cell()
        Lon_a.Resize(Lon_Arr, n)
        Lon_Arr = Lon_a.position
        Lat_a = grid_cell()
        Lat_a.Resize(Lat_Arr, n)
        Lat_Arr = Lat_a.position
        del Lon_a, Lat_a
        print('It takes %f seconds to resize Longitude and Latitude' % (time.time() - st))
    elif name == 'extract':
        np.save(layer_root+'Longitude.npy', Lon_Arr)
        np.save(layer_root+'Latitude.npy', Lat_Arr)

    st = time.time()
    Grids = uniform_grid()
    Grids.MakeGrid()
    Lon_Arr -= Grids.lon_min
    Lon_Arr /= Grids.h
    Lon_Arr = np.floor(Lon_Arr).astype(np.int16)
    Lat_Arr -= Grids.lat_min
    Lat_Arr /= Grids.v
    Lat_Arr = np.floor(Lat_Arr).astype(np.int16)
    rows, cols = Grids.size
    Lat_Arr = rows-1-Lat_Arr
    sub_lon, upleft = cod.N_sub(tempN, Lon_Arr)
    for iii in range(len(sub_lon)):
        sub_lon[iii] = sub_lon[iii].reshape(-1)
    jo.NMatrix_save(sub_lon, 'Longitude', temp_root)
    jo.Matrix_save(upleft, 'Upper Left Points of LL', temp_root)
    del sub_lon, Lon_Arr
    sub_lat, upleft = cod.N_sub(tempN, Lat_Arr)
    for iii in range(len(sub_lat)):
        sub_lat[iii] = sub_lat[iii].reshape(-1)
    jo.NMatrix_save(sub_lat, 'Latitude', temp_root)
    del sub_lat, Lat_Arr
    print('It takes %f seconds to save Longitude and Latitude\n' % (time.time() - st))
    return None



# *****************************************************************************************************************


class parallelogram_grid(object):
    def __init__(self):
        self.__v = 8.996e-4         # length in degrees
        self.__h = 1.211e-3         # length in degrees
        self.__size = (2101, 2820)
        self.__lon_min = -125.8370
        self.__lat_min = 41.3856
        self.__hup = 1.2338e-3
        self.__hdown = 1.1973e-3

    @property
    def size(self):
        return self.__size

    def MakeGrid(self):
        rows, cols = self.__size
        self.num = np.zeros([rows, cols], dtype=np.uint16)
        self.img = np.zeros([rows, cols], dtype=np.float32)
        return None



@jit(nopython=True, parallel=True)
def Paste(r1, c1, v1, img, num):
    Pixel_value, Pixel_num = img.copy(), num.copy()
    for ind in range(len(c1)):
        Pixel_num[r1[ind], c1[ind]] += 1
        Pixel_value[r1[ind], c1[ind]] += v1[ind]
    return Pixel_value, Pixel_num

# @profile
def IterateParts(rows, cols, n):
    C = np.load(temp_root + 'Longitude_Sub' + str(n+1) + '.npy')
    C_ind= np.where((C >= 0) & (C < cols))[0].astype(np.int32)
    R = np.load(temp_root + 'Latitude_Sub' + str(n+1) + '.npy')
    R_ind = np.where((R >= 0) & (R < rows))[0].astype(np.int32)
    T_ind = R_ind[np.isin(R_ind, C_ind)]
    del R_ind, C_ind
    r1, c1 = R[T_ind], C[T_ind]
    del R, C
    V = np.load(temp_root + 'SigmaNaught_Sub' + str(n+1) + '.npy').reshape(-1)
    v1 = V[T_ind]
    del V, T_ind
    return r1, c1, v1

def GridLL_SigmaNaught(mode = 'uniform'):
    def Preprocess():
        upper_left_xy = jo.Matrix_load('Upper Left Points of LL', temp_root)
        if mode == 'uniform':
            Grids = uniform_grid()
        else:
            Grids = parallelogram_grid()
        Grids.MakeGrid()
        LonMin, LatMin = Grids.lon_min, Grids.lat_min
        h, v = Grids.h, Grids.v
        rows, cols = Grids.size
        return upper_left_xy, Grids, h, v, LonMin, LatMin, rows, cols


    st = time.time()
    upper_left_xy, Grids, h, v, LonMin, LatMin, rows, cols = Preprocess()
    Pn0 = np.zeros_like(Grids.num)
    Pv0 = np.zeros_like(Grids.img)
    st1 = time.time()
    print('Preprocess:', st1-st, 'sec')
    func1 = partial(IterateParts, rows, cols)
    po = Pool(5)
    rcv = po.map(func1, range(tempN**2))
    po.close()
    po.join()
    st2 = time.time()
    print('Multiprocess:', st2-st1, 'sec')
    for i in range(tempN**2):
        r1, c1, v1 = rcv.pop(0)
        Pixel_value, Pixel_num = Paste(r1, c1, v1, Grids.img, Grids.num)
        Pv0 += Pixel_value
        Pn0 += Pixel_num
    st3 = time.time()
    print('Overlay:', st3- st2, 'sec')
    Pixel_num = Pn0.copy()
    Pixel_num[Pixel_num == 0] = 1
    Grids.img = Pv0/Pixel_num
    Grids.num = Pn0
    return Grids.img, Grids.num



# *****************************************************************************************************************




def CoastlineInGrid():

    def LoadCoastlineXYZ():
        name = 't1.xyz'
        f = open(coast_root + name)
        coastline_dat = []
        for line in f.readlines():
            l = list(map(float, line.split()))
            coastline_dat.append(l)
        f.close()
        coastline_dat = np.array(coastline_dat, dtype=np.float64)
        coastline = coastline_dat[:, 0:2].astype(np.float64)
        return coastline

    grd = uniform_grid()
    coastline_LL = LoadCoastlineXYZ()
    R, C, ind = grd.FindPosition(coastline_LL[:, 0], coastline_LL[:, 1])
    R, C = R[ind], C[ind]
    temp = np.zeros([len(ind), 2], dtype=np.int16)
    temp[:, 0], temp[:, 1] = R.copy(), C.copy()
    np.save(coast_root+'Coastline position for each row.npy', temp)
    return temp
