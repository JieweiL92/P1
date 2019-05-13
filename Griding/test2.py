import Griding.IOcontrol as jo
import Read_SentinelData.SentinelClass as rd
import math
import numpy as np
from numba import jit

dir = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST'


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


if __name__ == '__main__':
    d = rd.SentinelData()
    d.Get_List(dir)
    indice = 1
    temp = rd.Data_Level1(d.series[indice], d.FList[indice])
    temp.Get_Measure_Data()
    GCP = temp.GCPs
    xv, yv, lon_CP, lat_CP = GCP_Matrix(GCP)
