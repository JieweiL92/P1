import math
import numpy as np
import scipy.interpolate as itp


def GCP_sperate(data):
    # data is a list of points [p1,p2,p3....]  p1=[imx,imy,lon,lat]
    # imy increase, lat increase
    # dt = np.dtype([('x', 'int16'), ('y', 'int16'), ('lon', 'float32'), ('lat', 'float32')])
    dat = np.array(data)
    x = dat[:, 0]
    y = dat[:, 1]
    y = np.max(y) - y
    lon = dat[:, 2]
    lat = dat[:, 3]
    return x, y, lon, lat


def GCP2matrix(data):
    # data is a list of points [p1,p2,p3....]  p1=[imx,imy,lon,lat]
    # imy increase, lat increase
    # dt = np.dtype([('x', 'int16'), ('y', 'int16'), ('lon', 'float32'), ('lat', 'float32')])
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
    # x_vec and y_vec are vectors, lon and lat are matrices
    # x,y vectors and longitude latitude matrices, all of them follow python's convention
    # vectors are strictly ascending, and the matrices start from the upper left point


def L2XY_BaseLayer(data):
    [x, y, lon, lat] = GCP_sperate(data)
    L2X = itp.interp2d(lon, lat, x)
    L2Y = itp.interp2d(lon, lat, y)
    return L2X, L2Y


def XY2L_OtherLayer(data):
    [x, y, lon, lat] = GCP_sperate(data)
    row_num = x.tolist().count(0)
    col_num = y.tolist().count(0)
    x = x.reshape([row_num, col_num])
    y = y.reshape([row_num, col_num])
    x_vec = x[0, :]
    y_vec = np.max(y) - y[:, 0]
    lon = lon.reshape([row_num, col_num])
    lat = lat.reshape([row_num, col_num])
    lon = np.flip(lon, 0)
    lat = np.flip(lat, 0)
    XY2Lon = itp.RectBivariateSpline(y_vec, x_vec, lon)
    XY2Lat = itp.RectBivariateSpline(y_vec, x_vec, lat)
    # XY2Lon = itp.interp2d(x, y, lon)
    # XY2Lat = itp.interp2d(x, y, lat)
    return [XY2Lon, XY2Lat]


# [[1],[2],[3],
#  [4],[5],[6],
#  [7],[8],[9],]
# do not contain the last row/col, except matrices on left and lower side

def N_sub(n, data):
    [y0, x0] = data.shape
    x_len0 = math.ceil(x0 / n)
    y_len0 = math.ceil(y0 / n)
    sub_list, upper_left = [], []
    x_range, y_range = list(range(0, x0, x_len0)), list(range(0, y0, y_len0))
    x_range.append(x0 - 1)
    y_range.append(y0 - 1)
    for i in range(n):
        for j in range(n):
            up, down = y_range[i], y_range[i + 1]
            left, right = x_range[j], x_range[j + 1]
            if i == n - 1:
                down += 1
            if j == n - 1:
                right += 1
            temp = data[up:down, left:right]
            sub_list.append(temp)
            upper_left.append([up, left])
    upper_left = np.array(upper_left)
    return sub_list, upper_left


def N_sub_layers(n, data):
    [y0, x0] = data.shape
    x_len0 = math.ceil(x0 / n)
    y_len0 = math.ceil(y0 / n)
    sub_list, upper_left, vertices = [], [], []
    x_range, y_range = list(range(0, x0, x_len0)), list(range(0, y0, y_len0))
    x_range.append(x0 - 1)
    y_range.append(y0 - 1)
    for i in range(n):
        for j in range(n):
            up, down = y_range[i], y_range[i + 1]
            left, right = x_range[j], x_range[j + 1]
            if i == n - 1:
                down += 1
            if j == n - 1:
                right += 1
            temp = data[up:down, left:right]
            f_pts = [[up, left], [up, right - 1], [down - 1, right - 1], [down - 1, left]]
            f_pts = list(map(lambda t: data[t[0], t[1]], f_pts))
            sub_list.append(temp)
            vertices.append(f_pts)
            upper_left.append([up, left])
    upper_left = np.array(upper_left)
    vertices = np.array(vertices)
    return sub_list, upper_left, vertices


def ExpandGrid(data, n):
    [x_vec, y_vec, lon, lat] = GCP2matrix(data)
    New_x, New_y = x_vec, y_vec
    y_max = y_vec[-1]
    y_inter = y_vec[1:]
    for t in range(n):
        New_y = np.append(New_y, y_inter + y_max)
    y = y_vec
    New_lon = np.empty([len(New_y), len(New_x)], dtype=np.float_)
    New_lat = np.empty_like(New_lon)
    for c in range(len(x_vec)):
        v_lon, v_lat = lon[:, c], lat[:, c]
        f_lon = itp.interp1d(y, v_lon, kind='quadratic', fill_value='extrapolate')
        f_lat = itp.interp1d(y, v_lat, kind='quadratic', fill_value='extrapolate')
        New_lon[:, c] = f_lon(New_y)
        New_lat[:, c] = f_lat(New_y)
    return New_x, New_y, New_lon, New_lat


def Matrix2GCP(x_vec, y_vec, lon, lat):
    y_lim = y_vec[-1] + 1
    x_lim = x_vec[-1] + 1
    y = y_vec.max - y_vec
    xx, yy = np.mgrid(x_vec, y)
    x, y = xx.ravel(), yy.ravel()
    lon_n, lat_n = lon.ravel(), lat.ravel()
    GCP = np.empty([len(x), 4])
    GCP[:, 0], GCP[:, 1] = x, y
    GCP[:, 2], GCP[:, 3] = lon_n, lat_n
    GCP = GCP.tolist()
    return GCP, x_lim, y_lim


if __name__ == '__main__':
    pass
