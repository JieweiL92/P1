import math, cv2
import numpy as np

def Display(dataset, r = 0.1, n = 0):
    if len(dataset.shape) == 2:
        data = dataset
    else:
        data = dataset[:,:,n]
    min_d, max_d = np.nanmin(data), np.nanmax(data)
    if max_d>r:
        max_d = r
    data[np.isnan(data)] = min_d
    data[data>r] = r
    coef = 255 / (max_d - min_d)
    data_new = data - min_d
    img = np.around(data_new*coef).astype(np.uint8)
    rows, cols = data.shape
    cv2.resizeWindow('Image1', cols, rows)
    cv2.imshow('Image1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None


def Resize_SigmaNaught(dat, n):
    rows, cols = dat.shape
    nr, nc = math.floor(rows / n), math.floor(cols / n)
    dat = dat[(rows - nr * n):rows, (cols - nc * n):cols]
    new_pic = [[np.nanmean(np.array(dat[r * n:(r + 1) * n, c * n:(c + 1) * n])) for c in range(nc)] for r in range(nr)]
    new_pic = np.array(new_pic, dtype=np.float32)
    return new_pic


def Resize_LL(dat, n):
    rows, cols = dat.shape
    nr, nc = math.floor(rows / n), math.floor(cols / n)
    dat = dat[(rows - nr * n):rows, (cols - nc * n):cols]
    new_LL = [[(dat[r * n, c * n] + dat[r * n, (c + 1) * n - 1] + dat[(r + 1) * n - 1, c * n] + dat[
        (r + 1) * n - 1, (c + 1) * n - 1]) / 4 for c in range(nc)] for r in range(nr)]
    new_LL = np.array(new_LL, dtype=np.float_)
    return new_LL


def CoastLinePosition(Lon, Lat, resolution):
    path = input('Where do you save your coastline data?\nPlease ')
    name = input('The name of your file(do not include the .xyz):')
    f = open(path + '//' + name + '.xyz')
    coastline_dat = []
    for line in f.readlines():
        l = list(map(float, line.split()))
        coastline_dat.append(l)
    f.close()
    coastline_dat = np.array(coastline_dat, dtype=np.float64)
    coastline = coastline_dat[:, 0:2].astype(np.float64)
    del coastline_dat
    rows, cols = Lon.shape
    Yposition = np.empty([1, rows], dtype=np.int32)


def Distance(p0, coastline):
    px, py = p0[1], p0[0]
    d = list(map(lambda x, y: np.sqrt((x - px) ** 2 + (y - py) ** 2), coastline[:, 1], coastline[:, 0]))
    return max(d)
