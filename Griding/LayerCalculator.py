import math, cv2, os
import numpy as np

def Display(dataset, r = 0.1, n = 0, mode = 'linear'):
    if len(dataset.shape) == 2:
        data = dataset
    else:
        data = dataset[:,:,n]
    if mode == 'dB':
        data = np.log10(data) * 10
        data[np.isinf(data)] = np.nan
    min_d, max_d = np.nanmin(data), np.nanmax(data)
    if max_d>r and mode == 'linear':
        max_d = r
    if max_d<r and mode == 'dB':
        max_d = r
    data[np.isnan(data)] = min_d
    data[data>r] = max_d
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
    new_pic = [[np.nanmean(dat[r * n:(r + 1) * n, c * n:(c + 1) * n]) for c in range(nc)] for r in range(nr)]
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


def LoadCoastlineXYZ():
    path = input('Where do you save your coastline data?\nPlease input your file path:')
    name = input('The name of your file(do not include the .xyz):')
    f = open(path + '//' + name + '.xyz')
    coastline_dat = []
    for line in f.readlines():
        l = list(map(float, line.split()))
        coastline_dat.append(l)
    f.close()
    coastline_dat = np.array(coastline_dat, dtype=np.float64)
    coastline = coastline_dat[:, 0:2].astype(np.float64)
    return coastline




def Distance(p0, coastline):
    px, py = p0[1], p0[0]
    d = list(map(lambda x, y: np.sqrt((x - px) ** 2 + (y - py) ** 2), coastline[:, 1], coastline[:, 0]))
    return max(d)


def ReadSubImage(n, root = 'D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/'):
    file_list = os.listdir(root)
    file_list2 = [t for t in file_list if t.find('Sub'+str(n))>0]
    file_list2.sort()
    data= []
    for chr in file_list2:
        temp = np.load(root+chr)
        data.append(temp)
    Ma = np.array(data)
    return Ma


def Cal4Maps(data):
    MeanMap = np.nanmean(data, axis=0, dtype=np.float32)
    MedianMap = np.nanmedian(data, axis=0)
    StdMap = np.nanstd(data, axis=0, dtype=np.float64)
    PeakMap = np.nanmax(data, axis = 0) - np.nanmin(data, axis = 0)
    return MeanMap, MedianMap, StdMap, PeakMap


def Compare(dat, Mean, Median, Std):
    img = (dat-Mean)/Mean

