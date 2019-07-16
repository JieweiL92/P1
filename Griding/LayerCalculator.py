import math, cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import griddata
from Internship_RSMAS.Griding import Method2 as md2
from Internship_RSMAS.Read_SentinelData import SentinelClass as rd
import Internship_RSMAS.DataAnalysis.T1 as tt1

grid_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Grid/'
layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'
coast_root='D:/Academic/MPS/Internship/Data/coastline/'
cds_root = 'D:/Academic/MPS/Internship/Data/CDS/'
level1_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/'
level2_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level2/'

class imp(object):
    def __init__(self, data):
        self.__size = data.shape
        self.factor = 1
        self.__ma = np.ma.array(data, mask = np.isin(data, [0]))
        self.max = self.__ma.max()
        self.min = self.__ma.min()

    @property
    def size(self):
        return self.__size

    @property
    def ma(self):
        return self.__ma


    def dB(self, data):
        img = data
        img[img<=0] = np.nan
        img = 10*np.log10(img)
        img[np.isnan(img)] = -200
        return img


    def rebin(self, n, data = 0):
        if isinstance(data, int):
            img = self.__ma
        else:
            img = data
        rows, cols = img.shape
        nr, nc = math.floor(rows / n), math.floor(cols / n)
        dat = img[0:n * nr, 0:n * nc]
        kernel = np.ones([n, n], dtype=np.int16)
        temp = signal.convolve(dat, kernel, mode = 'valid')
        img = temp[::n, ::n]
        return img


    def Display(self, data, max='t', min='t', mode = 'linear'):
        if mode == 'dB':
            data = self.dB(data)
        if isinstance(max, str):
            max_d = data.max()
        else:
            max_d = max
        if isinstance(min, str):
            min_d = data.min()
        else:
            min_d = min
        img = (data-min_d)*255/(max_d-min_d)
        img[img>255] = 255
        img[img<0] = 0
        rows, cols = data.shape
        dpi = 100
        fig = plt.figure(figsize=(rows / dpi, cols / dpi), dpi=dpi)
        fig.figimage(img, cmap = 'gray')
        return None

    def save_fig(self, prod, root = layer_root):
        min_d = -30
        max_d = 0
        for ii in range(len(prod.data)):
            a1 = self.dB(prod.data[ii])
            img = (a1 - min_d) * 255 / (max_d - min_d)
            img[img > 255] = 255
            img[img < 0] = 0
            rows, cols = a1.shape
            dpi = 100
            fig = plt.figure(figsize=(cols / dpi, rows / dpi), dpi=dpi)
            fig.figimage(img, cmap='gray')
            plt.savefig(root+prod.t[ii][:-1]+'.png')
            plt.close(fig)
        print('Done!')
        return None

    def save_fig2(self, prod, root = layer_root):
        for ii in range(len(prod.count)):
            a1 = prod.count[ii]
            rows, cols = a1.shape
            dpi = 100
            fig = plt.figure(figsize=(cols / dpi+2, rows / dpi), dpi=dpi)
            fig.figimage(a1, cmap='gray')
            plt.savefig(root+prod.t[ii][:-1]+'.png')
            plt.close(fig)
        print('Done!')
        return None








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
    plt.imshow(img, cmap='gray')
    return None


def Resize_SigmaNaught(data, n):
    rows, cols = data.shape
    nr, nc = math.floor(rows / n), math.floor(cols / n)
    dat = data[0:n*nr, 0:n*nc]
    # new_pic = [[np.nanmean(dat[r * n:(r + 1) * n, c * n:(c + 1) * n]) for c in range(nc)] for r in range(nr)]
    # new_pic = np.array(new_pic).astype(np.float32)
    dat[np.isnan(dat)] = 0
    new_pic=[[dat[r * n:(r + 1) * n, c * n:(c + 1) * n].mean() for c in range(nc)] for r in range(nr)]
    new_pic = np.array(new_pic).astype(np.float32)
    return new_pic


def Resize_LL(data, n):
    rows, cols = data.shape
    nr, nc = math.floor(rows / n), math.floor(cols / n)
    dat = data[0:n*nr, 0:n*nc]
    upleft = dat[::n, ::n]
    upright = dat[::n, n - 1::n]
    lowleft = dat[n - 1::n, ::n]
    lowright = dat[n - 1::n, n-1::n]
    # new_LL = [[(dat[r * n, c * n] + dat[r * n, (c + 1) * n - 1] + dat[(r + 1) * n - 1, c * n] + dat[
    #     (r + 1) * n - 1, (c + 1) * n - 1]) / 4 for c in range(nc)] for r in range(nr)]
    # new_LL = np.array(new_LL, dtype=np.float_)
    new_LL = (upleft+upright+lowleft+lowright)/4
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




# @jit(nopython=True, parallel=True)
def CalMaps(data, count):
    # imgs = np.ma.array(data, mask = np.where(count<=0, True, False))
    # MeanMap = imgs.mean(axis=0, dtype=np.float32)
    # MedianMap = np.median(imgs, axis=0)
    # StdMap = imgs.std(axis=0, dtype=np.float32)
    # PeakMap = imgs.max(axis = 0) - imgs.min(axis = 0)
    num, rows, cols = data.shape
    MeanMap = np.zeros([rows, cols], dtype = np.float32)
    StdMap = np.zeros_like(MeanMap)
    PeakMap = np.zeros_like(MeanMap)
    for r in range(rows):
        for c in range(cols):
            dat= data[:, r, c]
            dat = dat[np.where(count[:, r, c]>0)]
            if len(dat) > 0:
                MeanMap[r, c] = dat.mean()
                StdMap[r, c] = dat.std()
                PeakMap[r, c] = dat.max()-dat.min()
    return MeanMap, StdMap, PeakMap


def CalM(data, tt, method):
    num, rows, cols = data.shape
    mmm = np.zeros([rows, cols], dtype=np.float32)
    for r in range(rows):
        for c in range(tt[r]):
            dat = data[:, r, c]
            d1 = dat[np.logical_not(np.isnan(dat))]
            if len(d1)>0:
                mmm[r, c] = method(d1)
    return mmm


def KbMap(xmin, xmax):
    N_list=list(range(38, 72))
    N_list.remove(39)
    N_list.remove(58)
    N_list.remove(69)
    temp = rd.Grid_data()
    temp.LoadData(ty = 'resize')
    wspeed_cds = np.sqrt(temp.u_cds*temp.u_cds+temp.v_cds*temp.v_cds)
    wspeed_s2 = np.sqrt(temp.u_s2*temp.u_s2+temp.v_s2*temp.v_s2)
    tt = np.zeros(15, dtype=np.int8)
    ss = np.zeros_like(tt)
    ss[:] = 18
    for r in range(15):
        for c in range(19):
            if wspeed_s2[42, r, c] > 0:
                tt[r] = max(tt[r], c)
                ss[r] = min(ss[r], c)
    nums, rows, cols = wspeed_s2.shape
    kmap = np.zeros([rows, cols], dtype = np.float_)
    bmap = np.zeros_like(kmap)
    kmap[:, :], bmap[:, :] = np.nan, np.nan
    for r in range(rows):
        for c in range(ss[r], tt[r]+1):
            dat1, dat2= [], []
            for n in N_list:
                if np.logical_not(np.isnan(wspeed_s2[n, r, c])) and np.logical_not(np.isnan(wspeed_cds[n, r, c])) and (xmin<wspeed_cds[n, r, c]<xmax):
                    dat1.append(wspeed_cds[n, r, c])
                    dat2.append(wspeed_s2[n, r, c])
            print(r, c, len(dat1))
            if len(dat1)!=0:
                x, y, k, b = tt1.LR(dat1, dat2)
                kmap[r, c] = k
                bmap[r, c] = b
    return kmap, bmap




def InterpolationWinds(r, c, u, v):
    rc = [[r[i], c[i]] for i in range(len(c))]
    rc = np.array(rc)
    grd = md2.uniform_grid()
    rows, cols = grd.size
    xy = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    xy = np.array(xy)
    u10 = griddata(rc, u, xy)
    v10 = griddata(rc, v, xy)
    u10 = u10.reshape([rows, cols])
    v10 = v10.reshape([rows, cols])
    return u10, v10


def InterpolationCDSwind(du, dv):
    rc, u, v = [], [], []
    rc_app, u_app, v_app = rc.append, u.append, v.append
    for key in du.keys():
        rc_app(key)
        u_app(np.mean(du[key]))
        v_app(np.mean(dv[key]))
    rc, u, v = np.array(rc), np.array(u), np.array(v)
    rows, cols = 15, 19
    xy = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    xy = np.array(xy)
    u10 = griddata(rc, u, xy)
    v10 = griddata(rc, v, xy)
    u10 = u10.reshape([rows, cols])
    v10 = v10.reshape([rows, cols])
    return u10, v10


def Extract_NRCS(R, C, data = 0):
    if isinstance(data, int):
        temp = rd.Grid_data()
        temp.LoadData(ty='resize')
        data = temp.NRCS
    N_list=list(range(38, 72))
    N_list.remove(39)
    N_list.remove(58)
    N_list.remove(69)
    s_list = np.zeros(len(N_list))
    r_range = [R*150, min((R+1)*150-1, 2101)]
    c_range = [C*150, min((C+1)*150-1, 2820)]
    now = 0
    for n in N_list:
        tem = data[n, r_range[0]: r_range[1]+1, c_range[0]:c_range[1]+1]
        tem = np.where(tem>0.8, np.nan, tem)
        s_list[now] = np.nanmean(tem)
        now += 1
    return s_list
