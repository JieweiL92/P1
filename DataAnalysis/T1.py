import numpy as np
from datetime import datetime, timedelta
import netCDF4 as ncdf
import Internship_RSMAS.Read_SentinelData.SentinelClass as rd
import Internship_RSMAS.Griding.LayerCalculator as Lc
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from matplotlib.colors import ListedColormap, BoundaryNorm


def ReadUpwellingIndex():
    root = 'F:/Jiewei/Upwelling Index/'
    fname = 'upwell42N125W.txt'
    fid = open(root+fname)
    line = fid.readline()
    while line.find('--------------')<0:
        line = fid.readline()
    dat = fid.readlines()
    time1, time2 = datetime.strptime('19'+dat[0][2:8],'%Y%m%d'),  datetime.strptime('20'+dat[-1][2:8],'%Y%m%d')
    td = timedelta(hours=6)
    l1 = int((time2 - time1)/td)
    time_list = [time1 + i*td for i in range(l1+4)]
    o_list = [[int(s[10:15]), int(s[15:20]), int(s[20:25]), int(s[25:30])] for s in dat]
    a_list = [[int(s[30:35]), int(s[35:40]), int(s[40:45]), int(s[45:50])] for s in dat]
    off_shore = [y if y!=-9999 else np.nan for x in o_list for y in x]
    along_shore = [y if y!=-9999 else np.nan for x in a_list for y in x]
    return time_list, off_shore, along_shore

def ReadBuoyWindData(path):
    fid = open(path)
    title = fid.readline()
    line = fid.readline()
    dat = fid.readlines()
    data = [list(map(float, ch.split())) for ch in dat]
    dat = np.array(data).astype(np.float16)
    wind_direction = dat[:, 5]
    wind_speed = dat[:, 6]
    times = [datetime(int(t[0]), int(t[1]),int(t[2]), int(t[3]), int(t[4])) for t in dat]
    wind_speed = np.where(wind_speed>80, np.nan, wind_speed)
    wind_direction = np.where(wind_direction>360, np.nan, wind_direction)
    fid.close()
    return times, wind_direction, wind_speed


def ReadCDSData(name, path = 'F:/Jiewei/CDS/'):
    dataset = ncdf.Dataset(path+name+'.nc')
    lon_arr = dataset.variables['longitude'][:].data             # 0, 0.25, 0.5, 0.75, ..., 359.5, 359.75   936:951+1
    lat_arr = dataset.variables['latitude'][:].data              # 90, 89.75, ... -90                       186:195+1
    lon_arr = lon_arr[936:952]-360
    lat_arr = lat_arr[186:196]
    u10 = dataset.variables['u10'][:].data[:,186:196,936:952]
    v10 = dataset.variables['v10'][:].data[:,186:196,936:952]
    del dataset
    d14_u = u10[0,:,:]
    d15_u = u10[1,:,:]
    d14_v = v10[0,:,:]
    d15_v = v10[1,:,:]
    del u10, v10
    u = (d14_u*7+d15_u*5)/12
    v = (d14_v*7+d15_v*5)/12
    return lon_arr, lat_arr, u, v


def MeanValue():
    ll = rd.layers()
    ll.LoadData()
    ll.CalMean()
    return None


def test_interp(wspeed_cds, wspeed_s2, n, tt):
    nums, rows, cols = wspeed_cds.shape
    cds_list, s2_list = [], []
    cds_app = cds_list.append
    s2_app = s2_list.append
    for r in range(rows):
        for c in range(tt[r]):
            if np.logical_not(np.isnan(wspeed_cds[n , r, c])) and np.logical_not(np.isnan(wspeed_s2[n, r, c])):
                cds_app(wspeed_cds[n, r, c])
                s2_app(wspeed_s2[n, r, c])
    wd = np.array([[cds_list[i], s2_list[i]] for i in range(len(cds_list)) if cds_list[i] != 0 and s2_list[i] != 0])
    return wd

def test_resize(wspeed_cds, wspeed_s2, n):
    nums, rows, cols = wspeed_cds.shape
    cds_list, s2_list = [], []
    cds_app = cds_list.append
    s2_app = s2_list.append
    for r in range(rows):
        for c in range(cols):
            if wspeed_cds[n , r, c]!=0 and wspeed_s2[n, r, c]!=0:
                cds_app(wspeed_cds[n, r, c])
                s2_app(wspeed_s2[n, r, c])
    wd = np.array([[cds_list[i], s2_list[i]] for i in range(len(cds_list)) if cds_list[i] != 0 and s2_list[i] != 0])
    return wd




def LR(xx, yy):
    reg = lm.LinearRegression()
    reg.fit(np.array(xx).reshape(-1,1), np.array(yy).reshape(-1,1))
    k, b = reg.coef_[0][0], reg.intercept_[0]
    x = np.arange(20)
    y = k * x + b
    return x, y, k, b


def LR2(xx, yy):
    # b = np.zeros(len(xx))
    # a = np.zeros([len(b), 2])
    # a[:, 1] = 0
    # a[:, 0] = xx
    # b[:] = yy
    # a_tem = np.dot(a.T, a)
    # a_tem = np.linalg.inv(a_tem)
    # a_tem = np.dot(a_tem, a.T)
    # result = np.dot(a_tem, b)
    def r_sq(k):
        r2 = (k*xx-yy)**2/(k*k+1)
        return np.sum(r2)

    k_min, k_max = (yy/xx).min(), (yy/xx).max()
    k_best = 0
    r_min = min(r_sq(k_min), r_sq(k_max))
    steps = 0
    while steps<20:
        k1 = k_min*3/4 + k_max/4
        k2 = k_min/2 + k_max/2
        k3 = k_min/4 + k_max*3/4
        if r_sq(k1)<r_sq(k3):
            k_max = k2
        elif r_sq(k1)>r_sq(k3):
            k_min = k1
        else:
            k_max = k3
            k_min = k2
        r_tem = r_sq(k2)
        if r_tem<r_min:
            k_best = k2
            r_min = r_tem
        steps += 1
    return k_best




def LRb(xx, yy):
    # b = np.zeros(len(xx))
    # a = np.zeros([len(b), 2])
    # a[:, 1] = 0
    # a[:, 0] = xx
    # b[:] = yy
    # a_tem = np.dot(a.T, a)
    # a_tem = np.linalg.inv(a_tem)
    # a_tem = np.dot(a_tem, a.T)
    # result = np.dot(a_tem, b)
    def r_sq(b):
        r2 = (xx-yy+b)**2/5
        return np.sum(r2)

    b_min, b_max = (yy-xx).min(), (yy-xx).max()
    b_best = 0
    r_min = min(r_sq(b_min), r_sq(b_max))
    steps = 0
    while steps<20:
        b1 = b_min*3/4 + b_max/4
        b2 = b_min/2 + b_max/2
        b3 = b_min/4 + b_max*3/4
        if r_sq(b1)<r_sq(b3):
            b_max = b2
        elif r_sq(b1)>r_sq(b3):
            b_min = b1
        else:
            b_max = b3
            b_min = b2
        r_tem = r_sq(b2)
        if r_tem<r_min:
            b_best = b2
            r_min = r_tem
        steps += 1
    return b_best

#
#
# def distanceLL(coastline):
#     def distLL(lon1, lat1, lon2, lat2):
#
#
#         pass
#
#
#     temp = rd.Grid_data()
#     lon, lat = temp.position_center()
#     dd = np.empty_like(lon)
#     rows, cols = dd.shape
#     for r in range(rows):
#         for c in range(cols):
#             for lonc, latc in coastline:
#                 dd[r, c] = min(dd[r, c], distLL(lon[r, c], lat[r, c], lonc, latc))
#     return dd






def Bydistance(dat1, dat2, ss, tt):
    nums, rows, cols = dat1.shape
    dx, dy = {}, {}
    for r in range(rows):
    # for r in [set_r]:
        if ss[r]<=tt[r]:
            for c in range(ss[r], tt[r] + 1):
                if dat2[3, r, c]!=0:
                    distance = tt[r] - c
                    # distance = dist[r, c]
                    if not(distance in dx):
                        dx[distance], dy[distance] = [], []
                    for n in range(nums):
                        if np.logical_not(np.isnan(dat1[n, r, c]) or np.isnan(dat2[n, r, c])):
                            dx[distance].append(dat1[n, r, c])
                            dy[distance].append(dat2[n, r, c])
    return dx, dy


def pp(wd, name):    # cds s2
    xx, yy, k, b = LR(wd[:, 0], wd[:, 1])
    wd_r = wd[:, 1] - wd[:, 0]
    lower = len(np.where(wd_r<0)[0])/len(wd_r)
    plt.plot(range(20), range(20))
    plt.plot(wd[:,0], wd[:, 1], '.', label = name)
    plt.plot(xx, yy, '--', label = 'Leastsq fitting')
    ax = plt.gca()
    plt.title(name)
    plt.legend()
    ax.text(7, 3, 'smaller than CDS:%.2f %%'%(lower*100))
    ax.set_aspect(1)
    ax.set_xlabel('CDS')
    ax.set_ylabel('S1_Level2')
    plt.xlim(0, 20)
    plt.ylim(0, 20)


def DrawMap(name, layer, alpha, vmin, vmax):
    data = np.load('D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/Layer40-20180518.npy')
    a = Lc.imp(data)
    a1 = a.dB(data)
    min_d, max_d = -30, 0
    img = (a1 - min_d) * 255 / (max_d - min_d)
    img[np.isnan(img)] = 255
    img[img > 255] = 255
    img[img < 0] = 0
    del a, a1
    rows, cols = img.shape
    dpi = 100
    fig = plt.figure(figsize=(cols / dpi, rows / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    cover = ax.imshow(layer.astype(np.float_), cmap = 'seismic', vmin =vmin, vmax = vmax, alpha = alpha)
    cbar = plt.colorbar(cover, fraction = 0.025, pad = 0.08)
    plt.gca().set_axis_off()
    plt.title(name, fontdict={'fontsize':36}, pad = 2.2)
    return None





if __name__ == '__main__':
    s1 = datetime(2017,1,11)
    s2 = datetime(2019,5,25)
    datelist = [s1]
    nows = s1
    while nows != s2:
        nows = nows + timedelta(days=12)
        datelist.append(nows)
    dates = [datetime.strftime(s, '%Y%m%d') for s in datelist]
    Nlist = list(range(36,67))
    temp = rd.Grid_data()
    temp.LoadData()
    wspeed_cds = np.sqrt(temp.u_cds*temp.u_cds+temp.v_cds*temp.v_cds)
    wspeed_s2 = np.sqrt(temp.u_s2*temp.u_s2+temp.v_s2*temp.v_s2)
    wd_cds = np.arctan2(-temp.u_cds, -temp.v_cds)*180/np.pi
    wd_s2 = np.arctan2(-temp.u_s2, -temp.v_s2)*180/np.pi
    tt = temp.mask[0]
    mask = temp.mask[1].astype(np.float_)
    del temp
    spd = wspeed_s2 - wspeed_cds
    wdd = wd_s2 - wd_cds
    del wspeed_cds, wspeed_s2, wd_cds, wd_s2
    mask[mask == 0] = np.nan
    nums, rows, cols = spd.shape
    for n in range(36,nums):
        spd[n, :, :] = spd[n, :, :] * mask
        wdd[n, :, :] = wdd[n, :, :] * mask

    start = 36
    stop = 67
    # stop = nums
    while start<=stop:
        DrawMap('Wind Direction Difference(degree) between Sentinel Product and ECMWF ERA5 Model in '+dates[start], wdd[start,:,:], 0.7, -20, 20)
        plt.savefig('D:/Academic/MPS/Internship/Data/winds/wind direction difference/'+dates[start]+'.png')
        plt.close()
        start += 1
    # plt.show()




    # for x in [0, 5, 10, 15, 1]:
    # # for x in [1]:
    #     y = x+5
    #     if x == 1:
    #         x = 0
    #         y = 100
    #     kmap= Lc.KMap(x, y)
    #     np.save('D:/Academic/MPS/Internship/Data/kmap-'+str(x)+'-'+str(y)+'.npy', kmap)
    #     ccolor = np.zeros([21, 4], dtype = np.float_)
    #     ccolor[10,:] = [1,1,1,1]
    #     ccolor[:, 3] = 1
    #     ccolor[:11, 0] = np.linspace(0, 1, 11)
    #     ccolor[:11, 1] = np.linspace(0, 1, 11)
    #     ccolor[:10, 2] = 1
    #     ccolor[10:, 2] = np.linspace(1, 0, 11)
    #     ccolor[10:, 1] = np.linspace(1, 0, 11)
    #     ccolor[10:, 0] = 1
    #     cmap = ListedColormap(ccolor)
    #     cmap.set_over('red')
    #     cmap.set_under('blue')
    #     bounds = list(np.arange(20)*0.1)
    #     norm = BoundaryNorm(bounds, cmap.N)
    #
    #     data = np.load('D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/Layer40-20180518.npy')
    #     a = Lc.imp(data)
    #     a1 = a.dB(data)
    #     min_d, max_d = -30, 0
    #     img = (a1 - min_d) * 255 / (max_d - min_d)
    #     img[np.isnan(img)] = 255
    #     img[img > 255] = 255
    #     img[img < 0] = 0
    #     del a, a1
    #     rows, cols = img.shape
    #     dpi = 100
    #     fig = plt.figure(figsize=(cols / dpi, rows / dpi), dpi=dpi)
    #     ax = fig.add_subplot(111)
    #     ax.imshow(img, cmap='gray')
    #
    #     # bb = np.ones_like(img)
    #     # rows, cols = img.shape
    #     # for r in range(rows):
    #     #     for c in range(cols):
    #     #         bb[r, c] = kmap[r//150, c//150]
    #     bb = kmap
    #     ddd = ax.imshow(bb, cmap=cmap, norm = norm, alpha = 0.7)
    #     fig.colorbar(ddd)
    #     plt.gca().set_axis_off()
    #     plt.title('Regression coefficient from wind speed range (%d, %d)'%(x,y))
    #     plt.show()
    # plt.show()