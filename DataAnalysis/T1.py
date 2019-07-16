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





if __name__ == '__main__':
    # temp = rd.Grid_data()
    # temp.LoadData(ty = 'resize')
    # wspeed_cds = np.sqrt(temp.u_cds*temp.u_cds+temp.v_cds*temp.v_cds)
    # wspeed_s2 = np.sqrt(temp.u_s2*temp.u_s2+temp.v_s2*temp.v_s2)
    #
    # tt = np.zeros(15, dtype=np.int8)
    # ss = np.zeros_like(tt)
    # ss[:] = 18
    # for r in range(15):
    #     for c in range(19):
    #         if wspeed_s2[42, r, c]>0:
    #             tt[r] = max(tt[r], c)
    #             ss[r] = min(ss[r], c)
    # print(ss)
    # print(tt)
    #
    # dx, dy = Bydistance(wspeed_cds[36:, :, :], wspeed_s2[36:, :, :], ss, tt)
    # mm = max(dx.keys())
    # klist, blist = [], []
    # for m in range(mm+1):
    #     if m in dx:
    #         x, y, k, b = LR(dx[m], dy[m])
    #         klist.append(k)
    #         blist.append(b)
    # #         plt.figure(m)
    # #         plt.plot(range(20), range(20))
    # #         plt.plot(dx[m], dy[m], '.')
    # #         plt.plot(x, y, '--')
    # #         plt.xlim(0, 20)
    # #         plt.ylim(0, 20)
    # #         ax = plt.gca()
    # #         ax.text(7, 3, '(k, b):(%.5f, %.5f)'%(k, b))
    # #         ax.set_aspect(1)
    # #         ax.set_xlabel('CDS')
    # #         ax.set_ylabel('S1_Level2')
    # # plt.show()

    # plt.plot(klist, label = 'k')
    # plt.plot(blist, label = 'b')
    # plt.legend()
    # plt.show()

    for x in [0, 5, 10, 15, 1]:
        y = x+5
        ttt = (x+y+1)/2
        if x == 1:
            x = 0
            y = 100
            ttt = 5
        kmap, bmap = Lc.KbMap(x, y)
        a = np.ones_like(kmap)
        b = kmap*(a*ttt)+bmap - ttt
        ccolor = np.zeros([21, 4], dtype = np.float_)
        ccolor[10,:] = [1,1,1,1]
        ccolor[:, 3] = 1
        ccolor[:11, 0] = np.linspace(0, 1, 11)
        ccolor[:11, 1] = np.linspace(0, 1, 11)
        ccolor[:10, 2] = 1
        ccolor[10:, 2] = np.linspace(1, 0, 11)
        ccolor[10:, 1] = np.linspace(1, 0, 11)
        ccolor[10:, 0] = 1
        cmap = ListedColormap(ccolor)
        cmap.set_over('red')
        cmap.set_under('blue')
        bounds = list(np.arange(21)*0.3 - 3)
        norm = BoundaryNorm(bounds, cmap.N)
        # plt.figure(0)
        # plt.imshow(kmap, cmap = 'bwr')
        # plt.colorbar()
        # plt.title('k value')
        # plt.figure(1)
        # plt.imshow(bmap, cmap = 'bwr')
        # plt.colorbar()
        # plt.title('b value')
        # plt.figure(2)
        # plt.imshow(b, cmap = 'bwr')
        # plt.colorbar()
        # plt.title('estimated wind speed while the true wind speed is 3 m/s')
        data = np.load('D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/Layer40-20180518.npy')
        a = Lc.imp(data)
        a1 = a.dB(data)
        min_d = -30
        max_d = 0
        img = (a1 - min_d) * 255 / (max_d - min_d)
        img[img > 255] = 255
        img[img < 0] = 0
        rows, cols = a1.shape
        dpi = 100
        fig = plt.figure(figsize=(cols / dpi, rows / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')

        bb = np.ones_like(img)
        rows, cols = img.shape
        for r in range(rows):
            for c in range(cols):
                bb[r, c] = b[r//150, c//150]
        ddd = ax.imshow(bb, cmap=cmap, norm = norm, alpha = 0.7)
        fig.colorbar(ddd)
        plt.title('Estimated wind speed from wind speed range (%d, %d) while the true wind speed is %f m/s'%(x,y, ttt))
    plt.show()