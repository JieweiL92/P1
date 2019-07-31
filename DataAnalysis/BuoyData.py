import Internship_RSMAS.Griding.Method2 as md2
import Internship_RSMAS.DataAnalysis.RowPlots as rp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os

buoy_root = 'F:/Jiewei/Buoys/'


def ReadBuoyWindData(path):
    fid = open(path)
    title = fid.readline()
    line = fid.readline()
    dat = fid.readlines()
    data = [list(map(float, ch.split()[:8])) for ch in dat]
    dat = np.array(data).astype(np.float16)
    wind_direction = dat[:, 5]
    wind_speed = dat[:, 6]
    times = [datetime(int(t[0]), int(t[1]),int(t[2]), int(t[3]), int(t[4])) for t in dat]
    wind_speed = np.where(wind_speed>80, np.nan, wind_speed)
    wind_direction = np.where(wind_direction>360, np.nan, wind_direction)
    fid.close()
    return times, wind_direction, wind_speed


def fliter(T, direc, speed, dates):
    T_n, dir_n, spd_n = [], [], []
    T_dates = np.array([d.date() for d in T])
    T = np.array(T)
    T_app, dir_app, spd_app = T_n.append, dir_n.append, spd_n.append
    standard = timedelta(hours = (14+25/60))/timedelta(seconds =1)
    for ds in dates:
        ind = np.where(T_dates == ds)
        T_temp, dir_temp, spd_temp = T[ind], direc[ind], speed[ind]
        if len(T_temp)==1:
            T_app(datetime(ds.year, ds.month, ds.day))
            dir_app(dir_temp[0])
            spd_app(spd_temp[0])
        elif len(T_temp)>1:
            T_app(datetime(ds.year, ds.month, ds.day))
            x = (T_temp -  datetime(ds.year, ds.month, ds.day))/timedelta(seconds =1)
            if (x==standard).any():
                ii = np.where(x==0)
                dir_app(dir_temp[ii])
                spd_app(spd_temp[ii])
            else:
                if len(x[x>standard])==0:
                    dir_app(dir_temp[-1])
                    spd_app(spd_temp[-1])
                elif len(x[x<standard])==0:
                    dir_app(dir_temp[0])
                    spd_app(spd_temp[0])
                else:
                    u, v = spd_temp * np.cos(np.pi / 180 * (270-dir_temp)), spd_temp * np.sin(np.pi / 180 * (270-dir_temp))
                    ss, tt = 0, len(x)
                    for xx in range(len(x)):
                        if x[xx]<standard:
                            ss = max(ss, xx)
                        else:
                            tt = min(tt, xx)
                    uu = u[ss] * (x[tt] - standard) / (x[tt] - x[ss]) + u[tt] * (standard - x[ss]) / (x[tt] - x[ss])
                    vv = v[ss] * (x[tt] - standard) / (x[tt] - x[ss]) + v[tt] * (standard - x[ss]) / (x[tt] - x[ss])
                    spd_app(np.sqrt(uu*uu+vv*vv))
                    dir_app(np.arctan2(-uu, -vv)*180/np.pi)
    T_n, dir_n, spd_n = np.array(T_n), np.array(dir_n), np.array(spd_n)
    dir_n = np.where(dir_n<0, dir_n+360, dir_n)
    return T_n, dir_n, spd_n



def MergeBuoy(root):
    s1 = datetime(2017,1,11)
    s2 = datetime(2019,5,25)
    datelist = [s1]
    nows = s1
    while nows != s2:
        nows = nows + timedelta(days=12)
        datelist.append(nows)
    dates = [d.date() for d in datelist]
    wdata = np.ones([len(datelist), 2], dtype = np.float32)
    wdata = np.nan*wdata
    lon, lat = 0, 0
    file_set = os.listdir(root)
    for file in file_set:
        if file.find('.txt')>0:
            if file == 'position.txt':
                fid = open(root+file)
                lat = fid.readline()
                lon = fid.readline()
                lon, lat = float(lon), float(lat)
            else:
                print(file)
                times, wdir, wsp = ReadBuoyWindData(root + file)
                times, wdir, wsp = fliter(times, wdir, wsp, dates)
                ind = ((times - s1)/timedelta(days=12)).astype(np.int_)
                wdata[ind, 0] = wdir
                wdata[ind, 1] = wsp
    return wdata, lon, lat





if __name__ == '__main__':
    # filelist = os.listdir(buoy_root)
    # wdat, LL = [], []
    # for iii in range(len(filelist)):
    #     print('\n',filelist[iii])
    #     wdata, lon, lat = MergeBuoy(buoy_root+filelist[iii]+'/')
    #     wdat.append(wdata)
    #     LL.append([lon, lat])
    # print(len(wdat))
    # LL = np.array(LL)
    # print(LL)
    # wdat = np.array(wdat)
    # print(wdat.shape)
    # np.save('D:/Academic/MPS/Internship/Data/Buoys/'+'wind(direction+speed).npy', wdat)
    # np.save('D:/Academic/MPS/Internship/Data/Buoys/'+'Position.npy', LL)
    # print('Done!')

    wdat = np.load('D:/Academic/MPS/Internship/Data/Buoys/'+'wind(direction+speed).npy')
    wdat = wdat[:, 36:67, :]
    LL = np.load('D:/Academic/MPS/Internship/Data/Buoys/'+'Position.npy')
    grd = md2.uniform_grid()
    r, c, i = grd.FindPosition(LL[:, 0], LL[:, 1])
    r, c = r // 150, c // 150
    print(r, c)
    sigma, ws2, wcds, ds2, dcds, ss, tt = rp.eliminate()

    for i in range(4):
        plt.figure(i)
        plt.plot(wdat[i, :, 1], label = 'Buoys')
        plt.plot(ws2[:, r[i], c[i]], label = 'Sentinel Product')
        plt.plot(wcds[:, r[i], c[i]], label = 'ECMWF')
        plt.legend()
        plt.title('Compare wind speed data')
    plt.show()
    #
    # for i in range(4):
    #     plt.figure(i)
    #     plt.plot(wdat[i, :, 0], label = 'Buoys')
    #     plt.plot(ds2[:, r[i], c[i]], label = 'Sentinel Product')
    #     plt.plot(dcds[:, r[i], c[i]], label = 'ECMWF')
    #     plt.legend()
    #     plt.title('Compare wind direction data')
    # plt.show()



