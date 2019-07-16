import Internship_RSMAS.Griding.Coordinates as cod
import Internship_RSMAS.Griding.GridMethod as gm
import Internship_RSMAS.Griding.Method2 as md2
import Internship_RSMAS.Griding.IOcontrol as jo
import Internship_RSMAS.Griding.LayerCalculator as Lc
import Internship_RSMAS.Read_SentinelData.SentinelClass as rd
import time
from datetime import datetime, timedelta
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


# use first layer (base) as standard grid and attain its lon and lat

# use bilinear interpolation to calculate the lon lat for new image
# then we get points (lon,lat) and their corresponding SigmaNaught
# apply delaunay triangulation-------------many small triangles
# figure the
cds_root = 'D:/Academic/MPS/Internship/Data/CDS/'
coast_root='D:/Academic/MPS/Internship/Data/coastline/'
level1_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/'
level2_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level2/'
layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'
temp_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Temp/'
file_root = 'F:/Jiewei/Sentinel-1/Level1-GRD-IW/WhiteCity/'
ocn_root = 'F:/Jiewei/Sentinel-1/Level2-OCN/'
tempN = 4

def MakeAGrid(name, dir):
    dat = rd.Data_Level1(name, dir)
    dat.OneStep()
    GCP = dat.GCPs
    SigmaNaught = dat.denoiseNRCS
    y_lim, x_lim = SigmaNaught.shape
    del dat, SigmaNaught
    n = 10
    gm.AllLonLat('Base', x_lim, y_lim, GCP, n)
    return None


def OtherLayer(d, n):
    for indice in range(n, len(d.series)):
        temp = rd.Data_Level1(d.series[indice], d.FList[indice])
        temp.OneStep()
        GCP = temp.GCPs
        SigmaNaught = temp.denoiseNRCS
        name = 'Layer' + str(indice) + '-' + d.series[indice]
        del temp
        y1, x1 = SigmaNaught.shape
        gm.AllLonLat(name, x1, y1, GCP, 5)
        start = time.time()
        SigmaNaught = Lc.Resize_SigmaNaught(SigmaNaught, 5)
        print('Time for resize SigmaNaught:', (time.time() - start) / 60, 'mins')
        Sigma_N, nums_arr = gm.GridSigmaNaught(SigmaNaught)
        print('Time for resampling 1 SAR image:', (time.time()-start)/60, 'mins')
        jo.Matrix_save(Sigma_N, name, layer_root)
    return None


def Resample(n, mode = 'uniform'):
    d = rd.SentinelData()
    d.Get_List(file_root)
    m = 1   # resize factor
    for indice in range(n, len(d.series)):
        stt = time.time()
        temp = rd.Data_Level1(d.series[indice], d.FList[indice])
        temp.OneStep()
        GCP = temp.GCPs
        SigmaNaught = temp.denoiseNRCS
        name = 'Layer' + str(indice) + '-' + d.series[indice]
        del temp
        y1, x1 = SigmaNaught.shape
        sub_img, up_left = cod.N_sub(tempN, SigmaNaught)
        del SigmaNaught, up_left
        jo.NMatrix_save(sub_img, 'SigmaNaught', temp_root)
        del sub_img
        md2.AllLonLat(name, x1, y1, GCP, m)
        img, pixel_num = md2.GridLL_SigmaNaught(mode)
        jo.Matrix_save(img, name, layer_root)
        jo.Matrix_save(pixel_num, name+'-distribution', layer_root)
        print('1 image:', (time.time()-stt)/60, 'mins')
        print('Done!\n')
    return None


def ResampleWind():
    tt = np.load(coast_root + 'Left coast for each row.npy')
    d = rd.SentinelData()
    d.Get_List_NetCDF(ocn_root)
    for indice in range(len(d.series)):
        temp = rd.Data_Level2(d.series[indice], d.FList[indice])
        temp.Get_WindData()
        temp.CalWindVector()
        mask= temp.windX.mask.reshape(-1)
        Ind = np.where(mask == 0)[0]
        lon = temp.lon.data.reshape(-1)
        lat = temp.lat.data.reshape(-1)
        U = temp.windX.data.reshape(-1)
        V = temp.windY.data.reshape(-1)
        lon, lat, U, V = lon[Ind], lat[Ind], U[Ind], V[Ind]
        r, c, u, v = md2.GridLL_Wind(lon, lat, U, V)
        ind = [xx for xx in range(len(r)) if tt[r[xx]]>c[xx]]
        r_new = [r[xx] for xx in ind]
        c_new = [c[xx] for xx in ind]
        u_new = [u[xx] for xx in ind]
        v_new = [v[xx] for xx in ind]
        r, c, u, v = r_new, c_new, u_new, v_new
        del r_new, c_new, u_new, v_new, ind
        # md2.save_fig(d.series[indice], r, c, u, v)
        if len(r)>=4:
            u10, v10 = Lc.InterpolationWinds(r, c, u, v)
            np.save(level2_root + 'Interp/' + d.series[indice] + '-u10.npy', u10)
            np.save(level2_root + 'Interp/' + d.series[indice] + '-v10.npy', v10)
        u10 = np.zeros([15, 19], dtype=np.float16)
        v10 = np.zeros_like(u10)
        du = {}
        dv = {}
        for ii in range(len(r)):
            r1, c1 = r[ii] // 150, c[ii] // 150
            if (r1, c1) in du:
                du[r1, c1].append(u[ii])
                dv[r1, c1].append(v[ii])
            else:
                du[r1, c1] = [u[ii]]
                dv[r1, c1] = [v[ii]]
        for rr, cc in du.keys():
            u10[rr, cc] = np.mean(du[rr, cc])
            v10[rr, cc] = np.mean(dv[rr, cc])
        np.save(level2_root + 'NotInterp/' + d.series[indice] + '-u10.npy', u10)
        np.save(level2_root + 'NotInterp/' + d.series[indice] + '-v10.npy', v10)
    return None


def MatchCDS():
    tt = np.load(coast_root + 'Left coast for each row.npy')
    data = rd.layers()
    data.LoadData()
    for indice in range(len(data.t)):
        lon, lat, u, v = md2.ReadCDSData(data.t[indice][:-1])
        rows, cols = u.shape
        lon_N = [lon[c] for r in range(rows) for c in range(cols)]
        lat_N = [lat[r] for r in range(rows) for c in range(cols)]
        lon_N = np.array(lon_N)
        lat_N = np.array(lat_N)
        u, v = u.reshape(-1), v.reshape(-1)
        r, c, u, v = md2.GridLL_Wind(lon_N, lat_N, u, v)
        # ind = [xx for xx in range(len(r)) if tt[r[xx]] > c[xx]]
        # r_new = [r[xx] for xx in ind]
        # c_new = [c[xx] for xx in ind]
        # u_new = [u[xx] for xx in ind]
        # v_new = [v[xx] for xx in ind]
        # r, c, u, v = r_new, c_new, u_new, v_new
        # del r_new, c_new, u_new, v_new, ind
        # md2.save_fig(data.t[indice][:-1], r, c, u, v, cds_root)
        # if len(r)>=4:
        #     u10, v10 = Lc.InterpolationWinds(r, c, u, v)
        #     np.save(cds_root + 'Interp/' + data.t[indice][:-1] + '-u10.npy', u10)
        #     np.save(cds_root + 'Interp/' + data.t[indice][:-1] + '-v10.npy', v10)
        # u10 = np.zeros([15,19], dtype = np.float16)
        # v10 = np.zeros_like(u10)
        du = {}
        dv = {}
        for ii in range(len(r)):
            r1, c1 = r[ii]//150, c[ii]//150
            if (r1, c1) in du:
                du[r1, c1].append(u[ii])
                dv[r1, c1].append(v[ii])
            else:
                du[r1, c1] = [u[ii]]
                dv[r1, c1] = [v[ii]]
        # for rr, cc in du.keys():
        #     u10[rr, cc] = np.mean(du[rr, cc])
        #     v10[rr, cc] = np.mean(dv[rr, cc])
        u10, v10 = Lc.InterpolationCDSwind(du, dv)
        np.save(cds_root + 'NotInterp/' + data.t[indice][:-1] + '-u10.npy', u10)
        np.save(cds_root + 'NotInterp/' + data.t[indice][:-1] + '-v10.npy', v10)
    return None



def Merge(root):
    s1 = datetime(2017, 1, 11)
    s2 = datetime(2019, 5, 25)
    datelist = [s1]
    nows = s1
    while nows != s2:
        nows = nows + timedelta(days=12)
        datelist.append(nows)
    data = rd.winddata()
    data.GetData(datelist, root, ty = 'resize')
    np.save(root + 'u10.npy', data.u)
    np.save(root + 'v10.npy', data.v)
    # data.GetData(datelist, root)
    # np.save(root + 'u10-interp.npy', data.u)
    # np.save(root + 'v10-interp.npy', data.v)
    return None


def DrawGrid():
    s1 = datetime(2017, 1, 11)
    s2 = datetime(2019, 5, 25)
    datelist = [s1]
    nows = s1
    while nows != s2:
        nows = nows + timedelta(days=12)
        datelist.append(nows)
    for ss in datelist:
        md2.pp_fig(datetime.strftime(ss, '%Y%m%d'), level1_root)
    print('Done!')
    return None






if __name__ == '__main__':
    # # 1. grid based on layers 20170216
    # # Grid
    # d = rd.SentinelData()
    # d.Get_List(file_root)
    # # start from 20170216
    # # MakeAGrid(d.series[2], d.FList[2])
    # OtherLayer(d, 0)


    # 2. uniform grid with constant Lon Lat interval
    # Resample(0, mode = 'uniform')
    #
    # # make images
    # ll = rd.layers()
    # ll.GetData(layer_root)
    # ll.LoadData()
    # a = Lc.imp(ll.data[1])
    # a.save_fig(ll)

    # ResampleWind()
    # MatchCDS()

    # Merge(level2_root+'Interp/')
    # Merge(cds_root+'Interp/')

    # Merge(level2_root + 'NotInterp/')
    # Merge(cds_root + 'NotInterp/')
    # DrawGrid()




    t = 1
    # for r in range(12):
    #     plt.figure(r)
    #     for n in range(31):
    #         dat = sigma[n, r, :tt[r] + 1]
    #         dat = dat[::-1]
    #         plt.plot(dat[t:] - dat[t], 'g--')
    #     plt.plot(range(8), np.zeros(8), 'r-')
    # plt.show()
    # wind from ECMWF
    for r in range(12):
        plt.figure(r)
        for n in range(31):
            dat = wcds[n, r, :tt[r] + 1]
            dat = dat[::-1]
            ax = plt.plot(dat[t:] - dat[t], '--', color = cmap(np.mean(dcds[n,r,:tt[r]])/360))
        plt.colormaps()
        plt.plot(range(8), np.zeros(8), 'r-')
    plt.show()
    # wind from Sentinel 1 image
    # for r in range(12):
    #     plt.figure(r)
    #     for n in range(31):
    #         dat = ws2[n, r, :tt[r] + 1]
    #         dat = dat[::-1]
    #         plt.plot(dat[t:] - dat[t], 'g--')
    #     plt.plot(range(8), np.zeros(8), 'r-')
    # plt.show()