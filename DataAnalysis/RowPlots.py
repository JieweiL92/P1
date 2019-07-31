import Internship_RSMAS.Read_SentinelData.SentinelClass as rd
import Internship_RSMAS.Griding.LayerCalculator as Lc
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



# load wind speed, wind direction from ECMWF and Sentinel 1, and SigmaNaught
def eliminate(sss):
    # N_list=list(range(38, 72))
    # N_list.remove(39)
    # N_list.remove(58)
    # N_list.remove(69)
    N_list = list(range(36, 67))
    temp = rd.Grid_data()
    temp.LoadData(ty = sss)
    wspeed_cds = np.sqrt(temp.u_cds*temp.u_cds+temp.v_cds*temp.v_cds)
    wspeed_s2 = np.sqrt(temp.u_s2*temp.u_s2+temp.v_s2*temp.v_s2)
    wd_cds = np.arctan2(-temp.u_cds, -temp.v_cds)*180/np.pi
    wd_s2 = np.arctan2(-temp.u_s2, -temp.v_s2)*180/np.pi
    NRCS = temp.NRCS
    del temp
    tt = np.zeros(15, dtype=np.int8)
    ss = np.zeros_like(tt)
    ss[:] = 18
    for r in range(15):
        for c in range(19):
            if wspeed_s2[42, r, c] > 0:
                tt[r] = max(tt[r], c)
                ss[r] = min(ss[r], c)
    nums = len(N_list)
    ws2 = np.zeros([nums, 15, 19], dtype = np.float_)
    wcds = np.zeros_like(ws2)
    ds2 = np.zeros_like(ws2)
    dcds =np.zeros_like(ws2)
    sigma = np.zeros_like(ws2)
    for r in range(15):
        if ss[r]<=tt[r]:
            for c in range(ss[r], tt[r]+1):
                slist = Lc.Extract_NRCS(r, c, NRCS)
                sigma[:, r, c] = slist
                now = 0
                for n in N_list:
                    ws2[now, r, c] = wspeed_s2[n, r, c]
                    wcds[now, r, c] = wspeed_cds[n, r, c]
                    ds2[now, r, c] = wd_s2[n, r, c]
                    dcds[now, r, c] = wd_cds[n, r, c]
                    now += 1
    # ds2 = np.where(ds2<0, ds2+360, ds2)
    for n in range(len(N_list)):
        for r in range(15):
            for c in range(19):
                if np.logical_not(np.isnan(ds2[n, r, c])):
                    if ds2[n, r, c]<0:
                        ds2[n,r,c] += 360
                if np.logical_not(np.isnan(dcds[n, r, c])):
                    if dcds[n, r, c]<0:
                        dcds[n,r,c] += 360
    # dcds = np.where(dcds<0, dcds+360, dcds)
    return sigma, ws2, wcds, ds2, dcds, ss, tt



# standardize with the coast
def RowPlot(data, wind_d, ss, tt):
    cmap = cm.get_cmap('bwr')
    t = 1
    w_towards = 76
    nums, rows, cols = data.shape
    color_c = wind_d.copy()
    ind = np.where(color_c>(180+w_towards))
    color_c = np.abs(color_c-w_towards)
    color_c[ind] = 180 - (color_c[ind] - (180+w_towards))
    color_c = color_c/180
    for r in range(rows):
        if ss[r]<tt[r]:
            plt.figure(r)
            for n in range(nums):
                dat = data[n, r, ss[r]:tt[r] + 1]
                dat = dat[::-1]
                dat = dat[t:] - dat[t]
                plt.plot(dat, '--', color = 'grey')
                for c in range(len(dat)):
                    plt.plot(c, dat[c], '.', color = cmap(color_c[n, r, c]))
            plt.colormaps()
            plt.title('red: off shore; blue: from sea')
            plt.plot(range(8), np.zeros(8), 'g-')
    plt.show()


# difference between ecmwf and sentinel 1
def RowPlot_d(data, wind_d, ss, tt):
    cmap = cm.get_cmap('bwr')
    t = 1
    w_towards = 76
    nums, rows, cols = data.shape
    color_c = wind_d.copy()
    ind = np.where(color_c>(180+w_towards))
    color_c = np.abs(color_c-w_towards)
    color_c[ind] = 180 - (color_c[ind] - (180+w_towards))
    color_c = color_c/180
    for r in range(rows):
        if ss[r]<tt[r]:
            plt.figure(r)                                                    # figure number
            for n in range(nums):
                dat = data[n, r, ss[r]:tt[r] + 1]
                dat = dat[::-1]
                dat = dat[t:]
                plt.plot(dat, '--', color = 'grey')
                for c in range(len(dat)):
                    # if color_c[n, r, c]<0.5:
                    plt.plot(c, dat[c], '.', color = cmap(color_c[n, r, c]))
            plt.colormaps()
            plt.title('red: off shore; blue: from sea')
            plt.plot(range(8), np.zeros(8), 'g-')
    plt.show()


# def Polar_d(data, ss, tt):
#     nums, rows, cols = data.shape
#     for r in range(rows):
#         if ss[r]<tt[r]:
#             for n in range(nums):
#                 dat = data[n, r, ss[r]:tt[r]+1]
#                 m, bins, patches = plt.hist(dat, bins = list(range(0, 361, 3)), range=(0,360))




if __name__ == '__main__':
    sigma, ws2, wcds, ds2, dcds, ss, tt = eliminate('resize')
    # t_ = [2,3,8,11,12]
    # for tx in t_:
    #     tt[tx] = tt[tx] - 1




    # color: red----off shore, blue----from sea to shore

    # RowPlot(sigma, dcds, ss, tt)  # SigmaNaught
    # RowPlot(ws2, dcds, ss, tt)    # wind from sentinel 1
    # RowPlot(wcds, dcds, ss, tt)   # wind from ecmwf

    # RowPlot_d(ws2-wcds, dcds, ss, tt)            # wind speed difference
    # RowPlot_d(np.abs(ws2-wcds), dcds, ss, tt)  # wind speed difference (absolute value)
    dd = ws2 - wcds
    ind1 = np.where(dd > 180)
    ind2 = np.where(dd < -180)
    dd[ind1] = dd[ind1] - 360
    dd[ind2] = dd[ind2] + 360
    # RowPlot_d(dd, dcds, ss, tt)                # wind direction difference
    # RowPlot_d(np.abs(dd), dcds, ss, tt)        # wind direction difference


