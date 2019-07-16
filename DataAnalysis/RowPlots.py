import Internship_RSMAS.Read_SentinelData.SentinelClass as rd
import Internship_RSMAS.Griding.LayerCalculator as Lc
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



# load wind speed, wind direction from ECMWF and Sentinel 1, and SigmaNaught
def eliminate():
    N_list=list(range(38, 72))
    N_list.remove(39)
    N_list.remove(58)
    N_list.remove(69)
    temp = rd.Grid_data()
    temp.LoadData(ty = 'resize')
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
    ds2 = np.where(ds2<0, ds2+360, ds2)
    dcds = np.where(dcds<0, dcds+360, dcds)
    return sigma, ws2, wcds, ds2, dcds, ss, tt


def RowPlot(data, ss, tt):
    cmap = cm.get_cmap('jet')
    t = 1
    for r in range(12):
        plt.figure(r)
        for n in range(31):
            dat = data[n, r, ss[r]:tt[r] + 1]
            dat = dat[::-1]
            plt.plot(dat[t:] - dat[t], '--', color=cmap(np.mean(dcds[n, r, :tt[r]]) / 360))
        plt.colormaps()
        plt.plot(range(8), np.zeros(8), 'r-')
    plt.show()


if __name__ == '__main__':
    sigma, ws2, wcds, ds2, dcds, ss, tt = eliminate()
    RowPlot(sigma, ss, tt)