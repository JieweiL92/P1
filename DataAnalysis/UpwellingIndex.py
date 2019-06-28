from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

root = 'D:/Academic/MPS/Internship/Data/'



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

def DrawFig():
    tim, off, alon = ReadUpwellingIndex()
    s1 = tim.index(datetime(2017,1,11,12))
    t = tim[s1::48]
    o = off[s1::48]
    a = alon[s1::48]
    plt.plot(t, o)
    plt.plot(t, a, '--')
    plt.plot(t, np.zeros(len(t)))
    for tt in range(len(t)):
        plt.annotate(tt, xy=(t[tt], o[tt]))
    plt.title('Upwelling Index(tag:off shore data)')
    # plt.savefig(root+'upwelling.png')
    # plt.close()
    return None



if __name__ == '__main__':
    DrawFig()