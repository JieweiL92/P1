import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def ReadUpwellingIndex():
    root = 'F:/Jiewei/Upwelling Index(MontereyBay)/'
    fname = 'upwell36N122W.txt'
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

def ReadWindData(path):
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


if __name__ == '__main__':
    Tseries, off_shore, along_shore = ReadUpwellingIndex()
    st = Tseries.index(datetime(2017,1,1))
    x = np.array(Tseries[st:])
    y1 = np.array(off_shore[st:])
    y2 = np.array(along_shore[st:])
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'b')
    plt.show()