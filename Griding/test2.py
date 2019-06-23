import numpy as np
import math
import Griding.LayerCalculator as Lc
import Griding.GridMethod as gm
import Read_SentinelData.SentinelClass as rd
import matplotlib.pyplot as plt


dir = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST'
grid_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Grid/'
layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'
coast_root='D:/Academic/MPS/Internship/Data/coastline/'
temp_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Temp/'
file_root = 'F:/Jiewei/Sentinel-1/Level1-GRD-IW/WhiteCity/'

def TryDisplay():
    dir ='D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/Temp/'
    filename = 'Layer2-20190304.npy'
    f1 = 'test_sub1.npy'
    data = np.load(dir+filename)
    Newdat = Lc.Resize_SigmaNaught(data, 3)
    Lc.Display(Newdat,r=0.07)
    return None

def HaveALook(r):
    path = input('Where do you save the product?')
    # path = 'F:\\Jiewei\\Sentinel-1\\Level1-GRD-IW'
    ans = int(input('Which product? Input the number:'))
    t = rd.SentinelData()
    t.Get_List(root=path)
    temp = rd.Data_Level1(t.series[ans-1], t.FList[ans-1])
    temp.OneStep()
    NRCS = temp.NRCS
    del temp, t
    data = Lc.Resize_SigmaNaught(NRCS, 30)
    Lc.Display(data, r=r)
    return None



def TryMerge(name, root):
    cp = np.load(root+'Upper Left Points of Subimages.npy')
    subname = name+'_Sub'
    up, left = cp[-1,0], cp[-1,1]
    arr_list = []
    for i in range(9):
        arr_list.append(np.load(root+ subname +str(i+1)+'.npy'))
    arr = arr_list[-1]
    rows, cols = arr.shape
    rows, cols = rows+up, cols+left
    arr = np.empty([rows, cols], dtype=np.float_)
    for i in range(9):
        sub_arr = arr_list[i]
        up, left = cp[i, 0], cp[i, 1]
        r,c =sub_arr.shape
        arr[up:up + r, left:left + c] = sub_arr
    return arr


class WGS84(object):
    def __init__(self):
        self.resolution = 100            # 100 m * 100 m grid
        self.a = 6378137                 # radius in equator  (meter)
        self.b = 6356752.3142            # semi-minor axis    (meter)

    def HorizontalDeg(self, lat):
        ang = math.radians(lat)
        r = self.a*math.cos(ang)         # radius at specific latitude  (meter)
        degree_100m = math.degrees(self.resolution/r)
        return degree_100m

    def HorizontalM(self, lat):
        ang = math.radians(lat)
        r = self.a * math.cos(ang)  # radius at specific latitude  (meter)\
        # d = self.HorizontalDeg(lat)
        d = 0.00121
        ang1 = math.radians(d)
        return ang1*r

    def VerticalDeg(self, lat):
        ang = math.radians(lat)
        sita = math.atan(math.tan(ang)*self.a/self.b)
        x = self.a * math.cos(sita)
        y = self.b * math.sin(sita)
        r = math.sqrt(x**2+y**2)
        degree_100m = math.degrees(self.resolution / r)
        return degree_100m


class uniform_grid(object):
    def __init__(self):
        self.__v = 8.996e-4         # length in degrees
        self.__h = 1.211e-3         # length in degrees
        self.__size = (2100, 2820)

    Lon_Max = -122.4216
    Lon_Min = -125.8370
    Lat_Max = 43.2754
    Lat_Min = 41.3856

    def LL2CR(self, LL):
        CR = LL - np.array([Lon_Min, Lat_Min]).reshape([1, 2])
        del LL
        CR = CR/np.array([self.__h, self.__v]).reshape([1, 2])
        CR = np.floor(CR).astype(np.int16)
        TF = [True if (0<=r<2100) and (0<=c<2820) else False for r, c in CR]
        return CR, TF


if __name__ == '__main__':
    Lon_Max = -122.4216
    Lon_Min = -125.8370
    Lat_Max = 43.2754
    Lat_Min = 41.3856
    s = WGS84()
    print('Horizontal(High):', s.HorizontalDeg(Lat_Max))
    print('Horizontal(Low):', s.HorizontalDeg(Lat_Min))
    print('Vertical(High):', s.VerticalDeg(Lat_Max))
    print('Vertical(Low):', s.VerticalDeg(Lat_Min))
    print('\n')
    print('How many meters:')
    print('Horizontal(High)', s.HorizontalM(Lat_Max))
    print('Horizontal(Low):', s.HorizontalM(Lat_Min))


    horizontal_up = 0.001233
    horizontal_down = 0.00119
    vertical_up = 0.0008997
    vertical_down = 0.0008996


    h = 1.211e-3
    v = 8.996e-4
    num = [2100, 2820]
    print('\n')
    print((Lon_Max-Lon_Min)/horizontal_up)
    print((Lon_Max-Lon_Min)/horizontal_down)
    print((Lon_Max-Lon_Min)/2820)
    print('\n')
    print((Lat_Max-Lat_Min)/vertical_up)
    print((Lat_Max-Lat_Min)/vertical_down)
    print((Lat_Max-Lat_Min)/v)