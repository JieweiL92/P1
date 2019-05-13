import csv
import gdal
import os
import sys
import time
import xml.etree.ElementTree as ET

import netCDF4 as ncdf
import numpy as np
import numpy.ma as ma
from numba import jit


def Bilinear(c1, c2, r1, r2, n11, n21, n12, n22):
    @jit(nopython=True, parallel=True)
    def LinerInterp(x, x1, x2, f1, f2):
        return ((x2 - x) * f1 + (x - x1) * f2) / (x2 - x1)

    @jit(nopython=True, parallel=True)
    def BilinearInterp(x, y, x1, x2, y1, y2, f11, f21, f12, f22):  # f(x1,y1), f(x1,y2)
        ty1 = LinerInterp(x, x1, x2, f11, f21)
        ty2 = LinerInterp(x, x1, x2, f12, f22)
        t = LinerInterp(y, y1, y2, ty1, ty2)
        return t

    @jit
    def Travel(c1, c2, r1, r2, n11, n21, n12, n22):
        arr = np.empty([r2 - r1 + 1, c2 - c1 + 1], dtype=np.float32)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                arr[r - r1, c - c1] = BilinearInterp(c, r, c1, c2, r1, r2, n11, n21, n12, n22)
        return arr

    s = Travel(c1, c2, r1, r2, n11, n21, n12, n22)
    return s


class SentinelData(object):
    def __init__(self):
        self.__FList = []
        self.__series = []

    @property
    def FList(self):
        return self.__FList

    @property
    def series(self):
        return self.__series

    def Get_List(self, root=os.getcwd()):
        root_list = os.listdir(root)
        C = '\\annotation\\calibration\\'
        M = '\\measurement\\'
        for s in root_list:
            if s.find('1SDV') >= 0 and s.find('SAFE') >= 0:
                s0 = root + '\\' + s
                s1 = s[17:25]
                C1 = s0 + C
                C2 = s0 + C
                M1 = s0 + M
                Cdir = os.listdir(C1)
                for t in Cdir:
                    if t[0:11] == 'calibration' and t.find('vv') >= 0:
                        C1 = C1 + t
                        break
                for t in Cdir:
                    if t[0:5] == 'noise' and t.find('vv') >= 0:
                        C2 = C2 + t
                        break
                Mdir = os.listdir(M1)
                for t in Mdir:
                    if t.find('vv') >= 0 and t[-5:] == '.tiff':
                        M1 = M1 + t
                self.__series.append(s1)
                self.__FList.append([C1, M1, C2])  # calibration file, measurement file, noise

    def Get_List_NetCDF(self, root=os.getcwd()):
        root_list = os.listdir(root)
        cuts = '\\measurement\\'
        for s in root_list:
            if s.find('OCN') >= 0 and s.find('SAFE') >= 0:
                paths = root + '\\' + s + cuts
                names = s[17:25]
                nc_file = os.listdir(paths)[0]
                self.__series.append(names)
                self.__FList.append(paths + nc_file)


class Data_Level1(object):  # 单独一份 level1 数据
    def __init__(self, name_x, f_address):
        self.__name = name_x  # name,             str
        self.__addr = f_address  # file address,     [c,m]
        self.__DNs = []  # from measurement, ndarray
        self.__GCPs = []  # from measurement, [[x,y,lon,lat][][][][]]
        self.__NRCS = []  # ndarray
        self.__K = 0  # from calibration, k
        self.__PixelRange = []  # from calibration, [0,40,60,,,,]
        self.__SigmaA = []  # from calibration, [6.28e3,6.27e3,,,,]
        self.__noise = []

    @property
    def name(self):
        return self.__name

    @property
    def addr(self):
        return self.__addr

    @property
    def DNs(self):
        return self.__DNs

    @property
    def GCPs(self):
        return self.__GCPs

    @property
    def NRCS(self):
        return self.__NRCS

    @property
    def K(self):
        return self.__K

    @property
    def PixelRrange(self):
        return self.__PixelRange

    @property
    def SigmaA(self):
        return self.__SigmaA

    @property
    def noise(self):
        return self.__noise

    def Get_Calibrated_Data(self):
        fid = ET.parse(self.__addr[0])  # No.n file and the calibration data [[c,m][]]
        f_root = fid.getroot()
        self.__K = float(f_root[1][0].text)  # absolute calibration constant
        # go to the calibration vector list
        C_vector = f_root[2]
        C_amount = int(C_vector.attrib['count'])  # how many calibration vector
        LutPixel, LutSigma0 = [], []  # look up table

        for i in range(C_amount):
            root = C_vector[i]  # <calibration vector>
            PixelS, SigmaS = root[2].text, root[3].text
            # cut and convert the value
            LutPixel.append(list(map(int, PixelS.split())))
            LutSigma0.append(list(map(float, SigmaS.split())))

        if self.Check_Data(LutPixel):
            self.__PixelRange = LutPixel[0]
        else:
            print('Errors in Calibration Data %s: Pixel Range' % self.__name)
        if self.Check_Data(LutSigma0):
            self.__SigmaA = LutSigma0[0]
        else:
            print('Errors in Calibration Data %s: SigmaA' % self.__name)

    def Get_Noise_Data(self):
        fid = ET.parse(self.__addr[2])  # No.n file and the calibration data [[c,m][]]
        f_root = fid.getroot()
        AzimuthVectorList = f_root[2]
        num = int(AzimuthVectorList.attrib['count'])

        rows, cols = self.__DNs.shape
        noise_arr = np.zeros([rows, cols], dtype=np.float32)
        for i in range(num):
            sub_dir = AzimuthVectorList[i]
            c1, c2 = int(sub_dir[2].text), int(sub_dir[4].text)
            lines = sub_dir[5].text
            noises = sub_dir[6].text
            line_num = rows - 1 - np.array(list(map(int, lines.split())))
            noises_s = list(map(float, noises.split()))
            lines = line_num.tolist()
            lines.reverse()
            noises_s.reverse()
            for j in range(len(lines) - 1):
                y1, y2 = lines[j], lines[j + 1]
                v1, v2 = noises_s[j], noises_s[j + 1]
                for y in range(y1, y2 + 1):
                    noise_arr[y, c1:c2 + 1] = ((y2 - y) * v1 + (y - y1) * v2) / (y2 - y1)
        self.__noise = noise_arr

    def Get_Measure_Data(self):
        ds = gdal.Open(self.__addr[1])
        ds_band = ds.GetRasterBand(1)
        self.__DNs = ds_band.ReadAsArray().astype('uint16')  # DN
        CPs = ds.GetGCPs()  # Ground Control Point
        arr1 = []
        arr1_append = arr1.append
        for p in CPs:
            tempA = [p.GCPPixel, p.GCPLine, p.GCPX, p.GCPY]  # [x,y,lon,lat]  [ImX, ImY, lon, lat]
            arr1_append(tempA)  # world lat (N)
        self.__GCPs = arr1

    def Check_Data(self, arr):  # return True of False
        first = arr[0]
        subtract_matrix = []
        for j in arr:
            subtract_matrix.append(list(map(lambda x, y: abs(x - y), j, first)))
        critic = list(map(sum, subtract_matrix))
        Pd = True
        for j in critic:
            if j > 0.0:
                Pd = False
        return Pd

    def CalNRCS(self):  # require to obtain measure and calibrated data first
        Mat = self.__DNs.astype('float32')
        sigmaA_Range = np.zeros(self.__PixelRange[-1] + 1)
        for i in range(len(self.__PixelRange) - 1):
            x1, x2 = self.__PixelRange[i], self.__PixelRange[i + 1]
            y1, y2 = self.__SigmaA[i], self.__SigmaA[i + 1]
            for xi in range(x1, x2):
                sigmaA_Range[xi] = (x2 - xi) * y1 / (x2 - x1) + (xi - x1) * y2 / (x2 - x1)
        sigmaA_Range[-1] = self.__SigmaA[-1]
        Mat = (Mat / sigmaA_Range) ** 2 - self.__noise / (sigmaA_Range * sigmaA_Range)
        self.__NRCS = np.array(Mat, dtype=np.float32)
        del Mat

    def OneStep(self):
        st = time.time()
        self.Get_Calibrated_Data()
        st1 = time.time()
        self.Get_Measure_Data()
        st2 = time.time()
        self.Get_Noise_Data()
        st3 = time.time()
        self.CalNRCS()
        st4 = time.time()
        print('Calibrate Data:', st1 - st, 'seconds')
        print('Measurement:', st2 - st1, 'seconds')
        print('Noise Data:', st3 - st2, 'seconds')
        print('Calculate NRCS:', st4 - st3, 'seconds')

    def LUTWriter(self, root='D:/Academic/MPS/Internship/Data/Sentinel/Level 1/Look up table'):
        with open(root + '/LUT' + self.__name + '.csv', 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows([self.__PixelRange, self.__SigmaA])

    def GCPWriter(self, root='D:/Academic/MPS/Internship/Data/Sentinel/Level 1/Ground Control Point'):
        with open(root + '/GCP' + self.__name + '.csv', 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(self.__GCPs)


class Data_Level2(object):
    def __init__(self, name_x, f_address):
        self.__name = name_x  # name,             str
        self.__addr = f_address  # file address,     str
        self.__lon = []  # 经度               masked array [azimuth, range]
        self.__lat = []  # 纬度               masked array [azimuth, range]
        self.__windX = []  # 90 degrees         masked array [azimuth, range] m/s
        self.__windY = []  # 0 degrees          masked array [azimuth, range] m/s
        self.__speed = []  # wind speed         masked array [azimuth, range] m/s
        self.__direct = []  # wind direction     masked array [azimuth, range] degrees
        self.__mask = []  # mask for owi       0: useful; 1: land; 2: ice; 4: missing   NOT the mask in masked array

    # For MaskedArray: arr
    # use arr.data to get value
    # use arr.mask to get the mask(bool), True(1) for cover
    # use arr.fillvalue to get the filled value

    @property
    def name(self):
        return self.__name

    @property
    def addr(self):
        return self.__addr

    @property
    def lon(self):
        return self.__lon

    @property
    def lat(self):
        return self.__lat

    @property
    def windX(self):
        return self.__windX

    @property
    def windY(self):
        return self.__windY

    @property
    def speed(self):
        return self.__speed

    @property
    def direct(self):
        return self.__direct

    @property
    def mask(self):
        return self.__mask

    def Get_WindData(self):
        DataSet = ncdf.Dataset(self.__addr)
        v = DataSet.variables
        self.__speed = v['owiWindSpeed'][:]
        self.__direct = v['owiWindDirection'][:]
        self.__mask = v['owiMask'][:].data
        self.__lon = v['owiLon'][:]
        self.__lat = v['owiLat'][:]

    def CalWindVector(self):
        DMask = self.__speed.mask
        w_speed = self.__speed.data
        w_direct = self.__direct.data
        vectorX = w_speed * np.sin(w_direct * np.pi / 180)
        vectorY = w_speed * np.cos(w_direct * np.pi / 180)
        self.__windX = ma.array(vectorX, mask=DMask, fill_value=-999)
        self.__windY = ma.array(vectorY, mask=DMask, fill_value=-999)


def WriteData(root):
    d = SentinelData()
    d.Get_List(root)
    for t in range(len(d.series)):
        temp = Data_Level1(d.series[t], d.FList[t])
        temp.CalNRCS()
        temp.Writer()
        sys.stdout.write('Writing%.3f%%' % float(t / len(d.series)) * 100 + '\r')
        sys.stdout.flush()
    print('Done!')
    return


if __name__ == '__main__':
    f_root = input('Please input the file you store Sentinel-1 level 1 product:\n')
    WriteData(f_root)
