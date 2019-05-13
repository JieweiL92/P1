import csv

import Read_SentinelData.SentinelClass as rd
import numpy.ma as ma

if __name__ == '__main__':
    root = input('Please enter the path you save your Sentinel-1 Data(Level2):\n')
    d = rd.SentinelData()
    d.Get_List_NetCDF(root)
    root = input('Where do you want to save the wind vector data:\n')
    ans = input('Do you want the wind speed and wind direction data? Y/N\n')
    for t in range(len(d.series)):
        temp = rd.Data_Level2(d.series[t], d.FList[t])
        temp.Get_WindData()
        temp.CalWindVector()
        # outcome is a list=[lon, lat, mask, vectorX, vectorY, wind speed, wind direction] speed and direction are optional
        # these are all matrix in the same format and they have a same fii value -999
        a = [temp.lon.data, temp.lat.data, temp.mask, temp.windX.data, temp.windY.data]
        if ans == 'Y':
            a.append(temp.speed.data)
            a.append(temp.direct.data)
        with open(root + '/WindField' + d.series[t] + '.csv', 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(a)
    print('Done!')
