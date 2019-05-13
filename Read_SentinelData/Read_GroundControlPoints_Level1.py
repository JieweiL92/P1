# [[imx,imy,lon,lat]]
# image X: from left to right
# image Y: from down to up
# world lon (E)
# world lat (N)

import Read_SentinelData.SentinelClass as rd

if __name__ == '__main__':
    ans = input('Please enter the path you save your Sentinel-1 Data(Level1):\n')
    ds = rd.SentinelData()
    ds.Get_List(ans)
    root = input('Please enter where you want to save these files:\n')
    for i in range(len(ds.series)):
        temp = rd.Data_Level1(ds.series[i], ds.FList[i])
        temp.Get_Measure_Data()
        temp.GCPWriter(root)
    print('Done!')
