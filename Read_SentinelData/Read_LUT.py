import Read_SentinelData.SentinelClass as rd

if __name__ == '__main__':
    ans = input('Please enter the path you save your Sentinel-1 Data(Level1):\n')
    ds = rd.SentinelData()
    ds.Get_List(ans)
    for i in range(len(ds.series)):
        temp = rd.Data_Level1(ds.series[i], ds.FList[i])
        temp.Get_Calibrated_Data()
        temp.LUTWriter()
    print('Done!')
