import csv

import Read_SentinelData.SentinelClass as rd

if __name__ == '__main__':
    ans = input('Please enter the path you save your Sentinel-1 Data(Level1):\n')
    ds = rd.SentinelData()
    ds.Get_List(ans)

    name = ds.series
    C_path, M_path = [], []
    for i in range(len(name)):
        C_path.append(ds.FList[i][0])
        M_path.append(ds.FList[i][1])

    root = input('Please enter where you want to save this path:\n')
    f = open(root + '\Path_Data.csv', 'w')
    f_csv = csv.writer(f)
    f_csv.writerows([name, C_path, M_path])
    print('Done!')
