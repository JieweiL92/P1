import numpy as np
import Griding.LayerCalculator as Lc
import Griding.GridMethod as gm



def TryDisplay():
    dir ='D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/Temp/'
    filename = 'Layer2-20190304.npy'
    f1 = 'test_sub1.npy'
    data = np.load(dir+filename)
    Newdat = Lc.Resize_SigmaNaught(data, 3)
    Lc.Display(Newdat,r=0.07)
    return None

def TryMerge(root = 'D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/BaseLayer_LL/'):
    root = 'D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/BaseLayer_LL/'
    cp = np.load(root+'Upper Left Points of Subimages.npy')
    lat_name = 'Base-Latitude_Sub'
    lon_name = 'Base-Longitude_Sub'
    up, left = cp[-1,0], cp[-1,1]
    lon_list, lat_list = [],[]
    for i in range(36):
        lon_list.append(np.load(root+lon_name+str(i+1)+'.npy'))
        lat_list.append(np.load(root+lat_name+str(i+1)+'.npy'))
    arr = lat_list[-1]
    rows, cols = arr.shape
    rows, cols = rows+up, cols+left
    lon_grid = np.empty([rows, cols], dtype=np.float_)
    lat_grid = np.empty_like(lon_grid)
    for i in range(36):
        lon_arr, lat_arr = lon_list[i], lat_list[i]
        up, left = cp[i, 0], cp[i, 1]
        r,c =lon_arr.shape
        lon_grid[up:up + r, left:left + c] = lon_arr
        lat_grid[up:up + r, left:left + c] = lat_arr
    return lon_grid, lat_grid


if __name__ == '__main__':
    lon_grid, lat_grid = TryMerge()
    coastXYZ = Lc.LoadCoastlineXYZ()
    gm.Line2Nodes(coastXYZ, lon_grid, lat_grid)



