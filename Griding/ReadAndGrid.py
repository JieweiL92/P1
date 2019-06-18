import Internship_RSMAS.Griding.Coordinates as cod
import Internship_RSMAS.Griding.GridMethod as gm
import Internship_RSMAS.Griding.IOcontrol as jo
import Internship_RSMAS.Griding.LayerCalculator as Lc
import Internship_RSMAS.Read_SentinelData.SentinelClass as rd
import numpy as np
from datetime import datetime
import time, os, math


# use first layer (base) as standard grid and attain its lon and lat

# use bilinear interpolation to calculate the lon lat for new image
# then we get points (lon,lat) and their corresponding SigmaNaught
# apply delaunay triangulation-------------many small triangles
# figure the

level1_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/'
layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'
temp_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Temp/'
file_root = 'F:/Jiewei/Sentinel-1/Level1-GRD-IW/WhiteCity/'

def FirstLayer(name, dir):
    dat = rd.Data_Level1(name, dir)
    dat.OneStep()
    GCP = dat.GCPs
    SigmaNaught = dat.denoiseNRCS
    y_lim, x_lim = SigmaNaught.shape
    del dat

    print('Do you want to expand to base map?\nPlease input an integer:')
    ans = input(' 0: No, I don''t want to\n n: grids will be 1+n\n')
    n = int(ans)
    if n != 0:
        print('expanding')
        [x_vec, y_vec, lon, lat] = cod.ExpandGrid(GCP, ans)
        [GCP, x_limN, y_limN] = cod.GCP2matrix(x_vec, y_vec, lon, lat)
        SigmaN = np.full([y_limN, x_limN], fill_value=np.nan)
        SigmaN[0:y_lim, 0:x_lim] = SigmaNaught
        SigmaNaught = SigmaN
        del x_limN, y_limN, SigmaN

    n = 10
    # SigmaNaught = Lc.Resize_SigmaNaught(SigmaNaught, n)
    # ny_lim, nx_lim = SigmaNaught.shape
    # [sub_image, pt_list] = cod.N_sub(3, SigmaNaught)
    # del SigmaNaught
    # jo.Matrix_save(pt_list, 'Upper Left Points of SubImages', layer_root)
    # jo.NMatrix_save(sub_image, 'BaseLayer', layer_root)
    # del sub_image
    ny_lim, nx_lim = math.floor(y_lim / n), math.floor(x_lim / n)
    gm.AllLonLat('Base', x_lim, y_lim, GCP, n)
    return nx_lim, ny_lim


def OtherLayer(B_xsize, B_ysize, d, n):
    for indice in range(n, len(d.series)):
        temp = rd.Data_Level1(d.series[indice], d.FList[indice])
        temp.OneStep()
        GCP = temp.GCPs
        SigmaNaught = temp.denoiseNRCS
        name = 'Layer' + str(indice) + '-' + d.series[indice]
        del temp
        y1, x1 = SigmaNaught.shape
        # gm.AllLonLat(name, x1, y1, GCP, 3)
        start = time.time()
        # Sigma_N = gm.NewSigmaNaught5(GCP, SigmaNaught, B_ysize, B_xsize)
        # print('Time for resampling 1 SAR image:', (time.time()-start)/60, 'mins')
        # del SigmaNaught, GCP
        # [sub_image, pt_list] = cod.N_sub(3, Sigma_N)
        # jo.NMatrix_save(sub_image, name, layer_root)
        SigmaNaught = Lc.Resize_SigmaNaught(SigmaNaught, 5)
        Sigma_N, nums_arr = gm.GridSigmaNaught(SigmaNaught)
        print('Time for resampling 1 SAR image:', (time.time()-start)/60, 'mins')
        jo.Matrix_save(Sigma_N, name, layer_root)
    return None


def GenerateMatrix(rows = 1671, cols = 2549):
    file_list = os.listdir(layer_root)
    file_set = set()
    for s in file_list:
        n = s.find('-')
        if n>0:
            file_set.add(s[:n+9])
    dicts = {}
    for name in file_set:
        arr = gm.Merge(name)
        date_str = name[-8:]
        dates = datetime.strptime(date_str, '%Y%m%d')
        dicts[dates] = arr
    data = np.empty([len(file_set), rows, cols], dtype=np.float32)
    datelist = sorted(dicts.keys())
    n = 0
    for s in datelist:
        data[n,:,:] = dicts[s]
        n += 1
    return data, datelist



if __name__ == '__main__':

    # Grid
    d = rd.SentinelData()
    d.Get_List(file_root)
    # start from 20170216
    # [x0, y0] = FirstLayer(d.series[2], d.FList[2])
    x0, y0 = 2549, 1671
    OtherLayer(x0, y0, d, 0)


    # # Merge and Overlap
    # data, date_list = GenerateMatrix()
    # # t = Lc.Resize_SigmaNaught(data[0,:,:], 2)
    # # ynew, xnew = t.shape
    # # resize_data = np.empty([73, ynew, xnew], dtype=np.float32)
    # # resize_data[0,:,:] = t
    # # for i in range(1,73):
    # #     print(i)
    # #     t = Lc.Resize_SigmaNaught(data[i,:,:], 2)
    # #     resize_data[i,:,:] = t
    # # np.save(level1_root+'data_resize.npy', resize_data)
    # np.save(level1_root+'data.npy', data)
    # del data
    # with open(level1_root+'date_list(matrix).txt', 'w') as fid:
    #     for dts in date_list:
    #         s = dts.strftime('%Y%m%d')
    #         fid.write(s)
    #         fid.write('\n')
    # print('finished')