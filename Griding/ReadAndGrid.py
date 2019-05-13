import Griding.Coordinates as cod
# import Griding.Method2 as md2
import Griding.GridMethod as gm
import Griding.IOcontrol as jo
import Griding.LayerCalculator as Lc
import Read_SentinelData.SentinelClass as rd
import numpy as np

# import time


# use first layer (base) as standard grid and attain its lon and lat

# use bilinear interpolation to calculate the lon lat for new image
# then we get points (lon,lat) and their corresponding SigmaNaught
# apply delaunay triangulation-------------many small triangles
# figure the

root = 'D:/Academic/MPS/Internship_RSMAS/Data/cathes/GraphicMethod/'


def FirstLayer(name, dir):
    global root
    dat = rd.Data_Level1(name, dir)
    dat.OneStep()
    GCP = dat.GCPs
    SigmaNaught = dat.NRCS
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
    SigmaNaught = Lc.Resize_SigmaNaught(SigmaNaught, n)
    ny_lim, nx_lim = SigmaNaught.shape
    [sub_image, pt_list] = cod.N_sub(3, SigmaNaught)
    del SigmaNaught
    jo.Matrix_save(pt_list, 'Upper Left Points of SubImages', root)
    jo.NMatrix_save(sub_image, 'BaseLayer', root)
    del sub_image
    gm.AllLonLat('Base', x_lim, y_lim, GCP, n)
    return nx_lim, ny_lim


def OtherLayer(B_xsize, B_ysize, dir):
    d = rd.SentinelData()
    d.Get_List(dir)
    global root
    # for indice in range(len(d.series)):
    for indice in [1]:
        temp = rd.Data_Level1(d.series[indice], d.FList[indice])
        temp.OneStep()
        GCP = temp.GCPs
        SigmaNaught = temp.NRCS
        name = 'Layer' + str(indice + 1) + '-' + d.series[indice]
        del temp
        Sigma_N = gm.NewSigmaNaught5(GCP, SigmaNaught, B_ysize, B_xsize)
        del SigmaNaught, GCP
        [sub_image, pt_list] = cod.N_sub(3, Sigma_N)
        jo.NMatrix_save(sub_image, name, root)
    return None


if __name__ == '__main__':
    name = '20190220'
    calibratedpath = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST\\S1A_IW_GRDH_1SDV_20190220T020703_20190220T020729_026007_02E5FB_31B6.SAFE\\annotation\\calibration\\calibration-s1a-iw-grd-vv-20190220t020703-20190220t020729-026007-02e5fb-001.xml'
    measurepath = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST\\S1A_IW_GRDH_1SDV_20190220T020703_20190220T020729_026007_02E5FB_31B6.SAFE\\measurement\\s1a-iw-grd-vv-20190220t020703-20190220t020729-026007-02e5fb-001.tiff'
    noisepath = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST\\S1A_IW_GRDH_1SDV_20190220T020703_20190220T020729_026007_02E5FB_31B6.SAFE\\annotation\\calibration\\noise-s1a-iw-grd-vv-20190220t020703-20190220t020729-026007-02e5fb-001.xml'
    otherspath = 'D:\\Academic\\MPS\\Internship\\Data\\Sentinel\\TEST'

    # [x0,y0] = FirstLayer(name, [calibratedpath, measurepath, noisepath])
    # print(x0, y0)
    x0 = 2528
    y0 = 1702
    OtherLayer(x0, y0, otherspath)
