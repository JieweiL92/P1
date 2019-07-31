import Internship_RSMAS.Read_SentinelData.SentinelClass as rd
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

layer_root = 'D:/Academic/MPS/Internship/Data/Sentinel/Level1/Layer/'



def standard(data, rmap, tt, mask):
    rows, cols = data.shape
    data[data<=0] = np.nan
    base = np.zeros_like(data)
    for r in range(rows):
        for c in range(cols):
            base[r, c] = data[rmap[r, c], tt[rmap[r, c]]]
    mask = mask.astype(np.float_)
    mask[mask==0] = np.nan
    result = (data - base) * mask
    return result


def shows(data):
    dpi = 200
    nums, rows, cols = data.shape
    fig = plt.figure(figsize=(cols/dpi, rows/dpi), dpi = dpi)
    ax = fig.add_subplot(111)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    frame.set_axis_off()
    for n in range(nums):
        ax.imshow(data[n, :, :], cmap = 'gray', animated = True)
    return None


if __name__ == '__main__':
    temp = rd.Grid_data()
    temp.LoadData()
    tt = temp.mask[0]
    mask = temp.mask[1].astype(np.float_)
    del temp
    dmap = np.zeros_like(mask, dtype=np.float_)
    position_map = np.zeros_like(dmap)
    rows, cols = mask.shape
    rowmap, colmap = np.mgrid[0:rows, 0:cols]
    for r in range(rows):
        dmap[r, :] = np.abs(tt[r] - colmap[r, :])**2
        position_map[r, :] = r
    for r in range(rows):
        c = tt[r]
        dd = (rowmap-r)**2 + (colmap-c)**2
        ind = np.where(dmap>dd)
        dmap[ind] = dd[ind]
        position_map[ind] = r
        print(r)
    dmap = np.sqrt(dmap)
    mask = mask.astype(np.float_)
    mask[mask==0] = np.nan
    dmap = dmap*mask
    position_map = position_map * mask
    np.save('D:/Academic/MPS/Internship/Data/Process/distance.npy', dmap)
    np.save('D:/Academic/MPS/Internship/Data/Process/row_no.npy', position_map)

    # dmap = np.load('D:/Academic/MPS/Internship/Data/Process/distance.npy')
    # rmap = np.load('D:/Academic/MPS/Internship/Data/Process/row_no.npy')
    # rmap[np.isnan(rmap)] = 1
    # rmap = rmap.astype(np.int_)
    # temp = rd.Grid_data()
    # temp.LoadData()
    # tt = temp.mask[0]
    # mask = temp.mask[1]
    # del temp
    # s1 = datetime(2017,1,11)
    # s2 = datetime(2019,5,25)
    # datelist = [s1]
    # nows = s1
    # while nows != s2:
    #     nows = nows + timedelta(days=12)
    #     datelist.append(nows)
    # dates = [datetime.strftime(s, '%Y%m%d') for s in datelist]
    # dates = np.array(dates)
    # file_list = os.listdir(layer_root)
    # arrs = []
    # for ds in dates:
    #     for s in file_list:
    #         if s.find(ds) > 0 and s.find('distribution') < 0:
    #             temp = s
    #     arrs.append(np.load(layer_root + temp))
    # arrs = np.array(arrs)
    # nums, rows, cols = arrs.shape
    # print(nums, rows, cols)
    # for n in range(nums):
    #     print(n)
    #     re = standard(arrs[n, :, :], rmap, tt, mask)
    #     arrs[n, :, :] = re
    # np.save('D:/Academic/MPS/Internship/Data/Process/NRCS(std with coast).npy', arrs)
    # plt.imshow(arrs[3, :, :])
    # plt.show()


