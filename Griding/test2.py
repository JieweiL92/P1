import numpy as np
import Griding.LayerCalculator as Lc
import Griding.GridMethod as gm
import Read_SentinelData.SentinelClass as rd



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


if __name__ == '__main__':
    # line = gm.OneStepCoastline()
    # print(line)
    HaveALook(0.1)



