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
    # path = input('Where do you save the product?')
    path = 'F:\\Jiewei\\Sentinel-1\\Level1-GRD-IW'
    ans = int(input('Which product? Input the number:'))
    t = rd.SentinelData()
    t.Get_List(root=path)
    temp = rd.Data_Level1(t.series[ans-1], t.FList[ans-1])
    temp.OneStep()
    NRCS = temp.denoiseNRCS
    del temp, t
    data = Lc.Resize_SigmaNaught(NRCS, 30)
    Lc.Display(data, r=r)
    return None


if __name__ == '__main__':
    # line = gm.OneStepCoastline()
    # print(line)
    HaveALook(0.02)



