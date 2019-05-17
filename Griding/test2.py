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



if __name__ == '__main__':
    line = gm.OneStepCoastline()
    print(line)



