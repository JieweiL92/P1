import numpy as np
import Griding.LayerCalculator as Lc



dir ='D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/Temp/'
filename = 'Layer2-20190304.npy'
f1 = 'test_sub1.npy'
data = np.load(dir+filename)
Newdat = Lc.Resize_SigmaNaught(data, 3)
Lc.Display(Newdat,r=0.07)

