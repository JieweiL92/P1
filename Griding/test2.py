import numpy as np
import cv2

def ToGray(data):
    rows, cols = data.shape
    mm = data[~np.isnan(data)].mean()
    for r in range(rows):
        for c in range(cols):
            if data[r,c] == np.nan:
                data[r,c] = mm
    coef = 255/(data.max()-data.min())
    data = data-data.min()
    img = np.around(data/coef).astype(np.uint8)
    return img

dir ='D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/Temp/'
filename = 'Layer2-20190304.npy'
f1 = 'test_sub1.npy'
data = np.load(dir+f1)
rows, cols = data.shape
arr = ToGray(data)
# cv2.resizeWindow('Image1', cols, rows)
# cv2.imshow('Image1', arr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
mm = data[~np.isnan(data)].mean()
print(mm)
print(data[~np.isnan(data)].max())
print(data[~np.isnan(data)].min())
print(arr.max(), arr.min())
print(rows, cols)



