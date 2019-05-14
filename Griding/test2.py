import numpy as np
import cv2

def ToGray(data):
    min_d = np.nanmin(data)
    max_d = np.nanmax(data)
    data[np.isnan(data)] = min_d
    coef = 255/(max_d-min_d)
    data_new = data-min_d
    img = np.around(data_new/coef).astype(np.uint8)
    return img

dir ='D:/Academic/MPS/Internship/Data/cathes/GraphicMethod/Temp/'
filename = 'Layer2-20190304.npy'
f1 = 'test_sub1.npy'
data = np.load(dir+f1)
rows, cols = data.shape
min_d = np.nanmin(data)
max_d = np.nanmax(data)
coef = 255 / (max_d - min_d)
data_new = data - min_d
arr = np.around(data_new*coef).astype(np.uint8)
# cv2.resizeWindow('Image1', cols, rows)
# cv2.imshow('Image1', arr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(np.nanmax(data), np.nanmin(data))
print(arr.max(), arr.min())


