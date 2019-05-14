#  for layers

import codecs
import json

import numpy as np
from numba import jit


@jit
def Matrix_save(arr, name, root='D:/Academic/MPS/Internship/Data/cathes/'):
    np.save(root + name + '.npy', arr)


@jit
def Matrix_load(name, root='D:/Academic/MPS/Internship/Data/cathes/'):
    arr = np.load(root + name + '.npy')
    return arr


def Pts_save(arr, name, root='D:/Academic/MPS/Internship/Data/cathes/'):
    # pickling
    b = arr.tolist()
    with codecs.open(root + name + '.dat', 'w', encoding='utf-8') as fid:
        json.dump(b, fid, separators=(',', ':'), sort_keys=True, indent=4)


@jit
def Pts_read(name, root='D:/Academic/MPS/Internship/Data/cathes/'):
    # unpickling
    obj_text = codecs.open(root + name + '.dat', 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    a_new = np.array(b_new)
    return a_new


@jit
def NMatrix_save(arr, name, root='D:/Academic/MPS/Internship/Data/cathes/'):
    n = 0
    for a in arr:
        n += 1
        Matrix_save(a, name + '_Sub' + str(n), root)
    return None

@jit
def LoadCoastlineGridded(root ='D:/Academic/MPS/Internship/Data/coastline/'):
    arr = np.load(root+'Coastline in Grid.npy')
    return arr