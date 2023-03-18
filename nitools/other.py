"""Nitools: Other
"""
import numpy as np
import nibabel as nb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def read_lut(fname):
    L = pd.read_csv(fname,header=None,sep=' ',names=['ind','R','G','B','label'])
    index = L.ind.to_numpy()
    colors = np.c_[L.R.to_numpy(),L.G.to_numpy(),L.B.to_numpy()]
    labels = list(L.label)
    return index,colors,labels


def save_lut(fname,index,colors,labels):
    """Save a set of colors and labels as a LUT file

    Args:
        fname (_type_): File name
        index (_type_): Numerical key
        colors (_type_): List or RGB tuples 0-1
        labels (_type_): Names of categories
    """
    # Save lut file
    L=pd.DataFrame({
            "key":index,
            "R":colors[:,0].round(4),
            "G":colors[:,1].round(4),
            "B":colors[:,2].round(4),
            "Name":labels})
    L.to_csv(fname,header=None,sep=' ',index=False)

    # Save cmap file (in accordance with FSLeyes-accepted colour maps)
    L.drop('key', axis=1).drop('Name', axis=1).to_csv(fname + '.cmap', header=None, sep=' ', index=False)

