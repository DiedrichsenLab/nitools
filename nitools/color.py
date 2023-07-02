""" Other neuroimaging file formats
"""
import numpy as np
import pandas as pd

def read_lut(fname):
    """Reads a Lookuptable file

    Args:
        fname (str): Filename

    Returns:
        index (ndarray): Numerical keys
        colors (ndarray): N x 3 ndarray of colors
        labels (list): List of labels
    """
    L = pd.read_csv(fname,header=None,sep=' ',names=['ind','R','G','B','label'])
    index = L.ind.to_numpy()
    colors = np.c_[L.R.to_numpy(),L.G.to_numpy(),L.B.to_numpy()]
    labels = list(L.label)
    return index,colors,labels


def save_lut(fname,index,colors,labels):
    """Save a set of colors and labels as a LUT file

    Args:
        fname (str): File name
        index (ndarray): Numerical key
        colors (ndarray): List or RGB tuples 0-1
        labels (list): Names of categories
    """
    # Save lut file
    L=pd.DataFrame({
            "key":index,
            "R":colors[:,0].round(4),
            "G":colors[:,1].round(4),
            "B":colors[:,2].round(4),
            "Name":labels})
    L.to_csv(fname,header=None,sep=' ',index=False)


def save_cmap(fname,colors):
    """Save a set of colors to a FSLeyes compatible cmap file

    Args:
        fname (str): File name
        colors (ndarray): List or RGB tuples 0-1
    """
    # Save lut file
    L=pd.DataFrame({
            "R":colors[:,0].round(4),
            "G":colors[:,1].round(4),
            "B":colors[:,2].round(4)})
    # Save cmap file (in accordance with FSLeyes-accepted colour maps)
    L.to_csv(fname + '.cmap', header=None, sep=' ', index=False)

