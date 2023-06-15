""" Other neuroimaging file formats
"""
import numpy as np
import nibabel as nb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET

class Border:
    def __init__(self,name=None,vertices=None,weights=None):
        self.name= name
        self.vertices = vertices     
        self.weights = weights

    def project(self,surf):
        """Project the border onto a surface
        Barycentric coordinates are used to project the border onto the surface
        """ 
        if isinstance(surf,str):    
            surf = nb.load(surf)
        coordV = surf.agg_data('NIFTI_INTENT_POINTSET')
        coords = coordV[self.vertices,:]
        weights = self.weights / np.sum(self.weights,axis=1)[:,np.newaxis]
        weights = weights.reshape(-1,3,1)
        coords = np.sum(coords*weights,axis=1)
        return coords

    def to_xml(self):
        raise NotImplementedError
        pass

def read_borders(fname):
    """Reads a Lookuptable file

    Args:
        fname (str): Filename

    Returns:
        index (ndarray): Numerical keys
        colors (ndarray): N x 3 ndarray of colors 
        labels (list): List of labels 
    """
    borders = []
    tree = ET.parse(fname)
    root = tree.getroot()
    for bclass in root:
        for border in bclass:
            print(border.get('Name'))
            for bp in border:
                V = bp.findall('Vertices')[0]
                W = bp.findall('Weights')[0]
                vert = np.fromstring(V.text,sep=' ',dtype=int).reshape(-1,3)
                weights = np.fromstring(W.text,sep=' ').reshape(-1,3)
                borders.append(Border(border.get('Name'),vert,weights))
    return borders

if __name__=="__main__":
    wdir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fs_LR_32/'
    fname = wdir + 'fs_LR.32k.L.border'
    sname = wdir + 'fs_LR.32k.R.flat.surf.gii'
    borders = read_borders(fname)
    surf = nb.load(sname)
    coords = borders[0].project(surf)
    pass
