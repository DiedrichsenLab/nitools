""" Other neuroimaging file formats
"""
import numpy as np
import nibabel as nb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET

class Border:
    """Border class
    
    Attributes:
        name (str): Name of the border
        vertices (np.array): Vertices of the border
        weights (np.array): Weights of Barycentric coordinates
    """
    def __init__(self,name=None,vertices=None,weights=None):
        self.name= name
        self.vertices = vertices     
        self.weights = weights

    def get_coords(self,surf):
        """Gets coords the border onto a surface
        Barycentric coordinates are transformed into Eucledian coordinates

        Args:
            surf (str or nibabel): Surface file
        Returns:
            coords (np.array): Coordinates of the border on the surface
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
    """ Reads a Workbench border file
    from XML format into a list of Border objects

    Args:
        fname (str): Filename

    Returns:
        borders (list): List of border objects
    """
    borders = []
    tree = ET.parse(fname)
    root = tree.getroot()
    for bclass in root:
        for border in bclass:
            for bp in border:
                V = bp.findall('Vertices')[0]
                W = bp.findall('Weights')[0]
                vert = np.fromstring(V.text,sep=' ',dtype=int).reshape(-1,3)
                weights = np.fromstring(W.text,sep=' ').reshape(-1,3)
                borders.append(Border(border.get('Name'),vert,weights))
    return borders
