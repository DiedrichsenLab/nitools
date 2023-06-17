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

def read_borders(fname):
    """ Reads a Workbench border file
    from XML format into a list of Border objects

    Args:
        fname (str): Filename

    Returns:
        borders (list): List of border objects
        binfo (dict): Dictionary of border-file information
    """
    borders = []
    tree = ET.parse(fname)
    root = tree.getroot()
    for bclass in root:
        for bn,border in enumerate(bclass):
            for i,bp in enumerate(border):
                V = bp.findall('Vertices')[0]
                W = bp.findall('Weights')[0]
                vert = np.fromstring(V.text,sep=' ',dtype=int).reshape(-1,3)
                weights = np.fromstring(W.text,sep=' ').reshape(-1,3)
                bb=Border(border.get('Name'),vert,weights)
                bb.num = bn
                bb.partnum = i
                borders.append(bb)
    return borders,root.attrib

def save_borders(borders,binfo,fname):
    root = ET.Element("BorderFile",attrib=binfo)
    bclass = ET.SubElement(root, "Class", attrib={'Name':"Class1",
                                                  "Red":"0",
                                                  "Green":"0",
                                                  "Blue":"0"})
    current_name = ''
    for b in borders:
        if b.name !=current_name:
            bo=ET.SubElement(bclass, "Border", attrib={'Name':b.name,
                                                  "Red":"0",
                                                  "Green":"0",
                                                  "Blue":"0"})
            current_name = b.name  
        bp=ET.SubElement(bo, "BorderPart", attrib={'Closed':'False'})
        V=ET.SubElement(bp, "Vertices")
        V.text = np.array2string(b.vertices).replace('[','')
        V.text = V.text.replace(']','')
        W=ET.SubElement(bp, "Weights")
        W.text = np.array2string(b.weights).replace('[','')
        W.text = W.text.replace(']','')

    tree = ET.ElementTree(root)
    tree.write(fname)